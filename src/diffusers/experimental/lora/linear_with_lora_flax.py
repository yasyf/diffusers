import copy
import dataclasses
from collections import defaultdict
from typing import Dict, List, Type, Union, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from diffusers.models.modeling_flax_utils import FlaxModelMixin
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import zeros
from flax.traverse_util import flatten_dict, unflatten_dict


def replace_module(parent, old_child, new_child):
    for k, v in parent.__dict__.items():
        if isinstance(v, nn.Module) and v.name == old_child.name:
            object.__setattr__(parent, k, new_child)
        elif isinstance(v, tuple):
            for i, c in enumerate(v):
                if isinstance(c, nn.Module) and c.name == old_child.name:
                    object.__setattr__(parent, k, v[:i] + (new_child,) + v[i + 1 :])

    parent._state.children[old_child.name] = new_child


class FlaxLinearWithLora(nn.Module):
    features: int
    in_features: int = -1
    rank: int = 5
    scale: float = 1.0
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        linear = nn.Dense(features=self.features, use_bias=self.use_bias, name="linear")
        lora_down = nn.Dense(features=self.rank, use_bias=False, name="lora_down")
        lora_up = nn.Dense(features=self.features, use_bias=False, kernel_init=zeros, name="lora_up")

        return linear(inputs) + lora_up(lora_down(inputs)) * self.scale


class FlaxLoraBase(nn.Module):
    @staticmethod
    def _get_children(model: nn.Module) -> Dict[str, nn.Module]:
        model._try_setup(shallow=True)
        return {k: v for k, v in model._state.children.items() if isinstance(v, nn.Module)}

    @staticmethod
    def _wrap_dense(params: dict, parent: nn.Module, model: Union[nn.Dense, nn.Module], name: str):
        if not isinstance(model, nn.Dense):
            return params, {}

        params_to_optimize = defaultdict(dict)

        # model.parent._state.in_setup = True
        # model.parent._state.setup_called = SetupState.TRANSFORMED

        # object.__setattr__(lora, "parent", model.parent)
        # setattr(parent, name, lora)
        # for k, v in parent.__dict__.items():
        #     if isinstance(v, nn.Module) and v.name == name:
        #         setattr(model.parent, k, lora)
        # lora.__post_init__()

        # model.parent._state.setup_called = SetupState.DONE
        # model.parent._state.in_setup = False

        parent._state.is_initialized = False
        parent._state.in_setup = True
        lora = FlaxLinearWithLora(
            in_features=jnp.shape(params["kernel"])[0],
            features=model.features,
            use_bias=model.use_bias,
            name=name,
            parent=None,
        )
        object.__setattr__(lora, "parent", model.parent)
        object.__setattr__(lora, "scope", model.scope)

        # lora_params = lora.init_weights(jax.random.PRNGKey(0)).unfreeze()["params"]
        # print("lora_params", lora_params)
        lora_params = {}
        lora_params["linear"] = params
        lora_params["lora_down"] = {
            "kernel": jax.random.normal(jax.random.PRNGKey(0), (lora.in_features, lora.rank)) * 1.0 / lora.rank
        }
        lora_params["lora_up"] = {"kernel": jnp.zeros((lora.rank, lora.features))}
        lora = lora.bind({"params": lora_params})

        replace_module(parent, model, lora)
        lora.__post_init__()

        parent._state.in_setup = False
        parent._state.is_initialized = True

        for n in ["lora_up", "lora_down"]:
            params_to_optimize[n] = {k: True for k in lora_params[n].keys()}
        params_to_optimize["linear"] = {k: False for k in lora_params["linear"].keys()}

        return lora_params, dict(params_to_optimize)

    @staticmethod
    def inject(
        params: Union[dict, FrozenDict],
        model: nn.Module,
        targets: List[str],
        is_target: bool = False,
    ):
        model = model.bind({"params": params})
        if hasattr(model, "init_weights"):
            model.init_weights(jax.random.PRNGKey(0))

        params = params.unfreeze() if isinstance(params, FrozenDict) else copy.copy(params)
        params_to_optimize = {}

        for name, child in FlaxLoraBase._get_children(model).items():
            if is_target:
                results = FlaxLoraBase._wrap_dense(params.get(name, {}), model, child, name)
            elif child.__class__.__name__ in targets:
                results = FlaxLoraBase.inject(params.get(name, {}), child, targets=targets, is_target=True)
            else:
                results = FlaxLoraBase.inject(params.get(name, {}), child, targets=targets)

            params[name], params_to_optimize[name] = results

        return params, params_to_optimize


class LoRA:
    pass


def wrap_in_lora(model: Type[nn.Module], targets: List[str], instance=None):
    class _FlaxLora(model, LoRA):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def wrap(self):
            for n, attr in self._state.children.items():
                if not isinstance(attr, nn.Module):
                    continue
                if isinstance(attr, LoRA):
                    continue

                if self.__class__.__name__ in targets and isinstance(attr, nn.Dense):
                    instance = FlaxLinearWithLora(
                        features=attr.features,
                        use_bias=attr.use_bias,
                        name=attr.name,
                        parent=None,
                    )
                else:
                    subattrs = {f.name: getattr(attr, f.name) for f in dataclasses.fields(attr) if f.init}
                    subattrs["parent"] = None
                    klass = wrap_in_lora(attr.__class__, instance=attr, targets=targets)
                    instance = klass(**subattrs)

                object.__setattr__(instance, "parent", attr.parent)
                object.__setattr__(instance, "scope", attr.scope)

                replace_module(self, attr, instance)

        def setup(self):
            super().setup()
            self.wrap()

    _FlaxLora.__name__ = f"{model.__name__}Lora"
    _FlaxLora.__annotations__ = model.__annotations__
    return _FlaxLora


def FlaxLora(model: Type[nn.Module], targets=["FlaxAttentionBlock", "FlaxTransformer2DModel", "FlaxGEGLU"]):
    targets = targets + [f"{t}Lora" for t in targets]

    class _LoraFlax(wrap_in_lora(model, targets=targets)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            instance, params = cast(Type[FlaxModelMixin], model).from_pretrained(*args, **kwargs)

            params, mask = FlaxLoraBase.inject(params, instance, targets=targets)

            subattrs = {f.name: getattr(instance, f.name) for f in dataclasses.fields(instance) if f.init}
            instance = cls(**subattrs)

            mask_values = flatten_dict(mask)
            object.__setattr__(
                instance,
                "get_mask",
                lambda params: unflatten_dict(
                    {k: mask_values.get(k, False) for k in flatten_dict(params, keep_empty_nodes=True).keys()}
                ),
            )
            return instance, params

    _LoraFlax.__name__ = f"{model.__name__}WithLora"
    return _LoraFlax
