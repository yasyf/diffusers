import copy
import dataclasses
from collections import defaultdict
from typing import Dict, List, Type, Union, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from diffusers.models.modeling_flax_utils import FlaxModelMixin
from flax.core.frozen_dict import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict


class FlaxLinearWithLora(nn.Module):
    out_features: int
    rank: int = 5
    in_features: int = 1
    scale: float = 1.0
    use_bias: bool = True

    def setup(self):
        self.linear = nn.Dense(features=self.out_features, use_bias=self.use_bias)
        self.lora_up = nn.Dense(features=self.out_features, use_bias=False)
        self.lora_down = nn.Dense(features=self.rank, use_bias=False)

    def init_weights(self, rng: jax.random.PRNGKey) -> FrozenDict:
        return self.init(rng, jnp.zeros((self.in_features, self.out_features)))

    def __call__(self, input):
        return self.linear(input) + self.lora_up(self.lora_down(input)) * self.scale


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
            out_features=model.features,
            use_bias=model.use_bias,
            name=name,
        )
        object.__setattr__(lora, "parent", model.parent)

        lora_params = lora.init_weights(jax.random.PRNGKey(0)).unfreeze()["params"]
        lora_params["linear"] = params
        lora = lora.bind({"params": lora_params})

        for k, v in parent.__dict__.items():
            if isinstance(v, nn.Module) and v.name == name:
                setattr(model.parent, k, lora)

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


def wrap_in_lora(model: Type[nn.Module], targets: List[str]):
    class _FlaxLora(model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def wrap(self):
            for n, attr in {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init}.items():
                subattrs = {f.name: getattr(attr, f.name) for f in dataclasses.fields(attr) if f.init}
                object.__setattr__(self, n, wrap_in_lora(attr.__class__, targets=targets)(**subattrs))

        # def clone(self, *, parent=None, **updates):
        #     """Creates a clone of this Module, with optionally updated arguments.

        #     Args:
        #     parent: The parent of the clone. The clone will have no parent if no
        #         explicit parent is specified.
        #     **updates: Attribute updates.
        #     Returns:
        #     A clone of the this Module with the updated attributes and parent.
        #     """
        #     self.wrap()
        #     attrs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init}
        #     attrs.update(parent=parent, **updates)
        #     return self.__class__(**attrs)

        def setup(self):
            super().setup()
            params = cast(FlaxModelMixin, self).init_weights(jax.random.PRNGKey(0))
            FlaxLoraBase.inject(params, self, targets=targets)
            self.wrap()

    _FlaxLora.__name__ = f"{model.__name__}Lora"
    return _FlaxLora


def FlaxLora(model: Type[nn.Module], targets=["FlaxAttentionBlock"]):
    class _LoraFlax(wrap_in_lora(model, targets=targets)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            instance, params = cast(Type[FlaxModelMixin], super()).from_pretrained(*args, **kwargs)

            params, mask = FlaxLoraBase.inject(params, instance, targets=targets)

            # subattrs = {f.name: getattr(instance, f.name) for f in dataclasses.fields(instance) if f.init}
            # instance = cls(**subattrs)

            mask_values = flatten_dict(mask)
            instance.get_mask = lambda params: unflatten_dict(
                {k: mask_values.get(k, False) for k in flatten_dict(params, keep_empty_nodes=True).keys()}
            )
            return instance, params

    _LoraFlax.__name__ = f"Lora{model.__name__}"
    return _LoraFlax
