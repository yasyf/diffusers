import copy
from collections import defaultdict
from typing import Dict, List, Type, Union, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from diffusers.modeling_flax_utils import FlaxModelMixin
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

        parent._in_setup = True
        lora = FlaxLinearWithLora(
            out_features=model.features,
            use_bias=model.use_bias,
            name=name,
            parent=parent,
        )

        lora_params = lora.init_weights(jax.random.PRNGKey(0)).unfreeze()["params"]
        lora_params["linear"] = params
        lora = lora.bind({"params": lora_params})

        for k, v in parent.__dict__.items():
            if isinstance(v, nn.Module) and v.name == name:
                setattr(model.parent, k, lora)

        parent._in_setup = False

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


def FlaxLora(model: Type[nn.Module], targets=["FlaxAttentionBlock"]):
    class _FlaxLora(model):
        def setup(self):
            super().setup()
            params = cast(FlaxModelMixin, self).init_weights(jax.random.PRNGKey(0))
            FlaxLoraBase.inject(params, self, targets=targets)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            instance, params = cast(Type[FlaxModelMixin], model).from_pretrained(*args, **kwargs)
            params, mask = FlaxLoraBase.inject(params, instance, targets=targets)
            mask_values = flatten_dict(mask)
            instance.get_mask = lambda params: unflatten_dict(
                {k: mask_values.get(k, False) for k in flatten_dict(params, keep_empty_nodes=True).keys()}
            )
            return instance, params

    _FlaxLora.__name__ = f"{model.__name__}Lora"

    return _FlaxLora
