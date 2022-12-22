import copy
from collections import defaultdict
from typing import Callable, List, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from diffusers.models.unet_2d_condition_flax import FlaxUNet2DConditionModel
from flax.core.frozen_dict import FrozenDict, freeze
from flax.linen.module import SetupState
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
        self.lora_down = nn.Dense(features=4, use_bias=False)

    def init_weights(self, rng: jax.random.PRNGKey) -> FrozenDict:
        return self.init(rng, jnp.zeros((self.in_features, self.out_features)))

    def __call__(self, input):
        return self.linear(input) + self.lora_up(self.lora_down(input)) * self.scale


class FlaxLoraBase(nn.Module):
    @staticmethod
    def _get_children(model: nn.Module, params: Union[dict, FrozenDict]):
        model = model.bind({"params": params})
        if hasattr(model, "init_weights"):
            model.init_weights(jax.random.PRNGKey(0))
        return {k: v for k, v in model._state.children.items() if isinstance(v, nn.Module)}

    @staticmethod
    def _wrap_dense(params: dict, parent: nn.Module, model: Union[nn.Dense, nn.Module], name: str):
        if not isinstance(model, nn.Dense):
            return params, {}

        params_to_optimize = defaultdict(dict)

        lora = FlaxLinearWithLora(
            out_features=model.features,
            use_bias=model.use_bias,
            name=name,
        )

        lora_params = lora.init_weights(jax.random.PRNGKey(0)).unfreeze()["params"]
        lora_params["linear"] = params
        lora = lora.bind({"params": lora_params})

        # model.parent._state.in_setup = True
        # model.parent._state.setup_called = SetupState.TRANSFORMED

        # object.__setattr__(lora, "parent", model.parent)
        # setattr(parent, name, lora)
        for k, v in model.parent.__dict__.items():
            if isinstance(v, nn.Module) and v.name == name:
                setattr(model.parent, k, lora)
        # lora.__post_init__()

        # model.parent._state.setup_called = SetupState.DONE
        # model.parent._state.in_setup = False

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
        mutable_params = params.unfreeze() if isinstance(params, FrozenDict) else copy.copy(params)
        params_to_optimize = {}

        for name, child in FlaxLoraBase._get_children(model, params).items():
            if is_target:
                results = FlaxLoraBase._wrap_dense(mutable_params.get(name, {}), model, child, name)
            elif child.__class__.__name__ in targets:
                results = FlaxLoraBase.inject(mutable_params.get(name, {}), child, targets=targets, is_target=True)
            else:
                results = FlaxLoraBase.inject(mutable_params.get(name, {}), child, targets=targets)

            mutable_params[name], params_to_optimize[name] = results

        return mutable_params, params_to_optimize


def FlaxLora(module_fn: Callable[[], Tuple[nn.Module, dict]], targets=["FlaxAttentionBlock"]):
    class _FlaxLora(FlaxLoraBase):
        def setup(self):
            self.wrapped, params = module_fn()
            self._params, self._mask = self.__class__.inject(params, self.wrapped, targets=targets)

        def __call__(self, *args, **kwargs):
            return self.wrapped(*args, **kwargs)

        def init_weights(self, rng: jax.random.PRNGKey) -> FrozenDict:
            return self.wrapped.init_weights(rng)

        @property
        def params(self) -> dict:
            return self._params

        @nn.nowrap
        def get_mask(self, params):
            mask_values = flatten_dict(self._mask)
            return unflatten_dict(
                {k: mask_values.get(k, False) for k in flatten_dict(params, keep_empty_nodes=True).keys()}
            )

    return _FlaxLora()
