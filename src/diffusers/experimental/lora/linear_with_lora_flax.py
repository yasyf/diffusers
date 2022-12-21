import copy
import re
from collections import defaultdict
from typing import Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict, freeze
from flax.traverse_util import flatten_dict


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

    @staticmethod
    def _get_children(model: nn.Module):
        return {k: v for k, v in model._state.children.items() if isinstance(v, nn.Module)}

    @staticmethod
    def _wrap_dense(params: dict, model: Union[nn.Dense, nn.Module], name: str):
        if not isinstance(model, nn.Dense):
            return params, {}

        params_to_optimize = defaultdict(dict)

        lora = FlaxLinearWithLora(
            out_features=model.features,
            use_bias=model.use_bias,
        )
        object.__setattr__(lora, "name", name)

        lora_params = lora.init_weights(jax.random.PRNGKey(0)).unfreeze()["params"]
        lora_params["linear"] = params
        lora = lora.bind({"params": lora_params})

        object.__setattr__(lora.parent, name, None)
        model.parent._state.in_setup = True
        setattr(model.parent, name, lora)
        model.parent._state.in_setup = False

        print("PARAMS", lora_params)
        for n in ["lora_up", "lora_down"]:
            params_to_optimize[n] = {k: True for k in lora_params[n].keys()}
            print(f"{n} {params_to_optimize[n]}")
        params_to_optimize["linear"] = {k: False for k in lora_params["linear"].keys()}

        print("OPT", params_to_optimize)
        return lora_params, dict(params_to_optimize)

    @staticmethod
    def inject(
        params: Union[dict, FrozenDict],
        model: nn.Module,
        targets=[
            "FlaxAttentionBlock",
        ],
        is_target: bool = False,
    ):
        model = model.bind({"params": params})
        if hasattr(model, "init_weights"):
            model.init_weights(jax.random.PRNGKey(0))

        mutable_params = params.unfreeze() if isinstance(params, FrozenDict) else copy.copy(params)
        params_to_optimize = {}

        for name, child in FlaxLinearWithLora._get_children(model).items():
            if is_target:
                results = FlaxLinearWithLora._wrap_dense(mutable_params.get(name, {}), child, name)
            elif child.__class__.__name__ in targets:
                results = FlaxLinearWithLora.inject(mutable_params.get(name, {}), child, is_target=True)
            else:
                results = FlaxLinearWithLora.inject(mutable_params.get(name, {}), child)

            mutable_params[name], params_to_optimize[name] = results

        return mutable_params, params_to_optimize
