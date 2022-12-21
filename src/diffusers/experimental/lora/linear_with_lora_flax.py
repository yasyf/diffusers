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
        return self.init(
            rng,
            jnp.zeros((self.in_features, self.out_features)),
            jnp.zeros((self.rank, self.out_features)),
            jax.random.normal(rng, (self.in_features, self.rank)) * (1 / self.rank**2),
        )

    def __call__(self, input):
        return self.linear(input) + self.lora_up(self.lora_down(input)) * self.scale

    @staticmethod
    def _get_children(model: nn.Module):
        return {k: v for k, v in model._state.children.items() if isinstance(v, nn.Module)}

    @staticmethod
    def _wrap_dense(params: dict, model: nn.Module):
        params_to_optimize = defaultdict(lambda: defaultdict(dict))

        for name, child in FlaxLinearWithLora._get_children(model).items():
            if child.__class__.__name__ == "Dense":
                lora = FlaxLinearWithLora(
                    out_features=child.features,
                    use_bias=child.use_bias,
                )

                lora_params = lora.init_weights(jax.random.PRNGKey(0)).unfreeze()
                lora_params["linear"] = params[name]

                setattr(model, name, lora)
                params[name] = lora_params

                for n in ["lora_up", "lora_down"]:
                    params_to_optimize[name][n].update({k: True for k in lora_params[n].keys()})
                params_to_optimize[name]["linear"].update({k: False for k in lora_params["linear"].keys()})

        return params, params_to_optimize

    @staticmethod
    def inject(
        params: Union[dict, FrozenDict],
        model: nn.Module,
        targets=[
            "FlaxAttentionBlock",
        ],
    ):
        model = model.bind(params)
        model.init_weights(jax.random.PRNGKey(0))

        mutable_params = params.unfreeze() if isinstance(params, FrozenDict) else params
        params_to_optimize = {}

        for name, child in FlaxLinearWithLora._get_children(model).items():
            if child.__class__.__name__ in targets:
                results = FlaxLinearWithLora._wrap_dense(mutable_params[name], child)
            else:
                results = FlaxLinearWithLora.inject(mutable_params[name], child)

            mutable_params[name], params_to_optimize[name] = results

        for name, val in params.items():
            if name not in params_to_optimize and not isinstance(val, dict):
                params_to_optimize[name] = False

        return mutable_params, params_to_optimize
