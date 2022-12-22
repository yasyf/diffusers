import os

import optax
from diffusers import FlaxUNet2DConditionModel
from diffusers.experimental.lora.linear_with_lora_flax import FlaxLinearWithLora, FlaxLora
from flax.training import train_state
from jax.experimental.compilation_cache import compilation_cache as cc


cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    revision="flax",
)

unet = FlaxLora(
    lambda: FlaxUNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        revision="flax",
    ),
)
unet_params = unet.params

optimizer = optax.masked(optax.adamw(1e-6), mask=unet.get_mask)

unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)
