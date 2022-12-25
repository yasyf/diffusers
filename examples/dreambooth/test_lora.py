import os
import pdb

import jax
import optax
from diffusers import FlaxUNet2DConditionModel
from diffusers.experimental.lora.linear_with_lora_flax import FlaxLinearWithLora, FlaxLora, FlaxLora2
from flax.training import train_state
from jax.config import config
from jax.experimental.compilation_cache import compilation_cache as cc


config.update("jax_traceback_filtering", "off")

cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

# unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     subfolder="unet",
#     revision="flax",
# )

unet = FlaxLora2(
    FlaxUNet2DConditionModel,
    {
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "subfolder": "unet",
        "revision": "flax",
    },
)
unet_params, get_mask = unet.mask()

optimizer = optax.masked(optax.adamw(1e-6), mask=get_mask)

unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)
