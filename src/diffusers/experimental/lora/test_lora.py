import os
import pdb

import optax
from diffusers import FlaxUNet2DConditionModel
from diffusers.experimental.lora.linear_with_lora_flax import FlaxLora
from flax.training import train_state
from jax.config import config
from jax.experimental.compilation_cache import compilation_cache as cc


config.update("jax_traceback_filtering", "off")
cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

if __name__ == "__main__":
    unet, unet_params = FlaxLora(FlaxUNet2DConditionModel).from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        revision="flax",
    )
    get_mask = unet.get_mask

    optimizer = optax.masked(optax.adamw(1e-6), mask=get_mask)
    unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    pdb.set_trace()
