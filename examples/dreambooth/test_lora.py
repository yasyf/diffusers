from diffusers import FlaxUNet2DConditionModel
from diffusers.experimental.lora.linear_with_lora_flax import FlaxLinearWithLora


unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    revision="flax",
)

unet_params, unet_masks = FlaxLinearWithLora.inject(unet_params, unet)
