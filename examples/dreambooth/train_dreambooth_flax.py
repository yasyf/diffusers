import argparse
import hashlib
import itertools
import logging
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset, IterableDataset

import jax
import jax.experimental.host_callback
import jax.numpy as jnp
import optax
import transformers
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.experimental.lora.linear_with_lora_flax import FlaxLinearWithLora
from diffusers.models.vae_flax import FlaxDiagonalGaussianDistribution
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.utils import check_min_version
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from flax.traverse_util import flatten_dict, unflatten_dict
from huggingface_hub import HfFolder, Repository, whoami
from jax.experimental.compilation_cache import compilation_cache as cc
from PIL import Image
from torchvision import io, transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel, set_seed


cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="Do not precompute and cache latents from VAE.",
        default=False,
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution (768 for stabilityai/stable-diffusion-2)"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--augment_images", action="store_true", help="Apply random data augmentations.")
    parser.add_argument("--lora", action="store_true", help="Use LoRA (https://arxiv.org/abs/2106.09685)")
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args


class DreamBoothDataset(IterableDataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        augment_images=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.augment_images = augment_images
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __iter__(self):
        return (self[i] for i in random.sample(range(self._length), self._length))

    def _augment(self, path, do_augment: int):
        rand_transforms = transforms.Compose(
            [
                transforms.RandomErasing(p=0.5 * do_augment),
                transforms.ToPILImage(),
                transforms.RandomOrder(
                    [
                        transforms.ColorJitter(
                            brightness=0.2 * do_augment,
                            contrast=0.2 * do_augment,
                            saturation=0.2 * do_augment,
                            hue=0.2 * do_augment,
                        ),
                        transforms.RandomResizedCrop(self.size * (0.8 + (0.2 * do_augment))),
                        transforms.RandomVerticalFlip(0.5 * do_augment),
                        transforms.RandomInvert(0.5 * do_augment),
                        transforms.RandomAdjustSharpness(2, p=0.5 * do_augment),
                        transforms.RandomAutocontrast(p=0.5 * do_augment),
                    ]
                ),
            ]
        )
        return rand_transforms(io.read_image(str(path)))

    def _instance_image(self, index):
        path = self.instance_images_path[index % self.num_instance_images]
        do_augment, index = divmod(index, self.num_instance_images)
        instance_image = self._augment(path, do_augment=do_augment) if self.augment_images else Image.open(path)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        return {
            "instance_images": self.image_transforms(instance_image),
            "instance_prompt_ids": self.tokenizer(
                self.instance_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids,
        }

    def _class_image(self, index):
        if not self.class_data_root:
            return {}

        class_image = Image.open(self.class_images_path[index % self.num_class_images])

        if not class_image.mode == "RGB":
            class_image = class_image.convert("RGB")

        return {
            "class_images": self.image_transforms(class_image),
            "class_prompt_ids": self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids,
        }

    def __getitem__(self, index):
        return {**self._instance_image(index), **self._class_image(index)}


class LatentsDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    rng = jax.random.PRNGKey(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path, safety_checker=None, revision=args.revision
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            total_sample_batch_size = args.sample_batch_size * jax.local_device_count()
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=total_sample_batch_size)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not jax.process_index() == 0
            ):
                prompt_ids = pipeline.prepare_inputs(example["prompt"])
                prompt_ids = shard(prompt_ids)
                p_params = jax_utils.replicate(params)
                rng = jax.random.split(rng)[0]
                sample_rng = jax.random.split(rng, jax.device_count())
                images = pipeline(prompt_ids, p_params, sample_rng, jit=True).images
                images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
                images = pipeline.numpy_to_pil(np.array(images))

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
    else:
        raise NotImplementedError()

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        batch = {k: v.numpy() for k, v in batch.items()}
        return batch

    total_train_batch_size = args.train_batch_size * jax.local_device_count()
    if len(train_dataset) < total_train_batch_size:
        raise ValueError(
            f"Training batch size is {total_train_batch_size}, but your dataset only contains"
            f" {len(train_dataset)} images. Please, use a larger dataset or reduce the effective batch size. Note that"
            f" there are {jax.local_device_count()} parallel devices, so your batch size can't be smaller than that."
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=total_train_batch_size,
        collate_fn=collate_fn,
        drop_last=True,
    )

    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16

    # Load models and create wrapper for stable diffusion
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", dtype=weight_dtype, revision=args.revision
    )
    feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    if args.pretrained_vae_name_or_path:
        vae_arg, vae_kwargs = (args.pretrained_vae_name_or_path, {"from_pt": True})
    else:
        vae_arg, vae_kwargs = (args.pretrained_model_name_or_path, {"subfolder": "vae", "revision": args.revision})

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        vae_arg,
        dtype=weight_dtype,
        **vae_kwargs,
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        dtype=weight_dtype,
        revision=args.revision,
    )

    noise_scheduler, _ = FlaxDDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
    )
    noise_scheduler_state = noise_scheduler.create_state()

    # Optimization
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )

    if args.lora:
        masks = {}
        unet_params, masks["unet"] = FlaxLinearWithLora.inject(unet_params, unet)
        if args.train_text_encoder:
            text_encoder._params, masks["text_encoder"] = FlaxLinearWithLora.inject(
                text_encoder.params, text_encoder.module, targets=["FlaxCLIPAttention"]
            )

        mask_values = flatten_dict(dict(itertools.chain(*[v.items() for v in masks.values()])))
        print("MARK", mask_values)
        all_mask = unflatten_dict(
            {
                k: mask_values.get(k, False)
                for k in itertools.chain(
                    flatten_dict(unet_params).keys(),
                    flatten_dict(text_encoder.params).keys(),
                )
            }
        )

        print(list(mask_values.keys())[1:10])
        print(list(unet_params.keys())[1:10])
        print(list(text_encoder.params.keys())[1:10])
        print(list(all_mask.keys())[1:10])

        optimizer = optax.masked(optimizer, mask=all_mask)

    unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)
    text_encoder_state = train_state.TrainState.create(
        apply_fn=text_encoder.__call__, params=text_encoder.params, tx=optimizer
    )

    # Initialize our training
    train_rngs = jax.random.split(rng, jax.local_device_count())

    @partial(jax.jit, donate_argnums=(1, 2, 3))
    def compute_loss(params, dropout_rng, sample_rng, batch):
        # Convert images to latent space
        if args.cache_latents:
            latent_dist = FlaxDiagonalGaussianDistribution(batch["pixel_values"], deterministic=True)
        else:
            latent_dist = vae.apply(
                {"params": params["vae_params"]}, batch["pixel_values"], deterministic=True, method=vae.encode
            ).latent_dist
        latents = latent_dist.sample(sample_rng)
        # (NHWC) -> (NCHW)
        latents = jnp.transpose(latents, (0, 3, 1, 2))
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise_rng, timestep_rng = jax.random.split(sample_rng)
        noise = jax.random.normal(noise_rng, latents.shape)
        # Sample a random timestep for each image
        bsz = latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng,
            (bsz,),
            0,
            noise_scheduler.config.num_train_timesteps,
        )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

        # Get the text embedding for conditioning
        if args.train_text_encoder:
            encoder_hidden_states = text_encoder_state.apply_fn(
                batch["input_ids"], params=params["text_encoder"], dropout_rng=dropout_rng, train=True
            )[0]
        elif args.cache_latents:
            encoder_hidden_states = batch["input_ids"]
        else:
            encoder_hidden_states = text_encoder(batch["input_ids"], params=text_encoder_state.params, train=False)[0]

        # Predict the noise residual
        model_pred = unet.apply(
            {"params": params["unet"]}, noisy_latents, timesteps, encoder_hidden_states, train=True
        ).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        if args.with_prior_preservation:
            # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = jnp.split(model_pred, 2, axis=0)
            target, target_prior = jnp.split(target, 2, axis=0)

            # Compute instance loss
            loss = (target - model_pred) ** 2
            loss = loss.mean()

            # Compute prior loss
            prior_loss = (target_prior - model_pred_prior) ** 2
            prior_loss = prior_loss.mean()

            # Add the prior loss to the instance loss.
            loss = loss + args.prior_loss_weight * prior_loss
        else:
            loss = (target - model_pred) ** 2
            loss = loss.mean()

        return loss

    def train_step(unet_state, text_encoder_state, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        params = {"vae_params": vae_params, "unet": unet_state.params}
        if args.train_text_encoder:
            params["text_encoder"] = text_encoder_state.params

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(params, dropout_rng, sample_rng, batch)
        grad = jax.lax.pmean(grad, "batch")

        new_unet_state = unet_state.apply_gradients(grads=grad["unet"])
        if args.train_text_encoder:
            new_text_encoder_state = text_encoder_state.apply_gradients(grads=grad["text_encoder"])
        else:
            new_text_encoder_state = text_encoder_state

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_unet_state, new_text_encoder_state, metrics, new_train_rng

    def cache_image_latents(pixel_values, vae_params):
        with torch.no_grad():
            _, results = vae.apply(
                {"params": vae_params},
                pixel_values,
                method=vae.encode,
                deterministic=True,
                capture_intermediates=True,
            )
            return results["intermediates"]["quant_conv"]["__call__"][0]

    def cache_text_latents(input_ids, text_encoder_state):
        with torch.no_grad():
            return text_encoder(
                input_ids,
                params=text_encoder_state.params,
                train=False,
            )[0]

    def cache_latents(batch, vae_params, text_encoder_state, train_text_encoder):
        return {
            "pixel_values": cache_image_latents(batch["pixel_values"], vae_params),
            "input_ids": batch["input_ids"]
            if train_text_encoder
            else cache_text_latents(batch["input_ids"], text_encoder_state),
        }

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0, 1, 3, 4))
    p_cache_latents = jax.pmap(cache_latents, "batch", donate_argnums=(0,), static_broadcasted_argnums=(3,))

    # Replicate the train state on each device
    unet_state = jax_utils.replicate(unet_state)
    text_encoder_state = jax_utils.replicate(text_encoder_state)
    vae_params = jax_utils.replicate(vae_params)

    # Cache latents
    if args.cache_latents:
        latents = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            batch_latents = p_cache_latents(shard(batch), vae_params, text_encoder_state, args.train_text_encoder)
            latents.append(batch_latents)

        train_dataloader = torch.utils.data.DataLoader(
            LatentsDataset(latents),
            batch_size=1,
            shuffle=True,
            collate_fn=lambda l: l[0],
        )

    # Train!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def checkpoint(step):
        # Create the pipeline using using the trained modules and save it.
        if jax.process_index() == 0:
            print(f"Checkpointing at step {step}...")

            scheduler = FlaxPNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                skip_prk_steps=True,
            )
            pipeline = FlaxStableDiffusionPipeline(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=feature_extractor,
                dtype=weight_dtype,
            )
            outdir = os.path.join(args.output_dir, str(step)) if args.save_steps else args.output_dir

            pipeline.save_pretrained(
                outdir,
                params={
                    "text_encoder": get_params_to_save(text_encoder_state.params),
                    "vae": get_params_to_save(vae_params),
                    "unet": get_params_to_save(unet_state.params),
                },
            )

            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    global_step = 0

    epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================

        train_metrics = []

        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in train_dataloader:
            batch = batch if args.cache_latents else shard(batch)
            unet_state, text_encoder_state, train_metric, train_rngs = p_train_step(
                unet_state, text_encoder_state, vae_params, batch, train_rngs
            )
            train_metrics.append(train_metric)

            train_step_progress_bar.update(jax.local_device_count())

            global_step += 1
            if args.save_steps and global_step % args.save_steps == 0:
                checkpoint(global_step)
            if global_step >= args.max_train_steps:
                break

        train_metric = jax_utils.unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

    if not args.save_steps or global_step % args.save_steps:
        checkpoint(global_step)


if __name__ == "__main__":
    main()
