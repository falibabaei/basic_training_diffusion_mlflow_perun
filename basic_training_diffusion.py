from dataclasses import dataclass, asdict
import mlflow
import os 
from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers import UNet2DModel

from PIL import Image
from diffusers import DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from accelerate import notebook_launcher

from perun import monitor, register_callback
from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
from huggingface_hub import split_torch_state_dict_into_shards  # add at top of file
import json
import math
 
#This example is basic_training.ipynb (https://huggingface.co/docs/diffusers/main/tutorials/basic_training)
#  from Huggingface diffusers library modified to log training information to MLflow and measure energy 
# consumption with Perun

mlflow.set_tracking_uri("https://mlflow.scc.kit.edu/")#(1) MLFLOW: add the tracking uri. You could also log it locally
mlflow.set_experiment("test_1") #(2) MLFLOW: set the experiment name

@dataclass
class TrainingConfig:
    image_size: int = 128  # the generated image resolution
    train_batch_size: int = 8
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_epochs: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 1
    save_model_epochs: int = 1
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_model_id: str = "falibabaei/ddpm-butterflies-128"  # TODO: Replace <your-username> with your Hugging Face username
    hub_private_repo: str = None
    overwrite_output_dir: bool = True
    seed: int = 0
    scheduler: str = "DDPMScheduler"
    num_processes: int = 2
    torch_dtype: str="auto" 
    distributed_type: str= 'MULTI_GPU'


config = TrainingConfig()
config_dict = asdict(config)

#load dataset
config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")



preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
# Create a UNet2DModel


model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
# Create a scheduler

sample_image = dataset[0]["images"].unsqueeze(0)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)


noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


@monitor()
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    import os, torch
    print(
        f"[rank {accelerator.process_index}] "
        f"device={accelerator.device}, "
        f"local_cuda={torch.cuda.current_device()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # ONLY rank 0 starts the MLflow run
    if accelerator.is_main_process:
        with mlflow.start_run() as active_run:
            mlflow.log_params(config_dict)

            @register_callback
            def perun2mlflow(node):
                mlflow.start_run(active_run.info.run_id)
                for metricType, metric in node.metrics.items():
                    name = f"{metricType.value}"
                    mlflow.log_metric(name, metric.value)

            for epoch in range(config.num_epochs):
                progress_bar = tqdm(
                    total=len(train_dataloader),
                    disable=not accelerator.is_local_main_process,
                )
                progress_bar.set_description(f"Epoch {epoch}")

                for step, batch in enumerate(train_dataloader):
                    clean_images = batch["images"]
                    noise = torch.randn(clean_images.shape, device=clean_images.device)
                    bs = clean_images.shape[0]
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bs,),
                        device=clean_images.device,
                        dtype=torch.int64,
                    )
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                    with accelerator.accumulate(model):
                        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                        loss = F.mse_loss(noise_pred, noise)
                        accelerator.backward(loss)

                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    progress_bar.update(1)
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    # MLflow metrics only from main process
                    mlflow.log_metric("training_loss", logs["loss"], step=global_step)
                    mlflow.log_metric("learning_rate", logs["lr"], step=global_step)

                    global_step += 1

                if config.push_to_hub:
                    pipeline = DDPMPipeline(
                        unet=accelerator.unwrap_model(model),
                        scheduler=noise_scheduler,
                    )
                    # save / upload etc. only from main process
                    pipeline.save_pretrained(config.output_dir)

                import time
                start = time.perf_counter()
                mlflow.log_artifacts(
                    config.output_dir,
                    artifact_path="model_epoch_{:03d}".format(epoch + 1),
                )
                end = time.perf_counter()
                print(f"MLflow artifact upload took {end - start:.1f} seconds")

    else:
        # Non‑main ranks still run training but do not touch MLflow
        for epoch in range(config.num_epochs):
            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bs,),
                    device=clean_images.device,
                    dtype=torch.int64,
                )
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
if __name__ == "__main__":

        train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
 
