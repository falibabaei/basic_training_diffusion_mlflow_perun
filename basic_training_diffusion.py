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
 
# This example is based on basic_training.ipynb from the Hugging Face diffusers tutorial.
# It is modified to log training information to MLflow and to measure energy consumption with Perun.

mlflow.set_tracking_uri("https://mlflow.cloud.ai4eosc.eu/")  # step (1) MLflow. Set the tracking URI. You can also log locally.
mlflow.set_experiment("basic_training_diffusion_mlflow_perun")  # step (2) MLflow. Set the experiment name.
# To use a different MLflow server, change the tracking URI above. Otherwise, export your username and password as environment variables:
# export MLFLOW_TRACKING_USERNAME="your-username"
# export MLFLOW_TRACKING_PASSWORD="your-password"   

@dataclass
class TrainingConfig:
    image_size: int = 128  # Target image resolution for generated images.
    train_batch_size: int = 16
    eval_batch_size: int = 16  # Number of images to sample during evaluation.
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 1
    save_model_epochs: int = 1
    mixed_precision: str = "fp16"  # Use "no" for float32 or "fp16" for automatic mixed precision.
    output_dir: str = "ddpm-butterflies-128"  # Model output directory both locally and on the HF Hub.
    push_to_hub: bool = False  # Whether to upload the saved model to the HF Hub.
    hub_model_id: str = "falibabaei/ddpm-butterflies-128"  # TODO. Replace <your-username> with your Hugging Face username.
    hub_private_repo: str = None
    overwrite_output_dir: bool = True
    seed: int = 0
    scheduler: str = "DDPMScheduler"

config = TrainingConfig()
config_dict = asdict(config)

# Load dataset.
dataset = load_dataset(config.dataset_name, split="train")

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# Create a UNet2DModel.
model = UNet2DModel(
    sample_size=config.image_size,  # Target image resolution.
    in_channels=3,  # Number of input channels, 3 for RGB images.
    out_channels=3,  # Number of output channels.
    layers_per_block=2,  # Number of ResNet layers to use per UNet block.
    block_out_channels=(128, 128, 256, 256, 512, 512),  # Output channels for each UNet block.
    down_block_types=(
        "DownBlock2D",  # Standard ResNet downsampling block.
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # ResNet downsampling block with spatial self attention.
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # Standard ResNet upsampling block.
        "AttnUpBlock2D",  # ResNet upsampling block with spatial self attention.
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

# Create a scheduler.
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
    # Sample images from random noise, which corresponds to the reverse diffusion process.
    # The default pipeline output type is List[PIL.Image].
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device="cpu").manual_seed(config.seed),  # Use a separate generator so the main training RNG state is not affected.
    ).images

    # Arrange sampled images into a grid.
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the image grid to disk.
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


#  step (3) This decorator is provided by Perun and starts measuring energy consumption for the wrapped function.
@monitor()
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize Accelerator and TensorBoard logging.
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare model, optimizer, dataloader and scheduler for accelerated training.
    # The objects must be unpacked in the same order in which they are passed to prepare.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    # step (4) start an MLflow run to log training information.
    with mlflow.start_run() as active_run:
        # step (5) Log configuration hyperparameters to MLflow.
        mlflow.log_params(config_dict)

        # step (6) Send the metrics collected by Perun to MLflow.
        @register_callback
        def perun2mlflow(node):
            mlflow.start_run(active_run.info.run_id)
            for metricType, metric in node.metrics.items():
                name = f"{metricType.value}"
                mlflow.log_metric(name, metric.value)

        # Train the model.
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                # Sample noise to add to the images.
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image.
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bs,),
                    device=clean_images.device,
                    dtype=torch.int64,
                )

                # Add noise to the clean images according to the schedule at each timestep.
                # This implements the forward diffusion process.
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timsteps)

                with accelerator.accumulate(model):
                    # Predict the noise that was added to the images.
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # step (7) Log training loss and learning rate to MLflow.
                mlflow.log_metric("training_loss", logs["loss"], step=global_step)
                mlflow.log_metric("learning_rate", logs["lr"], step=global_step)
                
                global_step += 1

            # After each epoch, optionally sample demo images and save the model.
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if config.push_to_hub:
                        upload_folder(
                            repo_id=repo_id,
                            folder_path=config.output_dir,
                            commit_message=f"Epoch {epoch}",
                            ignore_patterns=["step_*", "epoch_*"],
                        )
                    else:
                        pipeline.save_pretrained(config.output_dir)
                        # step (8) Log the saved model directory as an MLflow artifact.
                        mlflow.log_artifacts(config.output_dir, artifact_path="model_epoch_{:03d}".format(epoch + 1))


if __name__ == "__main__":
    notebook_launcher(
        train_loop,
        args=(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler),
        num_processes=1,
    )
