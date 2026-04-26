"""Command-line interface for video captioning system."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
import torch
from omegaconf import OmegaConf

from src.training import VideoCaptioningTrainer
from src.models import BLIP2VideoCaptioningModel
from src.eval import VideoCaptioningEvaluator
from src.data import VideoCaptioningDataset
from src.utils import set_seed, get_device, setup_logging


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.option("--device", default="auto", help="Device to use (auto/cuda/mps/cpu)")
@click.option("--seed", default=42, help="Random seed")
@click.pass_context
def cli(ctx, config: Optional[str], device: str, seed: int):
    """Video Captioning System CLI."""
    # Set seed
    set_seed(seed)
    
    # Load config
    if config:
        cfg = OmegaConf.load(config)
    else:
        cfg = OmegaConf.load("configs/config.yaml")
    
    # Override device if specified
    if device != "auto":
        cfg.device = device
    
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["logger"] = setup_logging()


@cli.command()
@click.option("--data-dir", type=click.Path(exists=True), help="Data directory")
@click.option("--output-dir", type=click.Path(), help="Output directory")
@click.option("--batch-size", type=int, help="Batch size")
@click.option("--learning-rate", type=float, help="Learning rate")
@click.option("--num-epochs", type=int, help="Number of epochs")
@click.pass_context
def train(ctx, data_dir: Optional[str], output_dir: Optional[str], 
          batch_size: Optional[int], learning_rate: Optional[float], 
          num_epochs: Optional[int]):
    """Train the video captioning model."""
    config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    # Override config with CLI arguments
    if data_dir:
        config.data_dir = data_dir
    if output_dir:
        config.output_dir = output_dir
    if batch_size:
        config.batch_size = batch_size
    if learning_rate:
        config.learning_rate = learning_rate
    if num_epochs:
        config.num_epochs = num_epochs
    
    logger.info("Starting training...")
    logger.info(f"Config: {config}")
    
    # Initialize trainer
    trainer = VideoCaptioningTrainer(config)
    
    # Train
    trainer.train()
    
    logger.info("Training completed!")


@cli.command()
@click.option("--model-path", type=click.Path(exists=True), required=True, 
              help="Path to trained model")
@click.option("--data-dir", type=click.Path(exists=True), help="Data directory")
@click.option("--split", default="test", help="Dataset split to evaluate")
@click.option("--output-file", type=click.Path(), help="Output file for results")
@click.pass_context
def evaluate(ctx, model_path: str, data_dir: Optional[str], 
             split: str, output_file: Optional[str]):
    """Evaluate the video captioning model."""
    config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    # Override config
    if data_dir:
        config.data_dir = data_dir
    
    logger.info(f"Loading model from {model_path}")
    
    # Load model
    model = BLIP2VideoCaptioningModel.from_pretrained(model_path)
    device = get_device(config.device)
    model.to(device)
    model.eval()
    
    # Load dataset
    dataset = VideoCaptioningDataset(
        data_dir=config.data_dir,
        split=split,
        **config.data
    )
    
    # Initialize evaluator
    evaluator = VideoCaptioningEvaluator(config.eval)
    
    logger.info(f"Evaluating on {split} split...")
    
    # Generate predictions
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Move to device
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            
            # Generate caption
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=config.eval.get("max_length", 77),
                num_beams=config.eval.get("num_beams", 4),
            )
            
            # Decode
            prediction = model.processor.decode(generated_ids[0], skip_special_tokens=True)
            reference = sample["caption"]
            
            all_predictions.append(prediction)
            all_references.append([reference])
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} samples")
    
    # Evaluate
    results = evaluator.evaluate(all_predictions, all_references)
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, score in results.items():
        logger.info(f"{metric}: {score:.4f}")
    
    # Save results
    if output_file:
        evaluator.save_results(results, output_file)
        logger.info(f"Results saved to {output_file}")


@cli.command()
@click.option("--model-path", type=click.Path(exists=True), required=True,
              help="Path to trained model")
@click.option("--video-path", type=click.Path(exists=True), required=True,
              help="Path to video file")
@click.option("--output-file", type=click.Path(), help="Output file for caption")
@click.pass_context
def caption(ctx, model_path: str, video_path: str, output_file: Optional[str]):
    """Generate caption for a video."""
    config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    logger.info(f"Loading model from {model_path}")
    
    # Load model
    model = BLIP2VideoCaptioningModel.from_pretrained(model_path)
    device = get_device(config.device)
    model.to(device)
    model.eval()
    
    # Load processor
    processor = model.processor
    
    logger.info(f"Processing video: {video_path}")
    
    # Process video
    import cv2
    import numpy as np
    from PIL import Image
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Sample frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, config.model.max_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, (config.model.image_size, config.model.image_size))
            frames.append(frame)
    
    cap.release()
    
    # Convert to PIL Images
    frame_images = [Image.fromarray(frame) for frame in frames]
    
    # Process with BLIP processor
    inputs = processor(images=frame_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=config.model.max_length,
            num_beams=config.model.num_beams,
            temperature=config.model.temperature,
        )
    
    # Decode caption
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # Print caption
    logger.info(f"Generated caption: {caption}")
    
    # Save caption
    if output_file:
        with open(output_file, "w") as f:
            f.write(caption)
        logger.info(f"Caption saved to {output_file}")


@cli.command()
@click.option("--data-dir", type=click.Path(exists=True), help="Data directory")
@click.option("--num-samples", type=int, default=10, help="Number of samples to create")
@click.pass_context
def create_dataset(ctx, data_dir: Optional[str], num_samples: int):
    """Create a dummy dataset for testing."""
    config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    if data_dir:
        config.data_dir = data_dir
    
    logger.info(f"Creating dummy dataset with {num_samples} samples...")
    
    # Create dataset (this will automatically create dummy data)
    dataset = VideoCaptioningDataset(
        data_dir=config.data_dir,
        split="train",
        **config.data
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    logger.info(f"Data directory: {config.data_dir}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
