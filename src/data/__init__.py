"""Data loading and preprocessing utilities for video captioning."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BlipProcessor
import decord
from decord import VideoReader
from PIL import Image


class VideoCaptioningDataset(Dataset):
    """Dataset for video captioning tasks."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_frames: int = 8,
        frame_sampling_strategy: str = "uniform",
        video_fps: float = 1.0,
        image_size: int = 224,
        max_caption_length: int = 200,
        tokenizer_name: str = "Salesforce/blip2-opt-2.7b",
        enable_augmentation: bool = True,
        augmentation_prob: float = 0.5,
    ):
        """Initialize video captioning dataset.
        
        Args:
            data_dir: Directory containing video data.
            split: Dataset split (train/val/test).
            max_frames: Maximum number of frames to sample.
            frame_sampling_strategy: Strategy for frame sampling.
            video_fps: Target FPS for video processing.
            image_size: Size to resize frames to.
            max_caption_length: Maximum caption length.
            tokenizer_name: Name of tokenizer to use.
            enable_augmentation: Whether to enable data augmentation.
            augmentation_prob: Probability of applying augmentation.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_frames = max_frames
        self.frame_sampling_strategy = frame_sampling_strategy
        self.video_fps = video_fps
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        self.enable_augmentation = enable_augmentation
        self.augmentation_prob = augmentation_prob
        
        # Load processor
        self.processor = BlipProcessor.from_pretrained(tokenizer_name)
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Set up video reader
        decord.bridge.set_bridge("torch")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load dataset annotations.
        
        Returns:
            List of annotation dictionaries.
        """
        annotation_file = self.data_dir / f"{self.split}.json"
        
        if annotation_file.exists():
            with open(annotation_file, "r") as f:
                return json.load(f)
        else:
            # Create dummy dataset if annotations don't exist
            return self._create_dummy_dataset()
    
    def _create_dummy_dataset(self) -> List[Dict[str, Any]]:
        """Create a dummy dataset for testing.
        
        Returns:
            List of dummy annotations.
        """
        dummy_annotations = []
        
        # Create dummy video files
        video_dir = self.data_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(10):
            video_path = video_dir / f"video_{i:03d}.mp4"
            
            # Create dummy video if it doesn't exist
            if not video_path.exists():
                self._create_dummy_video(str(video_path))
            
            dummy_annotations.append({
                "video_id": f"video_{i:03d}",
                "video_path": str(video_path),
                "caption": f"This is a sample video {i} showing various activities and scenes.",
                "duration": 10.0,
                "fps": 30.0,
            })
        
        return dummy_annotations
    
    def _create_dummy_video(self, video_path: str) -> None:
        """Create a dummy video file for testing.
        
        Args:
            video_path: Path to save the dummy video.
        """
        # Create a simple colored video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        for frame_idx in range(300):  # 10 seconds at 30 FPS
            # Create a frame with changing colors
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = (frame_idx * 2) % 255  # Red channel
            frame[:, :, 1] = (frame_idx * 3) % 255  # Green channel
            frame[:, :, 2] = (frame_idx * 4) % 255  # Blue channel
            
            # Add some text
            cv2.putText(
                frame,
                f"Frame {frame_idx}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            
            out.write(frame)
        
        out.release()
    
    def _sample_frames(self, video_path: str) -> torch.Tensor:
        """Sample frames from video.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Tensor of sampled frames.
        """
        try:
            vr = VideoReader(video_path)
            total_frames = len(vr)
            
            if total_frames == 0:
                raise ValueError(f"Video {video_path} has no frames")
            
            # Sample frames based on strategy
            if self.frame_sampling_strategy == "uniform":
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            elif self.frame_sampling_strategy == "random":
                frame_indices = np.random.choice(
                    total_frames, self.max_frames, replace=False
                )
                frame_indices = np.sort(frame_indices)
            else:
                # Default to uniform sampling
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            
            # Read frames
            frames = vr.get_batch(frame_indices)
            
            # Convert to RGB and resize
            frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            frames = torch.nn.functional.interpolate(
                frames, size=(self.image_size, self.image_size), mode="bilinear"
            )
            
            return frames
            
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            # Return dummy frames if video reading fails
            return torch.zeros(self.max_frames, 3, self.image_size, self.image_size)
    
    def _augment_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to frames.
        
        Args:
            frames: Input frames tensor.
            
        Returns:
            Augmented frames tensor.
        """
        if not self.enable_augmentation or np.random.random() > self.augmentation_prob:
            return frames
        
        # Convert to numpy for augmentation
        frames_np = frames.permute(0, 2, 3, 1).numpy()  # (T, C, H, W) -> (T, H, W, C)
        augmented_frames = []
        
        for frame in frames_np:
            # Horizontal flip
            if np.random.random() > 0.5:
                frame = cv2.flip(frame, 1)
            
            # Color jitter
            if np.random.random() > 0.5:
                frame = self._color_jitter(frame)
            
            # Gaussian blur
            if np.random.random() > 0.5:
                frame = self._gaussian_blur(frame)
            
            augmented_frames.append(frame)
        
        # Convert back to tensor
        augmented_frames = np.stack(augmented_frames)
        augmented_frames = torch.from_numpy(augmented_frames)
        augmented_frames = augmented_frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        
        return augmented_frames
    
    def _color_jitter(self, frame: np.ndarray) -> np.ndarray:
        """Apply color jittering to frame.
        
        Args:
            frame: Input frame.
            
        Returns:
            Color jittered frame.
        """
        # Brightness
        brightness = np.random.uniform(0.8, 1.2)
        frame = np.clip(frame * brightness, 0, 255)
        
        # Contrast
        contrast = np.random.uniform(0.8, 1.2)
        frame = np.clip((frame - 128) * contrast + 128, 0, 255)
        
        # Saturation
        saturation = np.random.uniform(0.8, 1.2)
        gray = np.dot(frame, [0.299, 0.587, 0.114])
        frame = np.clip(gray[:, :, np.newaxis] + saturation * (frame - gray[:, :, np.newaxis]), 0, 255)
        
        return frame.astype(np.uint8)
    
    def _gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to frame.
        
        Args:
            frame: Input frame.
            
        Returns:
            Blurred frame.
        """
        kernel_size = np.random.choice([3, 5, 7])
        sigma = np.random.uniform(0.5, 2.0)
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            Number of samples in dataset.
        """
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing video frames and caption.
        """
        annotation = self.annotations[idx]
        video_path = annotation["video_path"]
        caption = annotation["caption"]
        
        # Sample frames from video
        frames = self._sample_frames(video_path)
        
        # Apply augmentation
        frames = self._augment_frames(frames)
        
        # Process with BLIP processor
        # Convert frames to PIL Images for processor
        frames_pil = []
        for frame in frames:
            frame_np = frame.permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            frame_pil = Image.fromarray(frame_np)
            frames_pil.append(frame_pil)
        
        # Process frames and caption
        inputs = self.processor(
            images=frames_pil,
            text=caption,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_caption_length,
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # Remove batch dimension
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "video_id": annotation["video_id"],
            "caption": caption,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.
    
    Args:
        batch: List of batch items.
        
    Returns:
        Batched tensors.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "video_ids": [item["video_id"] for item in batch],
        "captions": [item["caption"] for item in batch],
    }
