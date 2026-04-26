"""BLIP2-based video captioning model implementation."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Blip2Config,
)
from omegaconf import DictConfig


class BLIP2VideoCaptioningModel(nn.Module):
    """BLIP2-based video captioning model with temporal attention."""
    
    def __init__(self, config: DictConfig):
        """Initialize BLIP2 video captioning model.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        
        self.config = config
        self.max_frames = config.max_frames
        self.max_length = config.max_length
        self.num_beams = config.num_beams
        self.temperature = config.temperature
        
        # Load BLIP2 model and processor
        self.model_name = config.vision_model_name
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if config.get("use_fp16", False) else torch.float32,
        )
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        
        # Temporal attention for video frames
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.blip2_model.config.vision_config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        
        # Frame fusion layer
        self.frame_fusion = nn.Linear(
            self.blip2_model.config.vision_config.hidden_size,
            self.blip2_model.config.vision_config.hidden_size,
        )
        
        # Freeze parameters if specified
        if config.get("freeze_vision_encoder", False):
            self._freeze_vision_encoder()
        
        if config.get("freeze_text_encoder", False):
            self._freeze_text_encoder()
    
    def _freeze_vision_encoder(self) -> None:
        """Freeze vision encoder parameters."""
        for param in self.blip2_model.vision_model.parameters():
            param.requires_grad = False
    
    def _freeze_text_encoder(self) -> None:
        """Freeze text encoder parameters."""
        for param in self.blip2_model.language_model.parameters():
            param.requires_grad = False
    
    def _process_video_frames(
        self, 
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Process video frames with temporal attention.
        
        Args:
            pixel_values: Video frames tensor of shape (B, T, C, H, W).
            
        Returns:
            Processed frame features.
        """
        batch_size, num_frames = pixel_values.shape[:2]
        
        # Reshape for batch processing
        pixel_values_flat = pixel_values.view(-1, *pixel_values.shape[2:])
        
        # Get vision features for all frames
        vision_outputs = self.blip2_model.vision_model(pixel_values_flat)
        frame_features = vision_outputs.last_hidden_state  # (B*T, N, D)
        
        # Reshape back to (B, T, N, D)
        frame_features = frame_features.view(
            batch_size, num_frames, *frame_features.shape[1:]
        )
        
        # Apply temporal attention
        # Use mean pooling over spatial dimensions for temporal attention
        frame_features_pooled = frame_features.mean(dim=2)  # (B, T, D)
        
        # Apply temporal attention
        attended_features, _ = self.temporal_attention(
            frame_features_pooled,
            frame_features_pooled,
            frame_features_pooled,
        )
        
        # Fuse attended features
        fused_features = self.frame_fusion(attended_features)
        
        # Use the last frame's features as the primary visual representation
        # but incorporate temporal information
        primary_features = frame_features[:, -1]  # (B, N, D)
        temporal_context = fused_features.mean(dim=1, keepdim=True)  # (B, 1, D)
        
        # Combine primary and temporal features
        enhanced_features = primary_features + temporal_context.expand_as(primary_features)
        
        return enhanced_features
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            pixel_values: Video frames tensor.
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            labels: Target labels for training.
            
        Returns:
            Model outputs.
        """
        # Process video frames
        vision_features = self._process_video_frames(pixel_values)
        
        # Create a dummy image tensor for BLIP2 (it expects image input)
        # We'll use the processed video features as the "image"
        batch_size = vision_features.shape[0]
        dummy_image = torch.zeros(batch_size, 3, 224, 224, device=vision_features.device)
        
        # Override the vision model output with our processed features
        original_forward = self.blip2_model.vision_model.forward
        
        def custom_vision_forward(pixel_values):
            # Return our processed features instead of running vision model
            return type('VisionOutput', (), {
                'last_hidden_state': vision_features,
                'pooler_output': vision_features.mean(dim=1),
            })()
        
        # Temporarily replace the vision model forward method
        self.blip2_model.vision_model.forward = custom_vision_forward
        
        try:
            # Forward pass through BLIP2
            outputs = self.blip2_model(
                pixel_values=dummy_image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            # Restore original forward method
            self.blip2_model.vision_model.forward = original_forward
        
        return outputs
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = False,
        early_stopping: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate captions for video frames.
        
        Args:
            pixel_values: Video frames tensor.
            max_length: Maximum generation length.
            num_beams: Number of beams for beam search.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            early_stopping: Whether to use early stopping.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated token IDs.
        """
        max_length = max_length or self.max_length
        num_beams = num_beams or self.num_beams
        temperature = temperature or self.temperature
        
        # Process video frames
        vision_features = self._process_video_frames(pixel_values)
        
        # Create dummy image for BLIP2
        batch_size = vision_features.shape[0]
        dummy_image = torch.zeros(batch_size, 3, 224, 224, device=vision_features.device)
        
        # Override vision model output
        original_forward = self.blip2_model.vision_model.forward
        
        def custom_vision_forward(pixel_values):
            return type('VisionOutput', (), {
                'last_hidden_state': vision_features,
                'pooler_output': vision_features.mean(dim=1),
            })()
        
        self.blip2_model.vision_model.forward = custom_vision_forward
        
        try:
            # Generate captions
            generated_ids = self.blip2_model.generate(
                pixel_values=dummy_image,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                early_stopping=early_stopping,
                **kwargs
            )
        finally:
            # Restore original forward method
            self.blip2_model.vision_model.forward = original_forward
        
        return generated_ids
    
    def get_attention_weights(
        self, 
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Get attention weights for visualization.
        
        Args:
            pixel_values: Video frames tensor.
            
        Returns:
            Attention weights.
        """
        # Process video frames
        vision_features = self._process_video_frames(pixel_values)
        
        # Get attention weights from temporal attention
        batch_size, num_frames = pixel_values.shape[:2]
        frame_features_pooled = vision_features.mean(dim=1)  # (B, D)
        
        # Reshape for attention computation
        frame_features_pooled = frame_features_pooled.unsqueeze(1)  # (B, 1, D)
        
        _, attention_weights = self.temporal_attention(
            frame_features_pooled,
            frame_features_pooled,
            frame_features_pooled,
        )
        
        return attention_weights
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save model and processor.
        
        Args:
            save_directory: Directory to save the model.
        """
        self.blip2_model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
        
        # Save custom components
        torch.save({
            'temporal_attention': self.temporal_attention.state_dict(),
            'frame_fusion': self.frame_fusion.state_dict(),
            'config': self.config,
        }, f"{save_directory}/video_captioning_components.pt")
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str, 
        config: Optional[DictConfig] = None
    ) -> "BLIP2VideoCaptioningModel":
        """Load model from pretrained checkpoint.
        
        Args:
            model_path: Path to the model directory.
            config: Model configuration.
            
        Returns:
            Loaded model instance.
        """
        # Load BLIP2 model
        blip2_model = Blip2ForConditionalGeneration.from_pretrained(model_path)
        
        # Create model instance
        if config is None:
            # Load config from saved components
            components = torch.load(f"{model_path}/video_captioning_components.pt")
            config = components['config']
        
        model = cls(config)
        
        # Load custom components
        if os.path.exists(f"{model_path}/video_captioning_components.pt"):
            components = torch.load(f"{model_path}/video_captioning_components.pt")
            model.temporal_attention.load_state_dict(components['temporal_attention'])
            model.frame_fusion.load_state_dict(components['frame_fusion'])
        
        return model
