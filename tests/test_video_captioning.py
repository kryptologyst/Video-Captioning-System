"""Tests for video captioning system."""

import os
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from src.models import BLIP2VideoCaptioningModel
from src.data import VideoCaptioningDataset, collate_fn
from src.eval import VideoCaptioningEvaluator
from src.utils import set_seed, get_device, EarlyStopping


class TestBLIP2VideoCaptioningModel:
    """Test cases for BLIP2VideoCaptioningModel."""
    
    @pytest.fixture
    def model_config(self):
        """Create a test model configuration."""
        return OmegaConf.create({
            "vision_model_name": "Salesforce/blip2-opt-2.7b",
            "text_model_name": "Salesforce/blip2-opt-2.7b",
            "max_frames": 4,
            "max_length": 50,
            "num_beams": 2,
            "temperature": 1.0,
            "num_attention_heads": 8,
            "freeze_vision_encoder": False,
            "freeze_text_encoder": False,
        })
    
    @pytest.fixture
    def model(self, model_config):
        """Create a test model."""
        return BLIP2VideoCaptioningModel(model_config)
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.max_frames == 4
        assert model.max_length == 50
        assert model.num_beams == 2
    
    def test_model_forward(self, model):
        """Test model forward pass."""
        batch_size = 2
        num_frames = 4
        height, width = 224, 224
        
        # Create dummy input
        pixel_values = torch.randn(batch_size, num_frames, 3, height, width)
        input_ids = torch.randint(0, 1000, (batch_size, 10))
        attention_mask = torch.ones(batch_size, 10)
        
        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)
        
        assert outputs is not None
        assert hasattr(outputs, 'loss') or hasattr(outputs, 'logits')
    
    def test_model_generate(self, model):
        """Test model generation."""
        batch_size = 1
        num_frames = 4
        height, width = 224, 224
        
        # Create dummy input
        pixel_values = torch.randn(batch_size, num_frames, 3, height, width)
        
        # Generate captions
        generated_ids = model.generate(pixel_values, max_length=20)
        
        assert generated_ids is not None
        assert generated_ids.shape[0] == batch_size
        assert generated_ids.shape[1] <= 20
    
    def test_model_attention_weights(self, model):
        """Test attention weights extraction."""
        batch_size = 1
        num_frames = 4
        height, width = 224, 224
        
        # Create dummy input
        pixel_values = torch.randn(batch_size, num_frames, 3, height, width)
        
        # Get attention weights
        attention_weights = model.get_attention_weights(pixel_values)
        
        assert attention_weights is not None
        assert attention_weights.shape[0] == batch_size


class TestVideoCaptioningDataset:
    """Test cases for VideoCaptioningDataset."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def dataset(self, temp_data_dir):
        """Create a test dataset."""
        return VideoCaptioningDataset(
            data_dir=temp_data_dir,
            split="train",
            max_frames=4,
            image_size=224,
            max_caption_length=100,
        )
    
    def test_dataset_initialization(self, dataset):
        """Test dataset initialization."""
        assert dataset is not None
        assert dataset.max_frames == 4
        assert dataset.image_size == 224
    
    def test_dataset_length(self, dataset):
        """Test dataset length."""
        assert len(dataset) > 0
    
    def test_dataset_getitem(self, dataset):
        """Test dataset item retrieval."""
        if len(dataset) > 0:
            sample = dataset[0]
            
            assert "pixel_values" in sample
            assert "input_ids" in sample
            assert "attention_mask" in sample
            assert "video_id" in sample
            assert "caption" in sample
            
            assert sample["pixel_values"].shape[0] == dataset.max_frames
            assert sample["pixel_values"].shape[1] == 3
            assert sample["pixel_values"].shape[2] == dataset.image_size
    
    def test_collate_fn(self, dataset):
        """Test collate function."""
        if len(dataset) >= 2:
            batch = [dataset[i] for i in range(2)]
            collated = collate_fn(batch)
            
            assert "pixel_values" in collated
            assert "input_ids" in collated
            assert "attention_mask" in collated
            assert "video_ids" in collated
            assert "captions" in collated
            
            assert collated["pixel_values"].shape[0] == 2
            assert collated["input_ids"].shape[0] == 2


class TestVideoCaptioningEvaluator:
    """Test cases for VideoCaptioningEvaluator."""
    
    @pytest.fixture
    def evaluator_config(self):
        """Create a test evaluator configuration."""
        return {
            "metrics": ["bleu", "meteor", "rouge"],
            "bleu": {"max_order": 4, "smooth": False},
            "meteor": {"alpha": 0.9, "beta": 3.0, "gamma": 0.5},
            "rouge": {"rouge_types": ["rouge1", "rouge2", "rougeL"], "use_stemmer": True},
        }
    
    @pytest.fixture
    def evaluator(self, evaluator_config):
        """Create a test evaluator."""
        return VideoCaptioningEvaluator(evaluator_config)
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator is not None
        assert "bleu" in evaluator.metrics
        assert "meteor" in evaluator.metrics
        assert "rouge" in evaluator.metrics
    
    def test_evaluate(self, evaluator):
        """Test evaluation."""
        predictions = [
            "A person is walking in a park",
            "A cat is sitting on a chair",
        ]
        references = [
            ["A person walks in the park"],
            ["A cat sits on a chair"],
        ]
        
        results = evaluator.evaluate(predictions, references)
        
        assert "bleu" in results
        assert "meteor" in results
        assert "rouge1" in results
        assert "rouge2" in results
        assert "rougeL" in results
        
        # Check that scores are reasonable
        for metric, score in results.items():
            assert 0.0 <= score <= 1.0
    
    def test_diversity_metrics(self, evaluator):
        """Test diversity metrics calculation."""
        predictions = [
            "A person is walking in a park",
            "A person is walking in a park",  # Duplicate
            "A cat is sitting on a chair",
        ]
        
        diversity_metrics = evaluator.calculate_diversity_metrics(predictions)
        
        assert "distinct_1" in diversity_metrics
        assert "distinct_2" in diversity_metrics
        
        # Check that scores are reasonable
        assert 0.0 <= diversity_metrics["distinct_1"] <= 1.0
        assert 0.0 <= diversity_metrics["distinct_2"] <= 1.0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        rand1 = torch.randn(10)
        rand2 = torch.randn(10)
        
        # Set seed again and generate same numbers
        set_seed(42)
        rand3 = torch.randn(10)
        rand4 = torch.randn(10)
        
        # Should be the same
        assert torch.allclose(rand1, rand3)
        assert torch.allclose(rand2, rand4)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert device is not None
        
        # Test specific devices
        cpu_device = get_device("cpu")
        assert str(cpu_device) == "cpu"
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # Create a dummy model
        model = torch.nn.Linear(10, 1)
        
        # Test improving scores
        assert not early_stopping(0.9, model)
        assert not early_stopping(0.95, model)
        assert not early_stopping(0.98, model)
        
        # Test non-improving scores
        assert not early_stopping(0.97, model)  # Still within patience
        assert not early_stopping(0.96, model)  # Still within patience
        assert not early_stopping(0.95, model)  # Still within patience
        assert early_stopping(0.94, model)  # Should trigger early stopping


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline."""
        # This is a basic integration test
        # In a real scenario, you would test the full pipeline
        
        # Test model creation
        config = OmegaConf.create({
            "vision_model_name": "Salesforce/blip2-opt-2.7b",
            "max_frames": 4,
            "max_length": 50,
            "num_beams": 2,
            "temperature": 1.0,
            "num_attention_heads": 8,
        })
        
        model = BLIP2VideoCaptioningModel(config)
        assert model is not None
        
        # Test dataset creation
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = VideoCaptioningDataset(
                data_dir=temp_dir,
                split="train",
                max_frames=4,
                image_size=224,
            )
            assert len(dataset) > 0
            
            # Test data loading
            sample = dataset[0]
            assert "pixel_values" in sample
            
            # Test model forward pass
            pixel_values = sample["pixel_values"].unsqueeze(0)
            outputs = model(pixel_values)
            assert outputs is not None


if __name__ == "__main__":
    pytest.main([__file__])
