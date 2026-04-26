"""Training utilities for video captioning models."""

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AdamW,
)
from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig

from src.models import BLIP2VideoCaptioningModel
from src.data import VideoCaptioningDataset, collate_fn
from src.eval import VideoCaptioningEvaluator
from src.utils import EarlyStopping, get_device, setup_logging


class VideoCaptioningTrainer:
    """Trainer for video captioning models."""
    
    def __init__(self, config: DictConfig):
        """Initialize trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        self.device = get_device(config.get("device", "auto"))
        self.logger = setup_logging()
        
        # Initialize model
        self.model = BLIP2VideoCaptioningModel(config.model)
        self.model.to(self.device)
        
        # Initialize datasets
        self.train_dataset = VideoCaptioningDataset(
            data_dir=config.data_dir,
            split="train",
            **config.data
        )
        self.val_dataset = VideoCaptioningDataset(
            data_dir=config.data_dir,
            split="val",
            **config.data
        )
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.data.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.data.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_fn,
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize evaluator
        self.evaluator = VideoCaptioningEvaluator(config.eval)
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 7),
            min_delta=config.get("min_delta", 0.001),
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = 0.0
        self.training_history = []
        
        # Mixed precision
        self.use_amp = config.get("mixed_precision", True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer.
        
        Returns:
            Configured optimizer.
        """
        optimizer_name = self.config.get("optimizer", "adamw")
        
        if optimizer_name.lower() == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.get("beta1", 0.9), self.config.get("beta2", 0.999)),
                eps=self.config.get("eps", 1e-8),
                weight_decay=self.config.get("weight_decay", 0.01),
            )
        elif optimizer_name.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.get("beta1", 0.9), self.config.get("beta2", 0.999)),
                eps=self.config.get("eps", 1e-8),
                weight_decay=self.config.get("weight_decay", 0.01),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler.
        
        Returns:
            Configured scheduler.
        """
        scheduler_name = self.config.get("scheduler", "linear_warmup_cosine")
        total_steps = len(self.train_loader) * self.config.num_epochs
        
        if scheduler_name == "linear_warmup_cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps,
            )
        elif scheduler_name == "linear_warmup":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Training metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Create labels (shifted input_ids)
            labels = input_ids.clone()
            labels[labels == self.model.processor.tokenizer.pad_token_id] = -100
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get("max_grad_norm", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.get("max_grad_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}: Loss = {loss.item():.4f}, "
                    f"LR = {self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set.
        
        Returns:
            Evaluation metrics.
        """
        self.model.eval()
        all_predictions = []
        all_references = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                captions = batch["captions"]
                
                # Create labels
                labels = input_ids.clone()
                labels[labels == self.model.processor.tokenizer.pad_token_id] = -100
                
                # Forward pass for loss
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss
                else:
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                
                total_loss += loss.item()
                
                # Generate predictions
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=self.config.eval.get("max_length", 77),
                    num_beams=self.config.eval.get("num_beams", 4),
                    temperature=self.config.eval.get("temperature", 1.0),
                    do_sample=self.config.eval.get("do_sample", False),
                )
                
                # Decode predictions
                predictions = self.model.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
                
                all_predictions.extend(predictions)
                all_references.extend([[caption] for caption in captions])
        
        # Calculate metrics
        eval_results = self.evaluator.evaluate(
            all_predictions, 
            all_references
        )
        
        eval_results["eval_loss"] = total_loss / len(self.val_loader)
        
        return eval_results
    
    def train(self) -> None:
        """Train the model."""
        self.logger.info("Starting training...")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            eval_metrics = self.evaluate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **eval_metrics}
            epoch_metrics["epoch"] = epoch
            self.training_history.append(epoch_metrics)
            
            # Log metrics
            self.logger.info(f"Epoch {epoch} - " + " | ".join([
                f"{k}: {v:.4f}" for k, v in epoch_metrics.items() 
                if isinstance(v, (int, float))
            ]))
            
            # Check for best model
            current_score = eval_metrics.get("cider", eval_metrics.get("bleu", 0))
            if current_score > self.best_score:
                self.best_score = current_score
                self.save_checkpoint(is_best=True)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint()
            
            # Early stopping
            if self.early_stopping(current_score, self.model):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best score: {self.best_score:.4f}")
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far.
        """
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        if is_best:
            model_path = os.path.join(checkpoint_dir, "best_model")
        else:
            model_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{self.current_epoch}")
        
        self.model.save_pretrained(model_path)
        
        # Save training state
        training_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_score": self.best_score,
            "training_history": self.training_history,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }
        
        torch.save(training_state, os.path.join(model_path, "training_state.pt"))
        
        # Save config
        with open(os.path.join(model_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        
        self.logger.info(f"Checkpoint saved to {model_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory.
        """
        # Load model
        self.model = BLIP2VideoCaptioningModel.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            self.current_epoch = training_state["epoch"]
            self.global_step = training_state["global_step"]
            self.best_score = training_state["best_score"]
            self.training_history = training_state["training_history"]
            
            # Load optimizer and scheduler states
            if "optimizer_state_dict" in training_state:
                self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
            if "scheduler_state_dict" in training_state and self.scheduler:
                self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
