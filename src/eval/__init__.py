"""Evaluation metrics and utilities for video captioning."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import BlipProcessor
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import jiwer
from omegaconf import DictConfig

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class VideoCaptioningEvaluator:
    """Evaluator for video captioning models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.metrics = config.get("metrics", ["cider", "bleu", "meteor", "rouge"])
        
        # Initialize metric-specific settings
        self.cider_config = config.get("cider", {})
        self.bleu_config = config.get("bleu", {})
        self.meteor_config = config.get("meteor", {})
        self.rouge_config = config.get("rouge", {})
        self.bert_score_config = config.get("bert_score", {})
        
        # Initialize rouge scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            self.rouge_config.get("rouge_types", ["rouge1", "rouge2", "rougeL"]),
            use_stemmer=self.rouge_config.get("use_stemmer", True)
        )
        
        # Smoothing function for BLEU
        self.smoothing_function = SmoothingFunction().method4
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[List[str]],
        video_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate predictions against references.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference caption lists.
            video_ids: Optional video IDs for tracking.
            
        Returns:
            Dictionary of metric scores.
        """
        results = {}
        
        # Ensure references are in the correct format
        if isinstance(references[0], str):
            references = [[ref] for ref in references]
        
        # Calculate metrics
        if "cider" in self.metrics:
            results["cider"] = self._calculate_cider(predictions, references)
        
        if "bleu" in self.metrics:
            results["bleu"] = self._calculate_bleu(predictions, references)
        
        if "meteor" in self.metrics:
            results["meteor"] = self._calculate_meteor(predictions, references)
        
        if "rouge" in self.metrics:
            rouge_scores = self._calculate_rouge(predictions, references)
            results.update(rouge_scores)
        
        if "bert_score" in self.metrics:
            bert_scores = self._calculate_bert_score(predictions, references)
            results.update(bert_scores)
        
        return results
    
    def _calculate_cider(
        self, 
        predictions: List[str], 
        references: List[List[str]]
    ) -> float:
        """Calculate CIDEr score.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference caption lists.
            
        Returns:
            CIDEr score.
        """
        try:
            from pycocoevalcap.cider.cider import Cider
            
            # Prepare data for CIDEr
            gts = {i: refs for i, refs in enumerate(references)}
            res = {i: [pred] for i, pred in enumerate(predictions)}
            
            # Calculate CIDEr
            cider = Cider()
            score, _ = cider.compute_score(gts, res)
            
            return float(score)
        except ImportError:
            print("Warning: pycocoevalcap not available, skipping CIDEr")
            return 0.0
    
    def _calculate_bleu(
        self, 
        predictions: List[str], 
        references: List[List[str]]
    ) -> float:
        """Calculate BLEU score.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference caption lists.
            
        Returns:
            BLEU score.
        """
        bleu_scores = []
        
        for pred, refs in zip(predictions, references):
            # Tokenize
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in refs]
            
            # Calculate BLEU
            bleu = sentence_bleu(
                ref_tokens, 
                pred_tokens, 
                smoothing_function=self.smoothing_function
            )
            bleu_scores.append(bleu)
        
        return np.mean(bleu_scores)
    
    def _calculate_meteor(
        self, 
        predictions: List[str], 
        references: List[List[str]]
    ) -> float:
        """Calculate METEOR score.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference caption lists.
            
        Returns:
            METEOR score.
        """
        meteor_scores = []
        
        for pred, refs in zip(predictions, references):
            # Tokenize
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in refs]
            
            # Calculate METEOR
            meteor = meteor_score(ref_tokens, pred_tokens)
            meteor_scores.append(meteor)
        
        return np.mean(meteor_scores)
    
    def _calculate_rouge(
        self, 
        predictions: List[str], 
        references: List[List[str]]
    ) -> Dict[str, float]:
        """Calculate ROUGE scores.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference caption lists.
            
        Returns:
            Dictionary of ROUGE scores.
        """
        rouge_scores = {metric: [] for metric in self.rouge_scorer.rouge_types}
        
        for pred, refs in zip(predictions, references):
            # Use the first reference for ROUGE calculation
            ref = refs[0]
            
            # Calculate ROUGE scores
            scores = self.rouge_scorer.score(ref, pred)
            
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Average scores
        return {metric: np.mean(scores) for metric, scores in rouge_scores.items()}
    
    def _calculate_bert_score(
        self, 
        predictions: List[str], 
        references: List[List[str]]
    ) -> Dict[str, float]:
        """Calculate BERTScore.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference caption lists.
            
        Returns:
            Dictionary of BERTScore metrics.
        """
        # Flatten references for BERTScore
        refs_flat = [refs[0] for refs in references]
        
        # Calculate BERTScore
        P, R, F1 = bert_score(
            predictions, 
            refs_flat, 
            model_type=self.bert_score_config.get("model_type", "microsoft/DialoGPT-medium"),
            lang=self.bert_score_config.get("lang", "en"),
            rescale_with_baseline=self.bert_score_config.get("rescale_with_baseline", True),
        )
        
        return {
            "bert_score_precision": float(P.mean()),
            "bert_score_recall": float(R.mean()),
            "bert_score_f1": float(F1.mean()),
        }
    
    def calculate_diversity_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics for generated captions.
        
        Args:
            predictions: List of predicted captions.
            
        Returns:
            Dictionary of diversity metrics.
        """
        # Tokenize all predictions
        all_tokens = []
        for pred in predictions:
            tokens = nltk.word_tokenize(pred.lower())
            all_tokens.extend(tokens)
        
        # Calculate distinct-n metrics
        distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
        
        # Calculate distinct-2
        bigrams = []
        for pred in predictions:
            tokens = nltk.word_tokenize(pred.lower())
            bigrams.extend([(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)])
        
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        
        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
        }
    
    def save_results(
        self, 
        results: Dict[str, float], 
        save_path: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results.
            save_path: Path to save results.
            additional_info: Additional information to save.
        """
        output = {
            "metrics": results,
            "config": self.config,
        }
        
        if additional_info:
            output.update(additional_info)
        
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
    
    def create_leaderboard(
        self, 
        results: Dict[str, Dict[str, float]], 
        save_path: Optional[str] = None
    ) -> str:
        """Create a leaderboard from multiple model results.
        
        Args:
            results: Dictionary mapping model names to their results.
            save_path: Optional path to save leaderboard.
            
        Returns:
            Formatted leaderboard string.
        """
        # Create leaderboard
        leaderboard = "Video Captioning Leaderboard\n"
        leaderboard += "=" * 50 + "\n\n"
        
        # Sort models by CIDEr score (if available)
        sorted_models = sorted(
            results.items(), 
            key=lambda x: x[1].get("cider", 0), 
            reverse=True
        )
        
        # Create table header
        metrics = ["Model", "CIDEr", "BLEU", "METEOR", "ROUGE-L", "BERTScore-F1"]
        leaderboard += " | ".join(f"{metric:>12}" for metric in metrics) + "\n"
        leaderboard += "-" * (12 * len(metrics) + 3 * (len(metrics) - 1)) + "\n"
        
        # Add model rows
        for model_name, scores in sorted_models:
            row = [
                model_name[:12],
                f"{scores.get('cider', 0):.4f}",
                f"{scores.get('bleu', 0):.4f}",
                f"{scores.get('meteor', 0):.4f}",
                f"{scores.get('rougeL', 0):.4f}",
                f"{scores.get('bert_score_f1', 0):.4f}",
            ]
            leaderboard += " | ".join(f"{item:>12}" for item in row) + "\n"
        
        if save_path:
            with open(save_path, "w") as f:
                f.write(leaderboard)
        
        return leaderboard
