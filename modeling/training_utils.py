"""
Training utilities for RL-STIR.
Includes loss functions, metrics, and training helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import f1_score, precision_recall_curve, auc
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] - logits
            targets: [batch_size] - class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] - logits
            targets: [batch_size] - class indices
        """
        num_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))


class DiceLoss(nn.Module):
    """Dice Loss for segmentation-like tasks (tamper detection)"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, seq_len] - probabilities
            targets: [batch_size, seq_len] - binary targets
        """
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class MetricsCalculator:
    """Calculate various metrics for evaluation"""
    
    def __init__(self, num_classes: int = 80):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               probabilities: Optional[torch.Tensor] = None):
        """Update metrics with new batch"""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = (predictions == targets).mean()
        
        # F1 scores
        metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(targets, predictions, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Per-class F1
        f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
        for i, f1 in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1
        
        # AUPR if probabilities available
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            if probabilities.ndim == 2:  # Multi-class
                # Compute AUPR for each class
                for i in range(probabilities.shape[1]):
                    precision, recall, _ = precision_recall_curve(
                        (targets == i).astype(int), probabilities[:, i]
                    )
                    aupr = auc(recall, precision)
                    metrics[f'aupr_class_{i}'] = aupr
            else:  # Binary
                precision, recall, _ = precision_recall_curve(targets, probabilities)
                metrics['aupr'] = auc(recall, precision)
        
        return metrics


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Args:
            score: Current validation score
            model: Model to potentially restore
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()


class LearningRateScheduler:
    """Custom learning rate scheduler"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 warmup_steps: int = 1000,
                 max_lr: float = 3e-4,
                 min_lr: float = 1e-6,
                 total_steps: int = 100000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class GradientClipper:
    """Gradient clipping utility"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients and return the gradient norm
        
        Args:
            model: Model to clip gradients for
        
        Returns:
            Gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )
        return total_norm.item()


class ModelEMA:
    """Exponential Moving Average of model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update EMA parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model: nn.Module):
        """Apply shadow parameters to model"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def compute_calibration_metrics(predictions: torch.Tensor, 
                               targets: torch.Tensor,
                               num_bins: int = 10) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE, MCE)
    
    Args:
        predictions: [batch_size, num_classes] - predicted probabilities
        targets: [batch_size] - true class labels
        num_bins: Number of bins for calibration
    
    Returns:
        Dictionary with calibration metrics
    """
    predictions = F.softmax(predictions, dim=-1)
    max_probs, predicted_classes = torch.max(predictions, dim=-1)
    
    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            # Compute accuracy in this bin
            accuracy_in_bin = (predicted_classes[in_bin] == targets[in_bin]).float().mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            
            # Update ECE and MCE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            mce = max(mce, torch.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return {
        'ece': ece.item(),
        'mce': mce.item()
    }


def compute_uncertainty_metrics(predictions: torch.Tensor,
                               uncertainties: torch.Tensor,
                               targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute uncertainty-aware metrics
    
    Args:
        predictions: [batch_size] - predicted classes
        uncertainties: [batch_size] - uncertainty estimates
        targets: [batch_size] - true labels
    
    Returns:
        Dictionary with uncertainty metrics
    """
    # Accuracy vs uncertainty
    correct = (predictions == targets).float()
    
    # Sort by uncertainty
    sorted_indices = torch.argsort(uncertainties, descending=True)
    sorted_correct = correct[sorted_indices]
    
    # Compute accuracy at different uncertainty thresholds
    total_samples = len(correct)
    accuracy_at_uncertainty = {}
    
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        high_uncertainty_mask = uncertainties > threshold
        if high_uncertainty_mask.sum() > 0:
            acc = correct[high_uncertainty_mask].mean().item()
            accuracy_at_uncertainty[f'acc_uncertainty_{threshold}'] = acc
    
    # Uncertainty correlation with error
    error = 1 - correct
    uncertainty_error_corr = torch.corrcoef(torch.stack([uncertainties, error]))[0, 1].item()
    
    return {
        'uncertainty_error_correlation': uncertainty_error_corr,
        **accuracy_at_uncertainty
    }
