"""
Prediction heads for RL-STIR multi-task learning.
Includes TTP detection, tamper detection, and RL policy/value heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class TTPDetectionHead(nn.Module):
    """Head for TTP (Tactics, Techniques, and Procedures) detection"""
    
    def __init__(self, 
                 input_dim: int,
                 num_ttps: int = 80,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_ttps = num_ttps
        
        # Multi-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_ttps)
        )
        
        # TTP-specific embeddings for better representation
        self.ttp_embeddings = nn.Embedding(num_ttps, input_dim)
        
        # Attention mechanism for TTP-specific features
        self.ttp_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, 
                features: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, input_dim]
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with logits, probabilities, and optionally attention
        """
        batch_size = features.size(0)
        
        # Get TTP embeddings for attention
        ttp_embeddings = self.ttp_embeddings.weight  # [num_ttps, input_dim]
        ttp_embeddings = ttp_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply attention between features and TTP embeddings
        features_expanded = features.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        attended_features, attention_weights = self.ttp_attention(
            features_expanded, ttp_embeddings, ttp_embeddings
        )
        attended_features = attended_features.squeeze(1)  # [batch_size, input_dim]
        
        # Combine original and attended features
        combined_features = features + attended_features
        
        # Classify
        logits = self.classifier(combined_features)
        probabilities = F.softmax(logits, dim=-1)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'predicted_ttps': torch.argmax(logits, dim=-1)
        }
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result


class TamperDetectionHead(nn.Module):
    """Head for tamper detection (binary classification per token)"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # Binary classifier for tamper detection
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Span detection for contiguous tampered regions
        self.span_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # start, end positions
        )
    
    def forward(self, 
                features: torch.Tensor,
                sequence_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            sequence_lengths: [batch_size] - actual sequence lengths
        
        Returns:
            Dictionary with tamper predictions
        """
        if features.dim() == 2:
            # Single feature vector per sample
            logits = self.classifier(features)
            probabilities = torch.sigmoid(logits)
            
            return {
                'logits': logits,
                'probabilities': probabilities,
                'is_tampered': (probabilities > 0.5).float()
            }
        
        else:
            # Sequence of features
            batch_size, seq_len, input_dim = features.shape
            
            # Reshape for processing
            features_flat = features.view(-1, input_dim)
            logits_flat = self.classifier(features_flat)
            logits = logits_flat.view(batch_size, seq_len, 1)
            
            probabilities = torch.sigmoid(logits)
            
            # Span detection
            span_logits = self.span_classifier(features_flat)
            span_logits = span_logits.view(batch_size, seq_len, 2)
            
            return {
                'logits': logits,
                'probabilities': probabilities,
                'is_tampered': (probabilities > 0.5).float(),
                'span_logits': span_logits
            }


class RLPolicyHead(nn.Module):
    """Policy head for RL actions"""
    
    def __init__(self, 
                 input_dim: int,
                 num_actions: int = 24,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_actions = num_actions
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Action masking support
        self.action_embeddings = nn.Embedding(num_actions, input_dim)
    
    def forward(self, 
                features: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, input_dim]
            action_mask: [batch_size, num_actions] - True for valid actions
        
        Returns:
            Dictionary with policy outputs
        """
        # Check for NaN/Inf in input features
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"Warning: NaN/Inf detected in policy head input features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get action logits
        logits = self.policy_net(features)
        
        # Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: NaN/Inf detected in policy logits")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply action mask if provided
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        # Compute probabilities with numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        # Check for NaN/Inf in probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"Warning: NaN/Inf detected in policy probabilities")
            # Fallback to uniform distribution
            probs = torch.ones_like(probs) / probs.size(-1)
            log_probs = torch.log(probs)
        
        # Sample actions
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        
        return {
            'logits': logits,
            'log_probs': log_probs,
            'probabilities': probs,
            'actions': actions,
            'entropy': dist.entropy()
        }


class RLValueHead(nn.Module):
    """Value head for RL state value estimation"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage estimation (for A2C-style algorithms)
        self.advantage_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, input_dim]
        
        Returns:
            Dictionary with value estimates
        """
        # State value
        values = self.value_net(features).squeeze(-1)
        
        # Advantage (for A2C)
        advantages = self.advantage_net(features).squeeze(-1)
        
        return {
            'values': values,
            'advantages': advantages
        }


class MultiTaskHead(nn.Module):
    """Combines all prediction heads for multi-task learning"""
    
    def __init__(self,
                 input_dim: int,
                 num_ttps: int = 80,
                 num_actions: int = 24,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # Individual heads
        self.ttp_head = TTPDetectionHead(input_dim, num_ttps, hidden_dim, dropout)
        self.tamper_head = TamperDetectionHead(input_dim, hidden_dim, dropout)
        self.policy_head = RLPolicyHead(input_dim, num_actions, hidden_dim, dropout)
        self.value_head = RLValueHead(input_dim, hidden_dim, dropout)
        
        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent NaN values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, 
                features: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, input_dim]
            action_mask: [batch_size, num_actions] - for policy head
            return_attention: Whether to return attention weights from TTP head
        
        Returns:
            Dictionary with all head outputs
        """
        # Check for NaN/Inf in input features
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"Warning: NaN/Inf detected in multi-task head input features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Process features
        processed_features = self.feature_processor(features)
        
        # Check for NaN/Inf in processed features
        if torch.isnan(processed_features).any() or torch.isinf(processed_features).any():
            print(f"Warning: NaN/Inf detected in processed features")
            processed_features = torch.nan_to_num(processed_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get outputs from all heads
        outputs = {}
        
        # TTP detection
        ttp_outputs = self.ttp_head(processed_features, return_attention)
        outputs.update({f'ttp_{k}': v for k, v in ttp_outputs.items()})
        
        # Tamper detection
        tamper_outputs = self.tamper_head(processed_features)
        outputs.update({f'tamper_{k}': v for k, v in tamper_outputs.items()})
        
        # RL policy
        policy_outputs = self.policy_head(processed_features, action_mask)
        outputs.update({f'policy_{k}': v for k, v in policy_outputs.items()})
        
        # RL value
        value_outputs = self.value_head(processed_features)
        outputs.update({f'value_{k}': v for k, v in value_outputs.items()})
        
        return outputs


class UncertaintyHead(nn.Module):
    """Head for uncertainty estimation (useful for active learning)"""
    
    def __init__(self, 
                 input_dim: int,
                 num_samples: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_samples = num_samples
        
        # Monte Carlo dropout layers
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, 
                features: torch.Tensor,
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, input_dim]
            training: Whether to use dropout for uncertainty estimation
        
        Returns:
            Dictionary with uncertainty estimates
        """
        if training:
            # Monte Carlo dropout for uncertainty estimation
            predictions = []
            for _ in range(self.num_samples):
                pred = self.uncertainty_net(features)
                predictions.append(pred)
            
            predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, 1]
            
            # Compute mean and variance
            mean_pred = torch.mean(predictions, dim=0)
            var_pred = torch.var(predictions, dim=0)
            uncertainty = torch.sqrt(var_pred + 1e-8)
            
            return {
                'predictions': mean_pred,
                'uncertainty': uncertainty,
                'variance': var_pred
            }
        else:
            # Single prediction without dropout
            prediction = self.uncertainty_net(features)
            return {
                'predictions': prediction,
                'uncertainty': torch.zeros_like(prediction)
            }
