"""
Main RL-STIR model combining encoders and heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import hydra
from omegaconf import DictConfig

from .encoders import LogTransformerEncoder, HeteroGraphEncoder, MultiModalEncoder
from .heads import MultiTaskHead, UncertaintyHead


class RLSTIRModel(nn.Module):
    """Main RL-STIR model for multi-task learning"""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.config = config
        self.model_config = config.model
        
        # Encoders
        self.log_encoder = LogTransformerEncoder(
            vocab_size=self.model_config.vocab_size,
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.log_layers,
            dim_feedforward=self.model_config.ff,
            max_seq_len=self.model_config.max_seq_len
        )
        
        # Graph encoder
        node_feature_dims = {
            'proc': 6,   # Based on ProcNode schema
            'file': 6,   # Based on FileNode schema  
            'socket': 6  # Based on SocketNode schema
        }
        
        self.graph_encoder = HeteroGraphEncoder(
            node_feature_dims=node_feature_dims,
            hidden_dim=self.model_config.gnn_hidden,
            num_layers=2,
            use_attention=False  # Use GraphSAGE for efficiency
        )
        
        # Multi-modal encoder
        self.multimodal_encoder = MultiModalEncoder(
            log_encoder=self.log_encoder,
            graph_encoder=self.graph_encoder,
            log_dim=self.model_config.d_model,
            graph_dim=self.model_config.gnn_hidden,
            fusion_dim=self.model_config.d_model + self.model_config.gnn_hidden
        )
        
        # Multi-task heads
        self.multi_task_head = MultiTaskHead(
            input_dim=self.model_config.d_model + self.model_config.gnn_hidden,
            num_ttps=80,  # MITRE ATT&CK techniques
            num_actions=24,  # Investigation actions
            hidden_dim=256,
            dropout=0.1
        )
        
        # Uncertainty estimation (optional)
        if hasattr(config, 'uncertainty') and config.uncertainty.enabled:
            self.uncertainty_head = UncertaintyHead(
                input_dim=self.model_config.d_model + self.model_config.gnn_hidden,
                num_samples=config.uncertainty.num_samples
            )
        else:
            self.uncertainty_head = None
        
        # Loss weights
        self.loss_weights = {
            'ttp': config.loss.ttp_weight,
            'tamper': config.loss.tamper_weight,
            'policy': 1.0,  # RL loss weight
            'value': 1.0,   # RL loss weight
            'aux': config.loss.aux_weight
        }
    
    def forward(self, 
                batch: Dict[str, torch.Tensor],
                return_loss: bool = False,
                return_uncertainty: bool = False) -> Dict[str, Any]:
        """
        Forward pass through the model
        
        Args:
            batch: Dictionary containing:
                - log_tokens: [batch_size, seq_len]
                - log_mask: [batch_size, seq_len]
                - graph: PyTorch Geometric Batch object
                - ttp_labels: [batch_size] (optional)
                - tamper_labels: [batch_size] (optional)
                - action_mask: [batch_size, num_actions] (optional)
            return_loss: Whether to compute and return losses
            return_uncertainty: Whether to compute uncertainty estimates
        
        Returns:
            Dictionary with model outputs and optionally losses
        """
        # Extract inputs
        log_tokens = batch['log_tokens']
        log_mask = batch['log_mask']
        graph_batch = batch['graph']
        
        # Extract graph components
        graph_x = graph_batch.x
        graph_edge_index = graph_batch.edge_index
        graph_node_type = graph_batch.node_type
        graph_edge_type = getattr(graph_batch, 'edge_type', None)
        
        # Multi-modal encoding
        fused_features = self.multimodal_encoder(
            log_input_ids=log_tokens,
            log_attention_mask=log_mask,
            graph_x=graph_x,
            graph_edge_index=graph_edge_index,
            graph_node_type=graph_node_type,
            graph_edge_type=graph_edge_type
        )
        
        # Check for NaN/Inf in fused features
        if torch.isnan(fused_features).any() or torch.isinf(fused_features).any():
            print(f"Warning: NaN/Inf detected in fused features")
            fused_features = torch.nan_to_num(fused_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Multi-task predictions
        action_mask = batch.get('action_mask', None)
        outputs = self.multi_task_head(
            features=fused_features,
            action_mask=action_mask,
            return_attention=return_uncertainty
        )
        
        # Uncertainty estimation (if enabled)
        if return_uncertainty and self.uncertainty_head is not None:
            uncertainty_outputs = self.uncertainty_head(fused_features, training=self.training)
            outputs.update({f'uncertainty_{k}': v for k, v in uncertainty_outputs.items()})
        
        # Compute losses if requested
        if return_loss:
            losses = self.compute_losses(outputs, batch)
            outputs['losses'] = losses
            outputs['total_loss'] = sum(losses.values())
        
        return outputs
    
    def compute_losses(self, 
                      outputs: Dict[str, torch.Tensor], 
                      batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all losses for multi-task learning"""
        losses = {}
        
        # TTP detection loss
        if 'ttp_labels' in batch:
            ttp_logits = outputs['ttp_logits']
            ttp_labels = batch['ttp_labels']
            ttp_loss = F.cross_entropy(ttp_logits, ttp_labels)
            losses['ttp_loss'] = ttp_loss * self.loss_weights['ttp']
        
        # Tamper detection loss
        if 'tamper_labels' in batch:
            tamper_logits = outputs['tamper_logits']
            tamper_labels = batch['tamper_labels']
            tamper_loss = F.binary_cross_entropy_with_logits(
                tamper_logits.squeeze(-1), tamper_labels
            )
            losses['tamper_loss'] = tamper_loss * self.loss_weights['tamper']
        
        # RL losses (will be computed by RL algorithm)
        # These are placeholders - actual RL losses computed in training loop
        if 'policy_logits' in outputs:
            # Policy entropy (for exploration)
            policy_entropy = outputs['policy_entropy']
            losses['entropy_loss'] = -policy_entropy.mean() * 0.01  # Small weight
        
        # Auxiliary losses (calibration, etc.)
        if 'aux_features' in batch:
            # Example: sequence length prediction as auxiliary task
            aux_loss = self.compute_auxiliary_loss(outputs, batch['aux_features'])
            losses['aux_loss'] = aux_loss * self.loss_weights['aux']
        
        return losses
    
    def compute_auxiliary_loss(self, 
                              outputs: Dict[str, torch.Tensor], 
                              aux_features: Dict[str, Any]) -> torch.Tensor:
        """Compute auxiliary losses for better representation learning"""
        # Example: predict number of logs as auxiliary task
        if 'num_logs' in aux_features:
            # Simple regression head for auxiliary task
            num_logs = torch.tensor(aux_features['num_logs'], dtype=torch.float)
            # This would require an additional head - simplified for now
            return torch.tensor(0.0, device=outputs['ttp_logits'].device)
        
        return torch.tensor(0.0, device=outputs['ttp_logits'].device)
    
    def get_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get fused embeddings for analysis or transfer learning"""
        with torch.no_grad():
            log_tokens = batch['log_tokens']
            log_mask = batch['log_mask']
            graph_batch = batch['graph']
            
            graph_x = graph_batch.x
            graph_edge_index = graph_batch.edge_index
            graph_node_type = graph_batch.node_type
            graph_edge_type = getattr(graph_batch, 'edge_type', None)
            
            fused_features = self.multimodal_encoder(
                log_input_ids=log_tokens,
                log_attention_mask=log_mask,
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,
                graph_node_type=graph_node_type,
                graph_edge_type=graph_edge_type
            )
            
            return fused_features
    
    def predict_ttp(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict TTPs for a batch"""
        with torch.no_grad():
            outputs = self.forward(batch, return_loss=False)
            return {
                'ttp_predictions': outputs['ttp_predicted_ttps'],
                'ttp_probabilities': outputs['ttp_probabilities'],
                'ttp_logits': outputs['ttp_logits']
            }
    
    def predict_tamper(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict tampering for a batch"""
        with torch.no_grad():
            outputs = self.forward(batch, return_loss=False)
            return {
                'tamper_predictions': outputs['tamper_is_tampered'],
                'tamper_probabilities': outputs['tamper_probabilities'],
                'tamper_logits': outputs['tamper_logits']
            }
    
    def get_action_distribution(self, 
                               batch: Dict[str, torch.Tensor],
                               action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Get action distribution for RL"""
        with torch.no_grad():
            outputs = self.forward(batch, return_loss=False)
            return {
                'action_probs': outputs['policy_probabilities'],
                'action_logits': outputs['policy_logits'],
                'actions': outputs['policy_actions'],
                'entropy': outputs['policy_entropy']
            }
    
    def get_state_value(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get state value for RL"""
        with torch.no_grad():
            outputs = self.forward(batch, return_loss=False)
            return outputs['value_values']


def create_model(config: DictConfig) -> RLSTIRModel:
    """Factory function to create RL-STIR model"""
    return RLSTIRModel(config)


def load_model_from_checkpoint(checkpoint_path: str, config: DictConfig) -> RLSTIRModel:
    """Load model from checkpoint"""
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def save_model_checkpoint(model: RLSTIRModel, 
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         loss: float,
                         checkpoint_path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config
    }
    torch.save(checkpoint, checkpoint_path)
