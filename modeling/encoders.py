"""
Neural network encoders for RL-STIR.
Includes log transformer and graph neural network encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GraphSAGE, GATv2Conv, HeteroConv
from torch_geometric.data import Data, HeteroData
from typing import Dict, Optional, Tuple, Union
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class LogTransformerEncoder(nn.Module):
    """Transformer encoder for processing log sequences"""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 1024,
                 activation: str = "gelu"):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,  # Use seq_len first for positional encoding
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        
        # Initialize other layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] - True for valid tokens
        
        Returns:
            hidden_states: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to key padding mask (True = ignore)
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # Transpose back and apply layer norm
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        x = self.layer_norm(x)
        
        return x


class HeteroGraphEncoder(nn.Module):
    """Heterogeneous graph encoder for process/file/socket nodes"""
    
    def __init__(self, 
                 node_feature_dims: Dict[str, int],
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_attention: bool = False):
        super().__init__()
        
        self.node_feature_dims = node_feature_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Node type embeddings
        self.node_type_embedding = nn.Embedding(3, hidden_dim)  # proc, file, socket
        
        # Feature projection layers for each node type with input normalization
        self.feature_projections = nn.ModuleDict({
            'proc': nn.Sequential(
                nn.LayerNorm(node_feature_dims['proc']),
                nn.Linear(node_feature_dims['proc'], hidden_dim)
            ),
            'file': nn.Sequential(
                nn.LayerNorm(node_feature_dims['file']),
                nn.Linear(node_feature_dims['file'], hidden_dim)
            ),
            'socket': nn.Sequential(
                nn.LayerNorm(node_feature_dims['socket']),
                nn.Linear(node_feature_dims['socket'], hidden_dim)
            )
        })
        
        # Graph convolution layers
        if use_attention:
            self.convs = nn.ModuleList([
                GATv2Conv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
                for _ in range(num_layers)
            ])
        else:
            self.convs = nn.ModuleList([
                GraphSAGE(hidden_dim, hidden_dim, num_layers=1, dropout=dropout)
                for _ in range(num_layers)
            ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent NaN values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for stability
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)  # Smaller std
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                node_type: torch.Tensor,
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, feature_dim] - node features
            edge_index: [2, num_edges] - edge connectivity
            node_type: [num_nodes] - node type (0=proc, 1=file, 2=socket)
            edge_type: [num_edges] - edge type (optional)
        
        Returns:
            node_embeddings: [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        
        # Check for NaN/Inf in input features and clip extreme values
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN/Inf detected in graph input features")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip extreme values to prevent overflow
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Check for NaN/Inf in node types
        if torch.isnan(node_type).any() or torch.isinf(node_type).any():
            print(f"Warning: NaN/Inf detected in node types")
            node_type = torch.nan_to_num(node_type, nan=0.0, posinf=1.0, neginf=-1.0).long()
        
        # Project features based on node type - vectorized approach
        # Create masks for each node type
        proc_mask = (node_type == 0)
        file_mask = (node_type == 1)
        socket_mask = (node_type == 2)
        
        # Initialize output tensor with same dtype as input
        x_projected = torch.zeros(num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        # Project features for each node type
        if proc_mask.any():
            proc_features = self.feature_projections['proc'](x[proc_mask])
            if torch.isnan(proc_features).any() or torch.isinf(proc_features).any():
                print(f"Warning: NaN/Inf detected in proc features after projection")
                proc_features = torch.nan_to_num(proc_features, nan=0.0, posinf=1.0, neginf=-1.0)
            x_projected[proc_mask] = proc_features.to(dtype=x.dtype)
            
        if file_mask.any():
            file_features = self.feature_projections['file'](x[file_mask])
            if torch.isnan(file_features).any() or torch.isinf(file_features).any():
                print(f"Warning: NaN/Inf detected in file features after projection")
                file_features = torch.nan_to_num(file_features, nan=0.0, posinf=1.0, neginf=-1.0)
            x_projected[file_mask] = file_features.to(dtype=x.dtype)
            
        if socket_mask.any():
            socket_features = self.feature_projections['socket'](x[socket_mask])
            if torch.isnan(socket_features).any() or torch.isinf(socket_features).any():
                print(f"Warning: NaN/Inf detected in socket features after projection")
                socket_features = torch.nan_to_num(socket_features, nan=0.0, posinf=1.0, neginf=-1.0)
            x_projected[socket_mask] = socket_features.to(dtype=x.dtype)
        
        x = x_projected
        
        # Check for NaN/Inf after feature projection
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN/Inf detected after feature projection")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Add node type embedding
        type_emb = self.node_type_embedding(node_type)
        
        # Check for NaN/Inf in type embedding
        if torch.isnan(type_emb).any() or torch.isinf(type_emb).any():
            print(f"Warning: NaN/Inf detected in type embedding")
            type_emb = torch.nan_to_num(type_emb, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x = x + type_emb
        
        # Check for NaN/Inf after adding type embedding
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN/Inf detected after adding type embedding")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply graph convolutions
        for i, (conv, layer_norm) in enumerate(zip(self.convs, self.layer_norms)):
            residual = x
            
            # Check for NaN/Inf before convolution
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN/Inf detected in graph features before conv layer {i}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if self.use_attention:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)
            
            # Check for NaN/Inf after convolution
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN/Inf detected in graph features after conv layer {i}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            x = layer_norm(x + residual)  # Residual connection
            
            # Check for NaN/Inf after layer norm
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN/Inf detected in graph features after layer norm {i}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        return x


class MultiModalEncoder(nn.Module):
    """Combines log transformer and graph encoder"""
    
    def __init__(self,
                 log_encoder: LogTransformerEncoder,
                 graph_encoder: HeteroGraphEncoder,
                 log_dim: int = 512,
                 graph_dim: int = 256,
                 fusion_dim: int = 768,
                 dropout: float = 0.1):
        super().__init__()
        
        self.log_encoder = log_encoder
        self.graph_encoder = graph_encoder
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(log_dim + graph_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion_dim = fusion_dim
    
    def forward(self,
                log_input_ids: torch.Tensor,
                log_attention_mask: torch.Tensor,
                graph_x: torch.Tensor,
                graph_edge_index: torch.Tensor,
                graph_node_type: torch.Tensor,
                graph_edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            log_input_ids: [batch_size, seq_len]
            log_attention_mask: [batch_size, seq_len]
            graph_x: [num_nodes, feature_dim]
            graph_edge_index: [2, num_edges]
            graph_node_type: [num_nodes]
            graph_edge_type: [num_edges] (optional)
        
        Returns:
            fused_embeddings: [batch_size, fusion_dim]
        """
        # Encode logs
        log_embeddings = self.log_encoder(log_input_ids, log_attention_mask)
        
        # Check for NaN/Inf in log embeddings
        if torch.isnan(log_embeddings).any() or torch.isinf(log_embeddings).any():
            print(f"Warning: NaN/Inf detected in log embeddings")
            log_embeddings = torch.nan_to_num(log_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Pool log embeddings (mean pooling over sequence)
        log_pooled = torch.mean(log_embeddings, dim=1)  # [batch_size, log_dim]
        
        # Check for NaN/Inf in log pooled
        if torch.isnan(log_pooled).any() or torch.isinf(log_pooled).any():
            print(f"Warning: NaN/Inf detected in log pooled")
            log_pooled = torch.nan_to_num(log_pooled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Encode graph
        graph_embeddings = self.graph_encoder(
            graph_x, graph_edge_index, graph_node_type, graph_edge_type
        )
        
        # Check for NaN/Inf in graph embeddings
        if torch.isnan(graph_embeddings).any() or torch.isinf(graph_embeddings).any():
            print(f"Warning: NaN/Inf detected in graph embeddings")
            graph_embeddings = torch.nan_to_num(graph_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Pool graph embeddings (mean pooling over nodes)
        graph_pooled = torch.mean(graph_embeddings, dim=0, keepdim=True)
        graph_pooled = graph_pooled.expand(log_pooled.size(0), -1)  # [batch_size, graph_dim]
        
        # Check for NaN/Inf in graph pooled
        if torch.isnan(graph_pooled).any() or torch.isinf(graph_pooled).any():
            print(f"Warning: NaN/Inf detected in graph pooled")
            graph_pooled = torch.nan_to_num(graph_pooled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Fuse embeddings
        combined = torch.cat([log_pooled, graph_pooled], dim=-1)
        
        # Check for NaN/Inf in combined
        if torch.isnan(combined).any() or torch.isinf(combined).any():
            print(f"Warning: NaN/Inf detected in combined features")
            combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
        
        fused = self.fusion(combined)
        
        return fused


class AttentionPooling(nn.Module):
    """Attention-based pooling for variable-length sequences"""
    
    def __init__(self, input_dim: int, attention_dim: int = 128):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len] - True for valid positions
        
        Returns:
            pooled: [batch_size, input_dim]
        """
        # Compute attention weights
        attn_weights = self.attention(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        
        # Softmax and apply
        attn_weights = F.softmax(attn_weights, dim=-1)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        
        return pooled


class GraphAttentionPooling(nn.Module):
    """Attention-based pooling for graph nodes"""
    
    def __init__(self, input_dim: int, attention_dim: int = 128):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, node_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, input_dim]
            node_type: [num_nodes] - node type for type-specific attention
        
        Returns:
            pooled: [input_dim]
        """
        # Compute attention weights
        attn_weights = self.attention(x).squeeze(-1)  # [num_nodes]
        
        # Apply softmax and pool
        attn_weights = F.softmax(attn_weights, dim=0)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=0)
        
        return pooled
