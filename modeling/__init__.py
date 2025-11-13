"""
RL-STIR modeling package.
Contains encoders, heads, and the main model.
"""

from .encoders import (
    LogTransformerEncoder,
    HeteroGraphEncoder, 
    MultiModalEncoder,
    PositionalEncoding,
    AttentionPooling,
    GraphAttentionPooling
)

from .heads import (
    TTPDetectionHead,
    TamperDetectionHead,
    RLPolicyHead,
    RLValueHead,
    MultiTaskHead,
    UncertaintyHead
)

from .model import (
    RLSTIRModel,
    create_model,
    load_model_from_checkpoint,
    save_model_checkpoint
)

from .tokenizer import (
    RLSTIRTokenizer,
    RLSTIRCollator,
    RLSTIRDataset,
    create_dataloader
)

__all__ = [
    # Encoders
    'LogTransformerEncoder',
    'HeteroGraphEncoder',
    'MultiModalEncoder', 
    'PositionalEncoding',
    'AttentionPooling',
    'GraphAttentionPooling',
    
    # Heads
    'TTPDetectionHead',
    'TamperDetectionHead',
    'RLPolicyHead',
    'RLValueHead',
    'MultiTaskHead',
    'UncertaintyHead',
    
    # Main model
    'RLSTIRModel',
    'create_model',
    'load_model_from_checkpoint',
    'save_model_checkpoint',
    
    # Tokenizer and data
    'RLSTIRTokenizer',
    'RLSTIRCollator',
    'RLSTIRDataset',
    'create_dataloader'
]
