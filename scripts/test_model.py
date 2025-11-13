#!/usr/bin/env python3
"""
Quick test script to verify model works without full training.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modeling.model import RLSTIRModel
from modeling.tokenizer import RLSTIRTokenizer
from data.schemas import LogRecord
import hydra
from omegaconf import DictConfig

def create_test_batch():
    """Create a minimal test batch"""
    batch_size = 2
    seq_len = 64
    num_nodes = 10
    num_edges = 15
    
    # Create test log tokens
    log_tokens = torch.randint(0, 1000, (batch_size, seq_len))
    log_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Create test graph data
    graph_x = torch.randn(num_nodes, 6)  # 6 features per node
    graph_edge_index = torch.randint(0, num_nodes, (2, num_edges))
    graph_node_type = torch.randint(0, 3, (num_nodes,))  # 0=proc, 1=file, 2=socket
    
    # Create a simple Data object
    from torch_geometric.data import Data
    graph_batch = Data(
        x=graph_x,
        edge_index=graph_edge_index,
        node_type=graph_node_type
    )
    
    # Create batch
    batch = {
        'log_tokens': log_tokens,
        'log_mask': log_mask,
        'graph': graph_batch,
        'ttp_labels': torch.randint(0, 80, (batch_size,)),
        'tamper_labels': torch.randint(0, 2, (batch_size,)).float()
    }
    
    return batch

def test_model_forward():
    """Test model forward pass"""
    print("Testing model forward pass...")
    
    # Create minimal config
    config = DictConfig({
        'model': {
            'd_model': 256,  # Smaller for testing
            'log_layers': 2,  # Fewer layers
            'nhead': 4,
            'ff': 512,
            'gnn_hidden': 128,
            'vocab_size': 1000,  # Smaller vocab
            'max_seq_len': 64,
            'graph_max_nodes': 100
        },
        'loss': {
            'ttp_weight': 1.0,
            'tamper_weight': 0.5,
            'aux_weight': 0.1
        }
    })
    
    # Create model
    model = RLSTIRModel(config)
    model.eval()
    
    # Create test batch
    batch = create_test_batch()
    
    # Test forward pass
    try:
        with torch.no_grad():
            outputs = model(batch, return_loss=True)
        
        print("✅ Model forward pass successful!")
        print(f"   Output keys: {list(outputs.keys())}")
        print(f"   Total loss: {outputs['total_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_placement():
    """Test device placement"""
    print("\nTesting device placement...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping device test")
        return True
    
    device = torch.device('cuda:0')
    
    # Create minimal config
    config = DictConfig({
        'model': {
            'd_model': 256,
            'log_layers': 2,
            'nhead': 4,
            'ff': 512,
            'gnn_hidden': 128,
            'vocab_size': 1000,
            'max_seq_len': 64,
            'graph_max_nodes': 100
        },
        'loss': {
            'ttp_weight': 1.0,
            'tamper_weight': 0.5,
            'aux_weight': 0.1
        }
    })
    
    # Create model and move to device
    model = RLSTIRModel(config).to(device)
    model.eval()
    
    # Create test batch and move to device
    batch = create_test_batch()
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch['graph'] = batch['graph'].to(device)
    
    try:
        with torch.no_grad():
            outputs = model(batch, return_loss=True)
        
        print("✅ Device placement test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Device placement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fp16():
    """Test FP16 training"""
    print("\nTesting FP16...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping FP16 test")
        return True
    
    device = torch.device('cuda:0')
    
    # Create minimal config
    config = DictConfig({
        'model': {
            'd_model': 256,
            'log_layers': 2,
            'nhead': 4,
            'ff': 512,
            'gnn_hidden': 128,
            'vocab_size': 1000,
            'max_seq_len': 64,
            'graph_max_nodes': 100
        },
        'loss': {
            'ttp_weight': 1.0,
            'tamper_weight': 0.5,
            'aux_weight': 0.1
        }
    })
    
    # Create model and move to device
    model = RLSTIRModel(config).to(device)
    model.train()
    
    # Create test batch and move to device
    batch = create_test_batch()
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch['graph'] = batch['graph'].to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    try:
        # Test FP16 forward pass
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(batch, return_loss=True)
            loss = outputs['total_loss']
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✅ FP16 test successful!")
        return True
        
    except Exception as e:
        print(f"❌ FP16 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Running quick model tests...\n")
    
    tests = [
        test_model_forward,
        test_device_placement,
        test_fp16
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Model is ready for training.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
