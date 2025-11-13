"""
Test script to verify RL-STIR setup.
Tests imports, model creation, and basic functionality.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from data.schemas import LogRecord, LogChannel
        print("✅ Data schemas imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data schemas: {e}")
        return False
    
    try:
        from modeling import create_model, RLSTIRTokenizer
        print("✅ Modeling modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import modeling modules: {e}")
        return False
    
    try:
        from envs.sim import SyntheticEpisodeGenerator
        print("✅ Environment modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import environment modules: {e}")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA is available: {device_name}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   PyTorch version: {torch.__version__}")
        return True
    else:
        print("❌ CUDA is not available")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        from modeling import create_model
        import hydra
        from omegaconf import DictConfig
        
        # Create a minimal config
        config = DictConfig({
            'model': {
                'd_model': 512,
                'log_layers': 8,
                'nhead': 8,
                'ff': 2048,
                'gnn_hidden': 256,
                'vocab_size': 32000,
                'max_seq_len': 1024,
                'graph_max_nodes': 5000
            },
            'loss': {
                'ttp_weight': 1.0,
                'tamper_weight': 0.5,
                'aux_weight': 0.1
            }
        })
        
        model = create_model(config)
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with dummy data
        batch_size = 2
        seq_len = 100
        
        dummy_batch = {
            'log_tokens': torch.randint(0, 1000, (batch_size, seq_len)),
            'log_mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
            'graph': {
                'x': torch.randn(10, 6),
                'edge_index': torch.randint(0, 10, (2, 20)),
                'node_type': torch.randint(0, 3, (10,)),
                'edge_type': torch.randint(0, 4, (20,))
            },
            'ttp_labels': torch.randint(0, 80, (batch_size,)),
            'tamper_labels': torch.randint(0, 2, (batch_size,)).float()
        }
        
        # Convert graph to PyTorch Geometric format
        from torch_geometric.data import Data, Batch
        graph_data = Data(
            x=dummy_batch['graph']['x'],
            edge_index=dummy_batch['graph']['edge_index'],
            node_type=dummy_batch['graph']['node_type'],
            edge_type=dummy_batch['graph']['edge_type']
        )
        dummy_batch['graph'] = Batch.from_data_list([graph_data])
        
        with torch.no_grad():
            outputs = model(dummy_batch, return_loss=True)
        
        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_episode_generation():
    """Test synthetic episode generation"""
    print("\nTesting episode generation...")
    
    try:
        from envs.sim import SyntheticEpisodeGenerator
        
        generator = SyntheticEpisodeGenerator(output_dir="test_episodes")
        
        # Generate a small episode
        episode_path = generator.generate_episode("phishing_script_lolbin", "test_episode")
        print(f"✅ Episode generated successfully: {episode_path}")
        
        # Check if files exist
        episode_dir = Path(episode_path)
        required_files = ["logs.parquet", "proc_nodes.parquet", "metadata.json"]
        
        for file_name in required_files:
            file_path = episode_dir / file_name
            if file_path.exists():
                print(f"   ✅ {file_name} exists")
            else:
                print(f"   ❌ {file_name} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Episode generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer():
    """Test tokenizer functionality"""
    print("\nTesting tokenizer...")
    
    try:
        from modeling import RLSTIRTokenizer
        from data.schemas import LogRecord, LogChannel
        
        tokenizer = RLSTIRTokenizer(vocab_size=1000)
        
        # Create a dummy log record
        log_record = LogRecord(
            host="test-host",
            ts_ns=1234567890,
            channel=LogChannel.SYSMON,
            template_id=10,
            pid=1234,
            ppid=5678,
            tid=1,
            exe="powershell.exe",
            cmdline="powershell -EncodedCommand <base64>",
            user="testuser",
            sid="S-1-5-21-test",
            event_id=1,
            fields_json='{"test": "value"}',
            label_ttp="T1059.001",
            tamper_tag=False
        )
        
        # Test text encoding
        text = tokenizer.encode_log_record(log_record)
        print(f"✅ Log record encoded to text: {text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 RL-STIR Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_cuda,
        test_model_creation,
        test_episode_generation,
        test_tokenizer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! RL-STIR setup is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
