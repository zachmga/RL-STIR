"""
Quick test to verify the tokenizer fix works.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from modeling import RLSTIRTokenizer

def test_tokenizer_training():
    """Test tokenizer training with small dataset"""
    print("Testing tokenizer training...")
    
    # Create sample text data
    sample_texts = [
        "<sysmon> <event_1> <template_10> powershell.exe <pid_5001> victim_user -EncodedCommand",
        "<auditd> <event_14> <template_20> sshd root authentication failure",
        "<evtx> <event_11> <template_40> encrypt.exe admin --encrypt --path=C:\\Users",
        "<sysmon> <event_1> <template_30> xmrig.exe <pid_6001> miner_user --pool=pool.example.com:4444",
        "<auth> <event_14> <template_20> sshd root authentication failure",
    ] * 100  # Repeat to get more data
    
    # Create tokenizer
    tokenizer = RLSTIRTokenizer(vocab_size=32000)
    
    # Test training
    try:
        tokenizer.train_tokenizer(sample_texts, "test_tokenizer.model")
        print("✅ Tokenizer training successful!")
        
        # Test encoding
        test_text = sample_texts[0]
        tokens = tokenizer.tokenize(test_text)
        decoded = tokenizer.detokenize(tokens)
        
        print(f"Original: {test_text}")
        print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
        print(f"Decoded: {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer training failed: {e}")
        return False

if __name__ == "__main__":
    success = test_tokenizer_training()
    if success:
        print("\n🎉 Tokenizer fix is working!")
    else:
        print("\n⚠️  Tokenizer still has issues.")
