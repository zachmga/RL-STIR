"""
Tokenizer and data collator for RL-STIR.
Handles text tokenization and GPU-friendly data preparation.
"""

import json
import pickle
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentencepiece import SentencePieceProcessor
from torch_geometric.data import Data, Batch
import networkx as nx

from data.schemas import LogRecord, LogChannel, ProcNode, FileNode, SocketNode, GraphEdge


class RLSTIRTokenizer:
    """Tokenizer for RL-STIR log data using SentencePiece"""
    
    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.sp_model = None
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.mask_token = "<mask>"
        
        # Channel tokens
        self.channel_tokens = {
            LogChannel.SYSMON: "<sysmon>",
            LogChannel.AUDITD: "<auditd>",
            LogChannel.AUTH: "<auth>",
            LogChannel.EVTX: "<evtx>"
        }
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train_tokenizer(self, text_data: List[str], output_path: str):
        """Train SentencePiece tokenizer on log data"""
        from sentencepiece import SentencePieceTrainer
        
        # Prepare training data
        train_file = Path(output_path).parent / "train_data.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in text_data:
                f.write(text + '\n')
        
        # Adjust vocabulary size based on data size
        # SentencePiece requires vocab_size to be much smaller than data size
        data_size = len(text_data)
        # Use a conservative ratio and cap: vocab_size = min(1800, max(1000, data_size // 15))
        max_vocab = min(1800, max(1000, data_size // 15))
        
        # Train SentencePiece model
        SentencePieceTrainer.train(
            input=train_file,
            model_prefix=output_path.replace('.model', ''),
            vocab_size=max_vocab,
            character_coverage=0.9995,
            model_type='unigram',
            user_defined_symbols=[
                self.pad_token, self.bos_token, self.eos_token, self.mask_token
            ] + list(self.channel_tokens.values()),
            hard_vocab_limit=False
        )
        
        # Load the trained model
        self.load_model(f"{output_path.replace('.model', '')}.model")
        
        # Clean up training file
        train_file.unlink()
    
    def load_model(self, model_path: str):
        """Load pre-trained SentencePiece model"""
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path
    
    def encode_log_record(self, record: LogRecord) -> str:
        """Convert log record to tokenized text"""
        tokens = []
        
        # Channel token
        if record.channel in self.channel_tokens:
            tokens.append(self.channel_tokens[record.channel])
        
        # Event ID
        if record.event_id is not None:
            tokens.append(f"<event_{record.event_id}>")
        
        # Template ID
        tokens.append(f"<template_{record.template_id}>")
        
        # Executable basename
        if record.exe:
            exe_basename = Path(record.exe).name
            tokens.append(exe_basename)
        
        # Process information
        if record.pid is not None:
            tokens.append(f"<pid_{record.pid}>")
        
        if record.ppid is not None:
            tokens.append(f"<ppid_{record.ppid}>")
        
        # User
        if record.user:
            tokens.append(record.user)
        
        # Command line (truncated)
        if record.cmdline:
            cmdline_tokens = record.cmdline.split()[:10]  # Limit to 10 tokens
            tokens.extend(cmdline_tokens)
        
        # Additional fields from JSON
        try:
            fields = json.loads(record.fields_json)
            for key, value in fields.items():
                if isinstance(value, (str, int, float)):
                    tokens.append(str(value))
        except (json.JSONDecodeError, TypeError):
            pass
        
        return " ".join(tokens)
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs"""
        if self.sp_model is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        return self.sp_model.encode(text, out_type=int)
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        if self.sp_model is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        return self.sp_model.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.sp_model is None:
            return self.vocab_size
        return self.sp_model.get_piece_size()
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs"""
        if self.sp_model is None:
            return {}
        
        return {
            'pad': self.sp_model.piece_to_id(self.pad_token),
            'unk': self.sp_model.piece_to_id(self.unk_token),
            'bos': self.sp_model.piece_to_id(self.bos_token),
            'eos': self.sp_model.piece_to_id(self.eos_token),
            'mask': self.sp_model.piece_to_id(self.mask_token)
        }


class RLSTIRCollator:
    """Data collator for RL-STIR batches"""
    
    def __init__(self, tokenizer: RLSTIRTokenizer, max_seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.special_tokens = tokenizer.get_special_token_ids()
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples"""
        batch_size = len(batch)
        
        # Prepare log tokens
        log_tokens = []
        log_masks = []
        
        for sample in batch:
            tokens = sample['log_tokens']
            
            # Truncate or pad to max_seq_len
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            else:
                # Pad with pad token
                pad_id = self.special_tokens.get('pad', 0)
                tokens.extend([pad_id] * (self.max_seq_len - len(tokens)))
            
            # Create attention mask
            mask = [1] * len(sample['log_tokens']) + [0] * (self.max_seq_len - len(sample['log_tokens']))
            if len(mask) > self.max_seq_len:
                mask = mask[:self.max_seq_len]
            
            log_tokens.append(tokens)
            log_masks.append(mask)
        
        # Convert to tensors
        log_tokens = torch.tensor(log_tokens, dtype=torch.long)
        log_masks = torch.tensor(log_masks, dtype=torch.bool)
        
        # Prepare auxiliary features
        aux_features = []
        for sample in batch:
            aux = sample.get('aux_features', {})
            aux_features.append(aux)
        
        result = {
            'log_tokens': log_tokens,
            'log_mask': log_masks,
            'aux_features': aux_features
        }
        
        # Add graph data if present
        if 'graph' in batch[0]:
            graphs = [sample['graph'] for sample in batch]
            result['graph'] = Batch.from_data_list(graphs)
        
        # Add labels if present
        if 'ttp_labels' in batch[0]:
            ttp_labels = torch.tensor([sample['ttp_labels'] for sample in batch], dtype=torch.long)
            result['ttp_labels'] = ttp_labels
        
        if 'tamper_labels' in batch[0]:
            tamper_labels = torch.tensor([sample['tamper_labels'] for sample in batch], dtype=torch.float)
            result['tamper_labels'] = tamper_labels
        
        return result


class RLSTIRDataset(Dataset):
    """Dataset for RL-STIR episodes"""
    
    def __init__(self, episodes_dir: str, tokenizer: RLSTIRTokenizer, 
                 max_seq_len: int = 1024, max_graph_nodes: int = 5000):
        self.episodes_dir = Path(episodes_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_graph_nodes = max_graph_nodes
        
        # Find all episode directories
        self.episode_dirs = [d for d in self.episodes_dir.iterdir() if d.is_dir()]
        
        # Load all episodes
        self.episodes = []
        for episode_dir in self.episode_dirs:
            episode_data = self._load_episode(episode_dir)
            if episode_data is not None:
                self.episodes.append(episode_data)
    
    def _load_episode(self, episode_dir: Path) -> Optional[Dict]:
        """Load a single episode"""
        try:
            # Load logs
            logs_df = pd.read_parquet(episode_dir / "logs.parquet")
            
            # Load graph data
            proc_nodes = pd.read_parquet(episode_dir / "proc_nodes.parquet")
            file_nodes = pd.read_parquet(episode_dir / "file_nodes.parquet")
            socket_nodes = pd.read_parquet(episode_dir / "sock_nodes.parquet")
            edges = pd.read_parquet(episode_dir / "edges.parquet")
            
            # Load metadata
            with open(episode_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Load ground truth
            with open(episode_dir / "gt_ttp_sequence.json", 'r') as f:
                ttp_gt = json.load(f)
            
            return {
                'logs': logs_df,
                'proc_nodes': proc_nodes,
                'file_nodes': file_nodes,
                'socket_nodes': socket_nodes,
                'edges': edges,
                'metadata': metadata,
                'ttp_gt': ttp_gt
            }
        except Exception as e:
            print(f"Error loading episode {episode_dir}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        episode = self.episodes[idx]
        
        # Convert logs to tokens
        log_texts = []
        for _, log_row in episode['logs'].iterrows():
            # Convert to LogRecord with NaN-safe handling for optional numeric fields
            row_dict = log_row.to_dict()
            for field in ['pid', 'ppid', 'tid', 'event_id']:
                if field in row_dict and pd.isna(row_dict[field]):
                    row_dict[field] = None
            log_record = LogRecord(**row_dict)
            text = self.tokenizer.encode_log_record(log_record)
            log_texts.append(text)
        
        # Tokenize all log texts
        all_tokens = []
        for text in log_texts:
            tokens = self.tokenizer.tokenize(text)
            all_tokens.extend(tokens)
        
        # Truncate if too long
        if len(all_tokens) > self.max_seq_len:
            all_tokens = all_tokens[:self.max_seq_len]
        
        # Build graph
        graph = self._build_graph(episode)
        
        # Prepare labels
        ttp_labels = self._prepare_ttp_labels(episode)
        tamper_labels = self._prepare_tamper_labels(episode)
        
        # Auxiliary features
        aux_features = {
            'num_logs': len(episode['logs']),
            'num_nodes': len(episode['proc_nodes']) + len(episode['file_nodes']) + len(episode['socket_nodes']),
            'num_edges': len(episode['edges']),
            'duration': episode['metadata']['duration_ns']
        }
        
        return {
            'log_tokens': all_tokens,
            'graph': graph,
            'ttp_labels': ttp_labels,
            'tamper_labels': tamper_labels,
            'aux_features': aux_features
        }
    
    def _build_graph(self, episode: Dict) -> Data:
        """Build PyTorch Geometric graph from episode data"""
        # Combine all nodes
        all_nodes = []
        node_types = []
        node_features = []
        
        # Process nodes
        for _, row in episode['proc_nodes'].iterrows():
            all_nodes.append(f"proc_{row['pid']}")
            node_types.append(0)  # Process type
            features = [
                row['first_ts'] / 1e12,  # Normalize timestamp
                row['last_ts'] / 1e12,
                1.0 if row['signed'] else 0.0,
                1.0 if row['rare_parent'] else 0.0,
                hash(row['user']) % 1000 / 1000.0,  # Hash user to float
                len(row['cmdline']) / 1000.0  # Normalize cmdline length
            ]
            node_features.append(features)
        
        # File nodes
        for _, row in episode['file_nodes'].iterrows():
            all_nodes.append(f"file_{len(all_nodes)}")
            node_types.append(1)  # File type
            features = [
                row['entropy'] / 8.0,  # Normalize entropy
                row['size'] / 1e6,  # Normalize size to MB
                row['created_ts'] / 1e12,
                row['modified_ts'] / 1e12,
                0.0, 0.0  # Padding
            ]
            node_features.append(features)
        
        # Socket nodes
        for _, row in episode['socket_nodes'].iterrows():
            all_nodes.append(f"sock_{len(all_nodes)}")
            node_types.append(2)  # Socket type
            features = [
                1.0 if row['l4_proto'] == 'TCP' else 0.0,
                row['dst_port'] / 65535.0,  # Normalize port
                hash(row['dst_ip']) % 1000 / 1000.0,  # Hash IP
                (row['asn'] or 0) / 100000.0,  # Normalize ASN
                0.0, 0.0  # Padding
            ]
            node_features.append(features)
        
        # Create node mapping
        node_map = {node: i for i, node in enumerate(all_nodes)}
        
        # Build edges
        edge_indices = []
        edge_types = []
        
        for _, row in episode['edges'].iterrows():
            src_id = row['src_id']
            dst_id = row['dst_id']
            
            if src_id in node_map and dst_id in node_map:
                edge_indices.append([node_map[src_id], node_map[dst_id]])
                
                # Map edge type to integer
                if row['etype'] == 'PROC_CHILD':
                    edge_types.append(0)
                elif row['etype'] == 'PROC_FILE_RW':
                    edge_types.append(1)
                elif row['etype'] == 'PROC_SOCK_CONN':
                    edge_types.append(2)
                else:
                    edge_types.append(3)  # Unknown
        
        # Create PyTorch Geometric Data object
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        x = torch.tensor(node_features, dtype=torch.float)
        node_type = torch.tensor(node_types, dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return Data(
            x=x,
            edge_index=edge_index,
            node_type=node_type,
            edge_type=edge_type,
            num_nodes=len(all_nodes)
        )
    
    def _prepare_ttp_labels(self, episode: Dict) -> int:
        """Prepare TTP labels for the episode"""
        # Map TTP sequence to a single label (simplified)
        ttp_sequence = episode['ttp_gt']['ttp_sequence']
        
        # Create a simple mapping to integer labels
        ttp_mapping = {
            'T1566.001': 0,  # Phishing
            'T1059.001': 1,  # PowerShell
            'T1218.011': 2,  # LOLBin
            'T1071.001': 3,  # C2
            'T1110.001': 4,  # Brute force
            'T1078.003': 5,  # Valid accounts
            'T1053.003': 6,  # Scheduled task
            'T1496': 7,      # Resource hijacking
            'T1486': 8,      # Data encrypted
            'T1489': 9       # Service stop
        }
        
        # Return the first TTP as the primary label
        if ttp_sequence:
            return ttp_mapping.get(ttp_sequence[0], 0)
        return 0
    
    def _prepare_tamper_labels(self, episode: Dict) -> torch.Tensor:
        """Prepare tamper detection labels"""
        # Check if tampering was applied
        tamper_applied = episode['metadata'].get('tamper_applied', False)
        
        # Create binary label
        return torch.tensor(1.0 if tamper_applied else 0.0, dtype=torch.float)


def create_dataloader(episodes_dir: str, tokenizer: RLSTIRTokenizer, 
                     batch_size: int = 8, max_seq_len: int = 1024,
                     num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """Create a DataLoader for RL-STIR dataset"""
    dataset = RLSTIRDataset(episodes_dir, tokenizer, max_seq_len)
    collator = RLSTIRCollator(tokenizer, max_seq_len)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
