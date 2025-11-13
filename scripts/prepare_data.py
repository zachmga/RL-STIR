"""
Data preparation script for RL-STIR supervised training.
Generates synthetic episodes and prepares training data.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.sim import SyntheticEpisodeGenerator
from modeling import RLSTIRTokenizer
from data.schemas import LogRecord, LogChannel


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


class DataPreparator:
    """Prepare training data for RL-STIR"""
    
    def __init__(self, output_dir: str = "data/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator = SyntheticEpisodeGenerator(
            output_dir=str(self.output_dir / "episodes")
        )
        
        self.tokenizer = None
    
    def generate_episodes(self, num_episodes: int = 1000, 
                         scenarios: Optional[List[str]] = None) -> List[str]:
        """Generate synthetic episodes"""
        if scenarios is None:
            scenarios = list(self.generator.scenarios.keys())
        
        episode_paths = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Generating episodes...", total=num_episodes)
            
            for i in range(num_episodes):
                scenario = scenarios[i % len(scenarios)]
                episode_path = self.generator.generate_episode(
                    scenario, f"episode_{i:06d}"
                )
                episode_paths.append(episode_path)
                progress.update(task, advance=1)
        
        logger.info(f"Generated {num_episodes} episodes")
        return episode_paths
    
    def train_tokenizer(self, episodes_dir: str, vocab_size: int = 32000):
        """Train SentencePiece tokenizer on episode data"""
        logger.info("Training tokenizer...")
        
        # Collect all text data from episodes
        text_data = []
        episodes_path = Path(episodes_dir)
        
        for episode_dir in episodes_path.iterdir():
            if not episode_dir.is_dir():
                continue
            
            # Load logs
            logs_file = episode_dir / "logs.parquet"
            if logs_file.exists():
                logs_df = pd.read_parquet(logs_file)
                
                # Convert to text
                for _, row in logs_df.iterrows():
                    try:
                        # Clean the row data - replace NaN with None for optional fields
                        row_dict = row.to_dict()
                        for field in ['pid', 'ppid', 'tid', 'event_id']:
                            if pd.isna(row_dict.get(field)):
                                row_dict[field] = None
                        
                        log_record = LogRecord(**row_dict)
                        text = self._log_to_text(log_record)
                        text_data.append(text)
                    except Exception as e:
                        logger.warning(f"Error processing log: {e}")
                        continue
        
        logger.info(f"Collected {len(text_data)} log entries for tokenizer training")
        
        # Train tokenizer
        self.tokenizer = RLSTIRTokenizer(vocab_size=vocab_size)
        tokenizer_path = self.output_dir / "tokenizer.model"
        self.tokenizer.train_tokenizer(text_data, str(tokenizer_path))
        
        logger.info(f"Tokenizer trained and saved to {tokenizer_path}")
        return str(tokenizer_path)
    
    def _log_to_text(self, log_record: LogRecord) -> str:
        """Convert log record to text for tokenizer training"""
        tokens = []
        
        # Channel
        if log_record.channel:
            tokens.append(f"<{log_record.channel.value}>")
        
        # Event ID
        if log_record.event_id is not None:
            tokens.append(f"<event_{log_record.event_id}>")
        
        # Template ID
        tokens.append(f"<template_{log_record.template_id}>")
        
        # Executable
        if log_record.exe:
            exe_basename = Path(log_record.exe).name
            tokens.append(exe_basename)
        
        # Process info
        if log_record.pid is not None:
            tokens.append(f"<pid_{log_record.pid}>")
        
        if log_record.ppid is not None:
            tokens.append(f"<ppid_{log_record.ppid}>")
        
        # User
        if log_record.user:
            tokens.append(log_record.user)
        
        # Command line (truncated)
        if log_record.cmdline:
            cmdline_tokens = log_record.cmdline.split()[:10]
            tokens.extend(cmdline_tokens)
        
        # Additional fields
        try:
            import json
            fields = json.loads(log_record.fields_json)
            for key, value in fields.items():
                if isinstance(value, (str, int, float)):
                    tokens.append(str(value))
        except (json.JSONDecodeError, TypeError):
            pass
        
        return " ".join(tokens)
    
    def create_dataset_splits(self, episodes_dir: str, 
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1,
                            test_ratio: float = 0.1):
        """Create train/val/test splits"""
        episodes_path = Path(episodes_dir)
        episode_dirs = [d for d in episodes_path.iterdir() if d.is_dir()]
        
        # Shuffle episodes
        np.random.shuffle(episode_dirs)
        
        # Split episodes
        n_episodes = len(episode_dirs)
        n_train = int(n_episodes * train_ratio)
        n_val = int(n_episodes * val_ratio)
        
        train_episodes = episode_dirs[:n_train]
        val_episodes = episode_dirs[n_train:n_train + n_val]
        test_episodes = episode_dirs[n_train + n_val:]
        
        # Create split directories
        splits = {
            'train': train_episodes,
            'val': val_episodes,
            'test': test_episodes
        }
        
        for split_name, episodes in splits.items():
            split_dir = self.output_dir / f"{split_name}_episodes"
            split_dir.mkdir(exist_ok=True)
            
            for episode_dir in episodes:
                # Create symlink or copy
                target = split_dir / episode_dir.name
                if not target.exists():
                    if os.name == 'nt':  # Windows
                        import shutil
                        shutil.copytree(episode_dir, target)
                    else:  # Unix-like
                        target.symlink_to(episode_dir.absolute())
        
        logger.info(f"Created splits: Train={len(train_episodes)}, "
                   f"Val={len(val_episodes)}, Test={len(test_episodes)}")
        
        return splits
    
    def analyze_dataset(self, episodes_dir: str) -> Dict[str, Any]:
        """Analyze the generated dataset"""
        logger.info("Analyzing dataset...")
        
        episodes_path = Path(episodes_dir)
        episode_dirs = [d for d in episodes_path.iterdir() if d.is_dir()]
        
        stats = {
            'total_episodes': len(episode_dirs),
            'scenarios': {},
            'total_logs': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'ttp_distribution': {},
            'tamper_ratio': 0.0
        }
        
        tampered_count = 0
        
        for episode_dir in episode_dirs:
            # Load metadata
            metadata_file = episode_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                attack_type = metadata.get('attack_type', 'unknown')
                stats['scenarios'][attack_type] = stats['scenarios'].get(attack_type, 0) + 1
                
                stats['total_logs'] += metadata.get('num_logs', 0)
                stats['total_nodes'] += metadata.get('num_nodes', 0)
                stats['total_edges'] += metadata.get('num_edges', 0)
                
                if metadata.get('tamper_applied', False):
                    tampered_count += 1
            
            # Load TTP ground truth
            ttp_file = episode_dir / "gt_ttp_sequence.json"
            if ttp_file.exists():
                with open(ttp_file, 'r') as f:
                    ttp_gt = json.load(f)
                
                for ttp in ttp_gt.get('ttp_sequence', []):
                    stats['ttp_distribution'][ttp] = stats['ttp_distribution'].get(ttp, 0) + 1
        
        stats['tamper_ratio'] = tampered_count / len(episode_dirs) if episode_dirs else 0.0
        
        # Save analysis
        analysis_file = self.output_dir / "dataset_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset analysis saved to {analysis_file}")
        return stats
    
    def prepare_all(self, num_episodes: int = 1000, vocab_size: int = 32000):
        """Prepare complete training dataset"""
        logger.info("Starting data preparation...")
        
        # Generate episodes
        episode_paths = self.generate_episodes(num_episodes)
        
        # Train tokenizer
        tokenizer_path = self.train_tokenizer(
            str(self.output_dir / "episodes"), vocab_size
        )
        
        # Create dataset splits
        splits = self.create_dataset_splits(str(self.output_dir / "episodes"))
        
        # Analyze dataset
        stats = self.analyze_dataset(str(self.output_dir / "episodes"))
        
        # Save preparation summary
        summary = {
            'num_episodes': num_episodes,
            'vocab_size': vocab_size,
            'tokenizer_path': tokenizer_path,
            'splits': {k: len(v) for k, v in splits.items()},
            'stats': stats
        }
        
        summary_file = self.output_dir / "preparation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data preparation complete! Summary saved to {summary_file}")
        console.print(f"[bold green]Data preparation completed successfully![/bold green]")
        console.print(f"Output directory: {self.output_dir}")
        console.print(f"Episodes: {num_episodes}")
        console.print(f"Tokenizer: {tokenizer_path}")
        
        return summary


def main():
    """Main data preparation function"""
    parser = argparse.ArgumentParser(description="Prepare RL-STIR training data")
    parser.add_argument("--num-episodes", type=int, default=1000,
                       help="Number of episodes to generate")
    parser.add_argument("--vocab-size", type=int, default=32000,
                       help="Vocabulary size for tokenizer")
    parser.add_argument("--output-dir", type=str, default="data/training",
                       help="Output directory for training data")
    
    args = parser.parse_args()
    
    # Create data preparator
    preparator = DataPreparator(args.output_dir)
    
    # Prepare all data
    preparator.prepare_all(args.num_episodes, args.vocab_size)


if __name__ == "__main__":
    main()
