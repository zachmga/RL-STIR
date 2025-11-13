"""
Supervised pretraining script for RL-STIR.
Trains the model on labeled TTP and tamper detection data.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from modeling import create_model, RLSTIRTokenizer, create_dataloader
from envs.sim import SyntheticEpisodeGenerator


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


class SupervisedTrainer:
    """Supervised training pipeline for RL-STIR"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'ttp_f1': [],
            'tamper_f1': [],
            'learning_rate': []
        }
    
    def setup_logging(self):
        """Setup logging and experiment tracking"""
        # Create output directory
        self.output_dir = Path(self.config.get('output_dir', 'outputs/supervised'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup wandb if enabled
        if self.config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=self.config.wandb.project,
                name=self.config.wandb.name or f"supervised_{int(time.time())}",
                config=OmegaConf.to_container(self.config, resolve=True)
            )
    
    def setup_model(self):
        """Initialize model, tokenizer, and optimizer"""
        logger.info("Setting up model...")
        
        # Create model
        self.model = create_model(self.config)
        self.model.to(self.device)
        
        # Create tokenizer
        self.tokenizer = RLSTIRTokenizer(
            vocab_size=self.config.model.vocab_size
        )
        
        # Train tokenizer if needed
        if not self.config.get('tokenizer_path'):
            logger.info("Training tokenizer...")
            self.train_tokenizer()
        else:
            self.tokenizer.load_model(self.config.tokenizer_path)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=self.config.optim.betas,
            eps=self.config.optim.eps
        )
        
        # Setup scheduler
        # Setup learning rate scheduler - will be configured after data loaders are created
        self.scheduler = None
        
        # Setup mixed precision
        if self.config.train.fp16:
            self.scaler = GradScaler()
        
        logger.info(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_tokenizer(self):
        """Train SentencePiece tokenizer on synthetic data"""
        logger.info("Generating training data for tokenizer...")
        
        # Generate synthetic episodes for tokenizer training
        generator = SyntheticEpisodeGenerator()
        text_data = []
        
        # Generate episodes and extract text
        for scenario_name in generator.scenarios.keys():
            episode_path = generator.generate_episode(scenario_name)
            # Load and extract text from episode
            # This is simplified - in practice you'd load the actual episodes
            text_data.extend([
                f"<sysmon> <event_1> <template_10> powershell.exe <pid_5001> victim_user -EncodedCommand",
                f"<auditd> <event_14> <template_20> sshd root authentication failure",
                f"<evtx> <event_11> <template_40> encrypt.exe admin --encrypt --path=C:\\Users"
            ])
        
        # Train tokenizer
        tokenizer_path = self.output_dir / "tokenizer.model"
        self.tokenizer.train_tokenizer(text_data, str(tokenizer_path))
        
        logger.info(f"Tokenizer trained and saved to {tokenizer_path}")
    
    def setup_data(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")
        
        # Check if episodes already exist for quick testing
        episodes_dir = self.output_dir / "episodes"
        if episodes_dir.exists() and len(list(episodes_dir.glob("train_*"))) >= self.config.data.num_train_episodes:
            logger.info("Using existing episodes for faster testing...")
        else:
            # Generate training episodes
            generator = SyntheticEpisodeGenerator(
                output_dir=str(episodes_dir)
            )
            
            # Generate training episodes
            train_episodes = []
            for i in range(self.config.data.num_train_episodes):
                scenario = list(generator.scenarios.keys())[i % len(generator.scenarios)]
                episode_path = generator.generate_episode(scenario, f"train_{i}")
                train_episodes.append(episode_path)
            
            # Generate validation episodes
            val_episodes = []
            for i in range(self.config.data.num_val_episodes):
                scenario = list(generator.scenarios.keys())[i % len(generator.scenarios)]
                episode_path = generator.generate_episode(scenario, f"val_{i}")
                val_episodes.append(episode_path)
        
        # Create data loaders
        self.train_loader = create_dataloader(
            episodes_dir=str(self.output_dir / "episodes"),
            tokenizer=self.tokenizer,
            batch_size=self.config.train.batch_size,
            max_seq_len=self.config.data.max_seq_len,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        self.val_loader = create_dataloader(
            episodes_dir=str(self.output_dir / "episodes"),
            tokenizer=self.tokenizer,
            batch_size=self.config.train.batch_size,
            max_seq_len=self.config.data.max_seq_len,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        logger.info(f"Data loaders created. Train: {len(self.train_loader)}, Val: {len(self.val_loader)}")
        
        # Setup learning rate scheduler with actual number of steps
        total_steps = self.config.train.max_epochs * len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.optim.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        ttp_correct = 0
        ttp_total = 0
        tamper_correct = 0
        tamper_total = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Training...", total=len(self.train_loader))
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Handle graph batch object separately
                if 'graph' in batch:
                    batch['graph'] = batch['graph'].to(self.device)
                
                # Forward pass
                if self.config.train.fp16:
                    with autocast():
                        outputs = self.model(batch, return_loss=True)
                        loss = outputs['total_loss']
                else:
                    outputs = self.model(batch, return_loss=True)
                    loss = outputs['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.config.train.fp16:
                    self.scaler.scale(loss).backward()
                    if self.config.train.max_grad_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.train.max_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config.train.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.train.max_grad_norm
                        )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.global_step += 1
                
                # Update metrics
                total_loss += loss.item()
                
                # Compute accuracy
                if 'ttp_labels' in batch:
                    ttp_pred = outputs['ttp_predicted_ttps']
                    ttp_labels = batch['ttp_labels']
                    ttp_correct += (ttp_pred == ttp_labels).sum().item()
                    ttp_total += ttp_labels.size(0)
                
                if 'tamper_labels' in batch:
                    tamper_pred = (outputs['tamper_probabilities'] > 0.5).float()
                    tamper_labels = batch['tamper_labels']
                    tamper_correct += (tamper_pred.squeeze() == tamper_labels).sum().item()
                    tamper_total += tamper_labels.size(0)
                
                # Log progress
                if batch_idx % self.config.train.log_interval == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                        f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
                    )
                    
                    if self.config.get('wandb', {}).get('enabled', False):
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/learning_rate': current_lr,
                            'train/step': self.global_step
                        })
                
                progress.update(task, advance=1)
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        ttp_accuracy = ttp_correct / ttp_total if ttp_total > 0 else 0.0
        tamper_accuracy = tamper_correct / tamper_total if tamper_total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'ttp_accuracy': ttp_accuracy,
            'tamper_accuracy': tamper_accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        ttp_correct = 0
        ttp_total = 0
        tamper_correct = 0
        tamper_total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Handle graph batch object separately
                if 'graph' in batch:
                    batch['graph'] = batch['graph'].to(self.device)
                
                # Forward pass
                if self.config.train.fp16:
                    with autocast():
                        outputs = self.model(batch, return_loss=True)
                        loss = outputs['total_loss']
                else:
                    outputs = self.model(batch, return_loss=True)
                    loss = outputs['total_loss']
                
                total_loss += loss.item()
                
                # Compute accuracy
                if 'ttp_labels' in batch:
                    ttp_pred = outputs['ttp_predicted_ttps']
                    ttp_labels = batch['ttp_labels']
                    ttp_correct += (ttp_pred == ttp_labels).sum().item()
                    ttp_total += ttp_labels.size(0)
                
                if 'tamper_labels' in batch:
                    tamper_pred = (outputs['tamper_probabilities'] > 0.5).float()
                    tamper_labels = batch['tamper_labels']
                    tamper_correct += (tamper_pred.squeeze() == tamper_labels).sum().item()
                    tamper_total += tamper_labels.size(0)
        
        # Compute validation metrics
        avg_loss = total_loss / len(self.val_loader)
        ttp_accuracy = ttp_correct / ttp_total if ttp_total > 0 else 0.0
        tamper_accuracy = tamper_correct / tamper_total if tamper_total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'ttp_accuracy': ttp_accuracy,
            'tamper_accuracy': tamper_accuracy
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved to {best_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting supervised training...")
        
        # Setup components
        self.setup_model()
        self.setup_data()
        
        # Training loop
        for epoch in range(self.config.train.max_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"TTP Acc: {val_metrics['ttp_accuracy']:.4f}, "
                f"Tamper Acc: {val_metrics['tamper_accuracy']:.4f}"
            )
            
            # Update metrics
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['ttp_f1'].append(val_metrics['ttp_accuracy'])
            self.metrics['tamper_f1'].append(val_metrics['tamper_accuracy'])
            self.metrics['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # Log to wandb
            if self.config.get('wandb', {}).get('enabled', False):
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'val/loss': val_metrics['loss'],
                    'val/ttp_accuracy': val_metrics['ttp_accuracy'],
                    'val/tamper_accuracy': val_metrics['tamper_accuracy'],
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['loss']
            
            if epoch % self.config.train.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)
        
        logger.info("Training completed!")
        
        # Save final model
        final_path = self.output_dir / "final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }, final_path)
        
        logger.info(f"Final model saved to {final_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="supervised_4060")
def main(config: DictConfig) -> None:
    """Main training function"""
    console.print(f"[bold blue]RL-STIR Supervised Training[/bold blue]")
    console.print(f"Config: {config}")
    
    # Create trainer
    trainer = SupervisedTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
