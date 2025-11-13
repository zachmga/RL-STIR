"""
Evaluation script for RL-STIR supervised model.
Evaluates TTP detection and tamper detection performance.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from modeling import create_model, RLSTIRTokenizer, create_dataloader, load_model_from_checkpoint
from modeling.training_utils import MetricsCalculator, compute_calibration_metrics


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


class SupervisedEvaluator:
    """Evaluator for supervised RL-STIR model"""
    
    def __init__(self, config: DictConfig, checkpoint_path: str):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        
        # Load model
        self.model = load_model_from_checkpoint(checkpoint_path, config)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = RLSTIRTokenizer()
        if config.get('tokenizer_path'):
            self.tokenizer.load_model(config.tokenizer_path)
        else:
            logger.warning("No tokenizer path provided. Using default tokenizer.")
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on dataset"""
        logger.info("Starting evaluation...")
        
        # Initialize metrics
        ttp_metrics = MetricsCalculator(num_classes=80)
        tamper_metrics = MetricsCalculator(num_classes=2)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainties = []
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch, return_loss=True, return_uncertainty=True)
                
                # Compute loss
                if 'total_loss' in outputs:
                    total_loss += outputs['total_loss'].item()
                    num_batches += 1
                
                # TTP evaluation
                if 'ttp_labels' in batch and 'ttp_logits' in outputs:
                    ttp_pred = outputs['ttp_predicted_ttps']
                    ttp_labels = batch['ttp_labels']
                    ttp_probs = outputs['ttp_probabilities']
                    
                    ttp_metrics.update(ttp_pred, ttp_labels, ttp_probs)
                    
                    all_predictions.extend(ttp_pred.cpu().numpy())
                    all_targets.extend(ttp_labels.cpu().numpy())
                    all_probabilities.extend(ttp_probs.cpu().numpy())
                
                # Tamper evaluation
                if 'tamper_labels' in batch and 'tamper_logits' in outputs:
                    tamper_pred = (outputs['tamper_probabilities'] > 0.5).float()
                    tamper_labels = batch['tamper_labels']
                    tamper_probs = outputs['tamper_probabilities']
                    
                    tamper_metrics.update(tamper_pred, tamper_labels, tamper_probs)
                
                # Uncertainty evaluation
                if 'uncertainty_uncertainty' in outputs:
                    uncertainties = outputs['uncertainty_uncertainty'].squeeze()
                    all_uncertainties.extend(uncertainties.cpu().numpy())
        
        # Compute metrics
        results = {
            'avg_loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'ttp_metrics': ttp_metrics.compute_metrics(),
            'tamper_metrics': tamper_metrics.compute_metrics()
        }
        
        # Calibration metrics
        if all_probabilities:
            all_probabilities = np.array(all_probabilities)
            all_targets = np.array(all_targets)
            
            # Convert to torch tensors for calibration computation
            probs_tensor = torch.tensor(all_probabilities)
            targets_tensor = torch.tensor(all_targets)
            
            calibration_metrics = compute_calibration_metrics(
                probs_tensor, targets_tensor
            )
            results['calibration_metrics'] = calibration_metrics
        
        # Uncertainty metrics
        if all_uncertainties:
            all_uncertainties = np.array(all_uncertainties)
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            
            uncertainty_metrics = compute_uncertainty_metrics(
                torch.tensor(all_predictions),
                torch.tensor(all_uncertainties),
                torch.tensor(all_targets)
            )
            results['uncertainty_metrics'] = uncertainty_metrics
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a nice format"""
        console.print("\n[bold blue]Evaluation Results[/bold blue]")
        
        # Overall metrics
        table = Table(title="Overall Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Average Loss", f"{results['avg_loss']:.4f}")
        table.add_row("TTP F1 (Macro)", f"{results['ttp_metrics']['f1_macro']:.4f}")
        table.add_row("TTP F1 (Micro)", f"{results['ttp_metrics']['f1_micro']:.4f}")
        table.add_row("TTP Accuracy", f"{results['ttp_metrics']['accuracy']:.4f}")
        table.add_row("Tamper F1", f"{results['tamper_metrics']['f1_macro']:.4f}")
        table.add_row("Tamper Accuracy", f"{results['tamper_metrics']['accuracy']:.4f}")
        
        console.print(table)
        
        # Calibration metrics
        if 'calibration_metrics' in results:
            cal_table = Table(title="Calibration Metrics")
            cal_table.add_column("Metric", style="cyan")
            cal_table.add_column("Value", style="magenta")
            
            cal_metrics = results['calibration_metrics']
            cal_table.add_row("Expected Calibration Error", f"{cal_metrics['ece']:.4f}")
            cal_table.add_row("Maximum Calibration Error", f"{cal_metrics['mce']:.4f}")
            
            console.print(cal_table)
        
        # Uncertainty metrics
        if 'uncertainty_metrics' in results:
            unc_table = Table(title="Uncertainty Metrics")
            unc_table.add_column("Metric", style="cyan")
            unc_table.add_column("Value", style="magenta")
            
            unc_metrics = results['uncertainty_metrics']
            unc_table.add_row("Uncertainty-Error Correlation", 
                            f"{unc_metrics['uncertainty_error_correlation']:.4f}")
            
            console.print(unc_table)
    
    def detailed_analysis(self, data_loader: DataLoader):
        """Perform detailed analysis of model performance"""
        logger.info("Performing detailed analysis...")
        
        all_ttp_predictions = []
        all_ttp_targets = []
        all_tamper_predictions = []
        all_tamper_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch, return_loss=False)
                
                if 'ttp_labels' in batch:
                    ttp_pred = outputs['ttp_predicted_ttps']
                    ttp_labels = batch['ttp_labels']
                    all_ttp_predictions.extend(ttp_pred.cpu().numpy())
                    all_ttp_targets.extend(ttp_labels.cpu().numpy())
                
                if 'tamper_labels' in batch:
                    tamper_pred = (outputs['tamper_probabilities'] > 0.5).float()
                    tamper_labels = batch['tamper_labels']
                    all_tamper_predictions.extend(tamper_pred.cpu().numpy())
                    all_tamper_targets.extend(tamper_labels.cpu().numpy())
        
        # TTP classification report
        if all_ttp_predictions:
            console.print("\n[bold blue]TTP Detection - Classification Report[/bold blue]")
            ttp_report = classification_report(
                all_ttp_targets, all_ttp_predictions, 
                target_names=[f"TTP_{i}" for i in range(80)],
                zero_division=0
            )
            console.print(ttp_report)
        
        # Tamper classification report
        if all_tamper_predictions:
            console.print("\n[bold blue]Tamper Detection - Classification Report[/bold blue]")
            tamper_report = classification_report(
                all_tamper_targets, all_tamper_predictions,
                target_names=["Normal", "Tampered"],
                zero_division=0
            )
            console.print(tamper_report)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(config: DictConfig) -> None:
    """Main evaluation function"""
    console.print(f"[bold blue]RL-STIR Supervised Evaluation[/bold blue]")
    
    # Get checkpoint path
    checkpoint_path = config.get('ckpt')
    if not checkpoint_path:
        console.print("[red]Error: No checkpoint path provided. Use ckpt=path/to/checkpoint.pt[/red]")
        return
    
    if not Path(checkpoint_path).exists():
        console.print(f"[red]Error: Checkpoint not found at {checkpoint_path}[/red]")
        return
    
    # Create evaluator
    evaluator = SupervisedEvaluator(config, checkpoint_path)
    
    # Create data loader
    data_loader = create_dataloader(
        episodes_dir=config.data.episodes_dir,
        tokenizer=evaluator.tokenizer,
        batch_size=config.eval.batch_size,
        max_seq_len=config.data.max_seq_len,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    # Run evaluation
    results = evaluator.evaluate(data_loader)
    
    # Print results
    evaluator.print_results(results)
    
    # Detailed analysis
    evaluator.detailed_analysis(data_loader)
    
    console.print("\n[bold green]Evaluation completed![/bold green]")


if __name__ == "__main__":
    main()
