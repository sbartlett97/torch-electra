import argparse
import logging
from dataclasses import dataclass
from typing import Optional

import optuna
from datasets import load_dataset
from transformers import ElectraConfig
from electra import ELECTRATrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ElectraTrainingConfig:
    """Configuration for ELECTRA training, following the original paper's settings"""
    # Model configuration
    generator_size: str = "small"  # small, base, or large
    discriminator_size: str = "base"  # small, base, or large
    
    # Training configuration
    batch_size: int = 256
    max_seq_length: int = 512
    num_train_steps: int = 1000000
    learning_rate: float = 2e-4
    warmup_steps: int = 10000
    
    # Optimizer configuration
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.01
    
    # Other settings
    accumulation_steps: Optional[int] = None  # Will be calculated based on batch size
    dataset_path: str = "Skylion007/openwebtext"
    run_name: str = "electra_pretrain"

    def __post_init__(self):
        if self.accumulation_steps is None:
            # Calculate gradient accumulation steps to maintain effective batch size of 256
            self.accumulation_steps = max(1, 256 // self.batch_size)

    @property
    def generator_config_name(self) -> str:
        return f"google/electra-{self.generator_size}-generator"
    
    @property
    def discriminator_config_name(self) -> str:
        return f"google/electra-{self.discriminator_size}-discriminator"

def create_model_configs(config: ElectraTrainingConfig) -> tuple[ElectraConfig, ElectraConfig]:
    """Create generator and discriminator configs with proper size ratios"""
    disc_config = ElectraConfig.from_pretrained(config.discriminator_config_name)
    gen_config = ElectraConfig.from_pretrained(config.generator_config_name)
    
    # Ensure generator and discriminator have compatible hidden sizes
    if config.generator_size == "small" and config.discriminator_size == "base":
        # Base discriminator has hidden_size=768
        # Generator should be 1/3 of that
        gen_config.hidden_size = disc_config.hidden_size // 3  # 256
        gen_config.embedding_size = disc_config.hidden_size  # Keep same as discriminator
        gen_config.num_attention_heads = disc_config.num_attention_heads // 3
        gen_config.intermediate_size = disc_config.intermediate_size // 3
    
    return gen_config, disc_config
    
def train_electra(config: ElectraTrainingConfig, trial: Optional[optuna.Trial] = None) -> Optional[float]:
    """Run ELECTRA training with given configuration"""
    logger.info(f"Starting ELECTRA training with configuration:\n{config}")
    
    gen_config, disc_config = create_model_configs(config)
    dataset = load_dataset(config.dataset_path, streaming=True, split="train")
    
    if trial is not None:
        # Hyperparameter optimization settings
        config.learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        config.adam_beta1 = trial.suggest_float('adam_b1', 0.8, 0.99)
        config.adam_beta2 = trial.suggest_float('adam_b2', 0.9, 0.999)
        config.adam_epsilon = trial.suggest_loguniform('adam_e', 1e-8, 1e-6)
        config.weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        config.warmup_steps = trial.suggest_int('lr_warmup_steps', 0, 10000)

    trainer = ELECTRATrainer(
        gen_config=gen_config,
        disc_config=disc_config,
        tokenizer_path=config.discriminator_config_name,
        dataset=dataset,
        name=config.run_name,
        batch_size=config.batch_size,
        accumulation_steps=config.accumulation_steps,
        lr=config.learning_rate,
        adam_b1=config.adam_beta1,
        adam_b2=config.adam_beta2,
        adam_e=config.adam_epsilon,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        train_steps=config.num_train_steps
    )
    
    loss = trainer.train(config.num_train_steps)
    return loss

def main():
    parser = argparse.ArgumentParser(description="ELECTRA Pre-training Pipeline")
    
    # Main configuration groups
    parser.add_argument('--preset', type=str, choices=['small', 'base', 'large'], default='base',
                       help='Use preset configurations from the original paper')
    parser.add_argument('--run_name', type=str, default="electra_pretrain",
                       help="Name of the training run (for saving models)")
    parser.add_argument('--batch_size', type=int,
                       help="Per-device batch size (default: auto-calculated based on preset)")
    parser.add_argument('--dataset_path', type=str, default="Skylion007/openwebtext",
                       help="HuggingFace dataset path")
    parser.add_argument('--steps', type=int,
                       help="Number of training steps (default: based on preset)")
    parser.add_argument('--optuna', action="store_true",
                       help="Perform hyperparameter optimization with Optuna")

    args = parser.parse_args()

    # Set up configuration based on preset
    preset_configs = {
        'small': ElectraTrainingConfig(
            generator_size='small',
            discriminator_size='small',
            batch_size=128,
            num_train_steps=1000000
        ),
        'base': ElectraTrainingConfig(
            generator_size='small',
            discriminator_size='base',
            batch_size=256,
            num_train_steps=2000000
        ),
        'large': ElectraTrainingConfig(
            generator_size='base',
            discriminator_size='large',
            batch_size=2048,
            num_train_steps=4000000
        )
    }

    config = preset_configs[args.preset]
    
    # Override defaults with any provided arguments
    if args.run_name:
        config.run_name = args.run_name
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.steps:
        config.num_train_steps = args.steps

    if args.optuna:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: train_electra(config, trial), n_trials=100)
        logger.info(f"Best parameters found: {study.best_params}")
        logger.info(f"Best loss achieved: {study.best_value}")
    else:
        train_electra(config)

if __name__ == "__main__":
    main()