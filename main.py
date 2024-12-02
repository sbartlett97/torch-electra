import argparse

import optuna
import pydantic

from datasets import load_dataset
from transformers import ElectraConfig
from electra import ELECTRATrainer

class TrainingOptions(pydantic.BaseModel):
    use_default: bool
    config_path: str
    dataset_path: str
    tokenizer_path: str
    run_name: str
    num_train_steps: int
    batch_size: int
    accumulation_steps: int
    optuna: bool

def main():
    pass

# TODO: Add configuration of hyperparemters for training
# TODO: Add support for experiment tracking (MLFlow, WeightsAndBiases etc..)
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run ELECTRA pre-training")

    parser.add_argument('--use_default', action='store_true',
                        help='Use the default electra configs/tokenizer')
    parser.add_argument('--config_path', type=str, default="",
                        help='Path to the directory containing custom ELECTRA configs')
    parser.add_argument('--dataset_path', type=str, default="Skylion007/openwebtext",
                        help="Path to the dataset (Can be a huggingface dataset)")
    parser.add_argument("--tokenizer_path", type=str, default="google/electra-base-discriminator",
                        help="If not using default options, the path to the tokenizer to use (can be a tokenizer stored on huggingface)")
    parser.add_argument('--run_name', type=str, required=True,
                        help="Name of the training run (for saving the models)")
    parser.add_argument("--num_train_steps", type=int, default=40000,
                        help="Number of train batches to run")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="size of mini-batches")
    parser.add_argument("--accumulation_steps", type=int, default=128,
                        help="Number of accumulation steps before performing gradient updates")
    parser.add_argument("--optuna", action="store_true", required=False, default=False,
                        help="Perform basic hparam search with optuna (For now this will search for optimum values for learning rate, batch_size, AdamW values, weight decay)")


    args = parser.parse_args()
    options = TrainingOptions(**vars(args))
    if not options.use_default:
        exit(0)
    else:
        def training_run(trial: optuna.Trial | None=None) -> float | None:
            disc_config = ElectraConfig.from_pretrained("google/electra-small-discriminator")
            gen_config = ElectraConfig.from_pretrained("google/electra-small-generator")
            tokenizer_path = "google/electra-small-discriminator"
            dataset = load_dataset(options.dataset_path, streaming=True, split="train")
            if trial is not None:
                lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)  # Learning rate
                adam_b1 = trial.suggest_float('adam_b1', 0.8, 0.99)
                adam_b2 = trial.suggest_float('adam_b2', 0.9, 0.999)
                adam_e = trial.suggest_loguniform('adam_e', 1e-8, 1e-6)
                weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
                lr_warmup_steps = trial.suggest_int('lr_warmup_steps', 0, 10000)
                trainer = ELECTRATrainer(gen_config, disc_config, tokenizer_path, dataset, options.run_name,
                                         options.batch_size, accumulation_steps=128//options.batch_size, lr=lr,
                                         adam_b1=adam_b1, adam_b2=adam_b2, adam_e=adam_e, weight_decay=weight_decay,
                                         warmup_steps=lr_warmup_steps)
                loss = trainer.train(options.num_train_steps)
                return loss
            else:
                trainer = ELECTRATrainer(gen_config, disc_config, tokenizer_path, dataset, options.run_name,
                                       options.batch_size, 128//options.batch_size)
                trainer.train(options.num_train_steps)
                return None


        if options.optuna:
            study = optuna.create_study(training_run)
            print(f"Best parameters found: {study.best_params}")
        else:
            training_run()