import argparse
import pydantic

from datasets import load_dataset
from fsspec.registry import default
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

def main():
    pass

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


    args = parser.parse_args()
    options = TrainingOptions(**vars(args))
    if not options.use_default:
        exit(0)
    else:
        disc_config = ElectraConfig.from_pretrained("google/electra-small-discriminator")
        gen_config = ElectraConfig.from_pretrained("google/electra-small-generator")
        tokenizer_path = "google/electra-base-discriminator"
        dataset = load_dataset(options.dataset_path, streaming=True, split="train")
        trainer = ELECTRATrainer(gen_config, disc_config, tokenizer_path, dataset, options.run_name, options.batch_size)
        trainer.train(options.num_train_steps)
