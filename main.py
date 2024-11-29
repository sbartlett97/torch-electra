import argparse
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

def main():
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run ELECTRA pre-training")

    # Add arguments
    parser.add_argument('--use_default', action='store_true',
                        help='Use the default electra configs/tokenizer')
    parser.add_argument('--config_path', type=str, default="",
                        help='Path to the directory containing custom ELECTRA configs')
    parser.add_argument('--dataset_path', type=str, default="",
                        required=True, help="Path to the dataset (Can be a huggingface dataset)")
    parser.add_argument("--tokenizer_path", type=str, default="",
                        help="If not using default options, the path to the tokenizer to use (can be a tokenizer stored on huggingface)")
    parser.add_argument('--run_name', type=str, required=True,
                        help="Name of the training run (for saving the models)")

    # Parse the arguments
    args = parser.parse_args()
    options = TrainingOptions(**vars(args))
    if not options.use_default:
        exit(0)
    else:
        disc_config = ElectraConfig.from_pretrained("google/electra-small-discriminator")
        gen_config = ElectraConfig.from_pretrained("google/electra-small-generator")
        tokenizer_path = "google/electra-base-discriminator"
        dataset = load_dataset(options.dataset_path)
        trainer = ELECTRATrainer(gen_config, disc_config, tokenizer_path, dataset, options.run_name)

