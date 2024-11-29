import argparse
import pydantic

from datasets import load_dataset
from transformers import ElectraConfig


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
    parser.add_argument('--config_path', type=str,
                        help='Path to the directory containing custom ELECTRA configs')
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset (Can be a huggingface dataset)")
    parser.add_argument("--tokenizer_path", type=str,
                        help="If not using default options, the path to the tokenizer to use (can be a tokenizer stored on huggingface)")
    parser.add_argument('--run_name', type=str, required=True,
                        help="Name of the training run (for saving the models)")

    # Parse the arguments
    args = parser.parse_args()
    main()