import traceback

import torch
import csv

from typing import Any, Optional, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining, \
    DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

import pandas as pd
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class DataCollatorForELECTRA(DataCollatorForLanguageModeling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked language modeling: 85% MASK, 15% original.

        A slight modification on the standard MLM objective, masks 85% of the time, does nothing for the rest (15%)
        as opposed to the original - MASK=80%, Random token=10%, No change=10%
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.85)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # The rest of the time (20% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class ELECTRATrainer(object):
    progress_bar = None
    steps = 0

    def __init__(self, gen_config, disc_config, tokenizer_path, dataset, name):
        super(ELECTRATrainer, self).__init__()
        self.tokenizer = ElectraTokenizerFast.from_pretrained(tokenizer_path)
        self._name = name
        gen_config.vocab_size = len(self.tokenizer)
        disc_config.vocab_size = len(self.tokenizer)
        self.generator = ElectraForMaskedLM(gen_config)
        self.discriminator = ElectraForPreTraining(disc_config)

        self.discriminator.electra.embeddings = self.generator.electra.embeddings
        self.generator.generator_lm_head.weight = self.generator.electra.embeddings.word_embeddings.weight

        self.data_collator = DataCollatorForELECTRA(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        tokenized_dataset = dataset.map(self.tokenize, batched=True)
        tokenized_dataset.with_format("torch")
        self.dataloader = DataLoader(tokenized_dataset, batch_size=1, pin_memory=True)
        self.generator.to(device)
        self.discriminator.to(device)

        grouped_params = self.get_layerwise_lr([self.generator, self.discriminator], base_lr=2e-4, layer_decay=0.95)

        self.optim = torch.optim.AdamW(grouped_params, eps=1e-6, lr=2e-4)

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optim, num_warmup_steps=1000, num_training_steps=20000000
        )

        self.disc_loss_func = torch.nn.BCEWithLogitsLoss()
        self.gen_loss_func = torch.nn.CrossEntropyLoss()

        self.gumbel = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=self.generator.dtype),
                                                        torch.tensor(1., device=device, dtype=self.generator.dtype))

    @staticmethod
    def get_layerwise_lr(models, base_lr, layer_decay):
        """
        Assign different learning rates to each layer in the model.
        :param models: The list of models whose layers will have different learning rates.
        :param base_lr: The base learning rate.
        :param layer_decay: The decay factor to apply to deeper layers.
        :return: List of parameter groups with their corresponding learning rates.
        """
        optimizer_grouped_parameters = []
        for model_num, model in enumerate(models):
            num_layers = len(list(model.electra.encoder.layer))

            for layer_idx, (name, param) in enumerate(model.named_parameters()):
                if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
                    weight_decay = 0.0
                else:
                    weight_decay = 0.01

                if 'embeddings' in name and model_num == 1:
                    continue
                if 'embeddings' in name:
                    layer_lr = base_lr
                else:
                    depth = layer_idx / num_layers
                    layer_lr = base_lr * (layer_decay ** depth)

                optimizer_grouped_parameters.append({
                    "params": param,
                    "lr": layer_lr,
                    "weight_decay": weight_decay
                })

            return optimizer_grouped_parameters

    def generator_forward(self, masked_input_ids, attn_mask, labels):
        outputs = self.generator(input_ids=masked_input_ids.to(device), attention_mask=attn_mask,
                                 labels=labels.to(device))
        loss = outputs.loss

        predictions = outputs.logits
        return predictions, loss

    def discriminator_forward(self, input_ids, attn_mask, labels):
        outputs = self.discriminator(input_ids=input_ids, attention_mask=attn_mask,
                                     token_type_ids=labels.long())
        loss = self.disc_loss_func(outputs.logits, labels)
        return loss

    def epoch(self, max_steps):
        with open('training_log.csv', mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Loss"])
            running_loss = 0.0
            accumulation_loss = 0.0
            steps = 0
            accumulation_steps = 128
            scaler = torch.amp.GradScaler()
            for batch_idx, batch in enumerate(self.dataloader):
                with torch.amp.autocast("cuda"):
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask'].to(device)
                    masked_input_ids, labels = self.data_collator.torch_mask_tokens(
                        input_ids, special_tokens_mask=batch["special_tokens_mask"]
                    )

                    gen_predictions, gen_loss = self.generator_forward(masked_input_ids.to(device), attention_mask,
                                                                       labels.to(device))
                    masked_indices = labels != -100
                    replaced_input_ids, disc_labels = self.replace_masked_tokens_with_predictions(
                        masked_input_ids, gen_predictions, masked_indices
                    )

                    disc_loss = self.discriminator_forward(replaced_input_ids, attention_mask, disc_labels)
                    combined_loss = gen_loss * 1.0 + disc_loss * 50

                combined_loss += combined_loss / accumulation_steps

                scaler.scale(combined_loss).backward()
                running_loss += combined_loss.item()
                steps += 1

                if (batch_idx + 1) % accumulation_steps == 0:

                    scaler.step(self.optim)
                    scaler.update()
                    self.lr_scheduler.step()

                    self.optim.zero_grad()
                    if steps % 640 == 0:
                        tqdm.write(f"Step [{steps}/{max_steps}], Loss = {accumulation_loss/accumulation_steps}")
                    writer.writerow([steps, accumulation_loss/accumulation_steps])
                    self.progress_bar.set_description(f"Training Progress - Loss: {accumulation_loss/accumulation_steps}")
                    accumulation_loss = 0.0
                else:
                    accumulation_loss += combined_loss.item()
                self.progress_bar.update(1)

                if steps >= max_steps:
                    return steps

            return steps

    def train(self, steps):
        total_steps = 0

        self.progress_bar = tqdm(total=steps, desc="Training Progress", unit=" steps")

        while total_steps < steps:
            try:
                epoch_steps = self.epoch(steps)
                total_steps += epoch_steps
            except BaseExceptionGroup:
                self.discriminator.save_pretrained(f"checkpoint_{self.steps}_discriminator")
                self.generator.save_pretrained(f"checkpoint_{self.steps}_generator")
                self.progress_bar.close()
                print(traceback.format_exc())
                exit(1)
        self.progress_bar.close()
        file_path = 'training_log.csv'
        data = pd.read_csv(file_path)

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(data['Step'], data['Loss'], label='Loss', color='b', marker='+', markersize=1)

        # Customize the plot
        plt.title('Step vs Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.savefig("loss_curve.png", dpi=600)
        self.discriminator.save_pretrained(f"./models/{self._name}")
        self.generator.save_pretrained(f"./models/{self._name}")

    def tokenize(self, sample):
        return self.tokenizer(sample['text'], padding="max_length", truncation=True, max_length=512,
                              return_special_tokens_mask=True,
                              return_tensors="pt")

    def replace_masked_tokens_with_predictions(self, input_ids, g_predictions, masked_indices):
        """Replaces masked tokens with generator's most confident predictions and creates labels for discriminator.
        """
        replaced_input_ids = input_ids.clone().to(device)
        gumbel_noise = self.gumbel.sample(g_predictions.shape).to(device)
        gumbel_logits = g_predictions + gumbel_noise

        sampled_predictions = gumbel_logits.argmax(dim=-1)

        replaced_input_ids[masked_indices] = sampled_predictions[masked_indices]
        labels = torch.zeros_like(input_ids, dtype=torch.float).to(device)
        labels[masked_indices] = 1.0
        return replaced_input_ids, labels