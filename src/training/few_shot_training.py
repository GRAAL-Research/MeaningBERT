import argparse
import logging

import wandb
from datasets import load_dataset
from poutyne import set_seeds
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from metrics.metrics import compute_metrics, eval_compute_metrics_identical, eval_compute_metrics_unrelated
from tools import (
    bool_parse,
)

log = logging.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(logging.ERROR)

num_epoch = 250

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=45,
    help="The seed to use for training.",
)

parser.add_argument(
    "--root",
    type=str,
    default=".",
    help="Root directory.",
)

parser.add_argument(
    "--data_augmentation",
    type=bool_parse,
    default=True,
    help="Either or not to do data augmentation.",
)

args = parser.parse_args()

seed = args.seed
root = args.root
data_augmentation = args.data_augmentation

set_seeds(seed=seed)

if data_augmentation:
    csmd_dataset = load_dataset("davebulaval/CSMD", "meaning_with_data_augmentation")
else:
    csmd_dataset = load_dataset("davebulaval/CSMD", "meaning")

holdout_identical_dataset = load_dataset("davebulaval/CSMD", "meaning_holdout_identical")
holdout_unrelated_dataset = load_dataset("davebulaval/CSMD", "meaning_holdout_unrelated")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["original"], example["simplification"], truncation=True, padding=True)


tokenized_csmd_dataset = csmd_dataset.map(tokenize_function, batched=True)
tokenize_holdout_identical_dataset = holdout_identical_dataset.map(tokenize_function, batched=True)
tokenize_holdout_unrelated_dataset = holdout_unrelated_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="meaning_bert_train",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=num_epoch,
    save_total_limit=num_epoch,
    save_strategy="epoch",
    load_best_model_at_end=True,  # By default, use the eval loss to retrieve the best model.
    seed=seed,
    metric_for_best_model="eval_loss",
)

# num_labels to 1 to create a regression head
# REF: https://discuss.huggingface.co/t/fine-tune-bert-and-camembert-for-regression-problem/332/17
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_csmd_dataset["train"],
    eval_dataset=tokenized_csmd_dataset["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("----------Training start----------")
trainer.train()

wandb.run.config.update({"data_augmentation": f"{str(data_augmentation)} {num_epoch} holdout fixed"})
wandb.log({"Best model checkpoint path": trainer.state.best_model_checkpoint})

print("----------Test Set Evaluation start----------")
trainer.evaluate(eval_dataset=tokenized_csmd_dataset["test"], metric_key_prefix="test")

# We change the computed metrics for the holdout splits
trainer.compute_metrics = eval_compute_metrics_identical
trainer.evaluate(
    eval_dataset=tokenize_holdout_identical_dataset["test"],
    metric_key_prefix="test/identical_sentences",
)

trainer.compute_metrics = eval_compute_metrics_unrelated
trainer.evaluate(
    eval_dataset=tokenize_holdout_unrelated_dataset["test"],
    metric_key_prefix="test/unrelated_sentences",
)

trainer.save_model(f"meaningbert_best_model")
