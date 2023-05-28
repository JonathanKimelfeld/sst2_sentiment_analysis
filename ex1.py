from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
import numpy as np
import torch
import wandb
import datasets
import sys

# wandb.login(key="79632defaeeb483a925b9a8dbce7414432eea228")

BERT = "bert-base-uncased"
ROBERTA = "roberta-base"
ELECTRA_BASE = "google/electra-base-generator"

# Parse command line arguments
num_seeds = int(sys.argv[1])
num_train_samples = int(sys.argv[2])
num_val_samples = int(sys.argv[3])
num_pred_samples = int(sys.argv[4])

# NUM_SEEDS = 3

# Load sst2 dataset and define the three models
dataset = datasets.load_dataset("sst2")
model_names = [BERT, ROBERTA, ELECTRA_BASE]

# seed num
# num_seeds = NUM_SEEDS

# A dict of accuracies per model
accuracies = {}

# fine-tuning each one of the three models
for model_name in model_names:
    # current model accuracies to be stored here
    model_accuracies = []

    # for each one of the seeds (set to 3)
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        tokenizer = None
        model = None

        if "bert" in model_name:
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        elif "roberta" in model_name:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
        elif "electra" in model_name:
            tokenizer = ElectraTokenizer.from_pretrained(model_name)
            model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=2)


        # Prepare the dataset
        def encode_batch(batch):
            # Tokenize the input texts and label encoding
            batch_encoding = tokenizer(
                batch["sentence"],
                padding="longest",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )

            # Dynamic padding
            if "input_ids" in batch_encoding:
                batch_encoding["input_ids"] = batch_encoding["input_ids"][:, :tokenizer.model_max_length]
            if "attention_mask" in batch_encoding:
                batch_encoding["attention_mask"] = batch_encoding["attention_mask"][:, :tokenizer.model_max_length]

            batch_encoding["labels"] = batch["label"]
            return batch_encoding


        train_dataset = dataset["train"].select(range(num_train_samples)) if num_train_samples != -1 else dataset[
            "train"].map(encode_batch, batched=True)
        eval_dataset = dataset["validation"].select(range(num_val_samples)) if num_val_samples != -1 else dataset[
            "validation"].map(encode_batch, batched=True)
        pred_dataset = dataset["test"].select(range(num_pred_samples)) if num_pred_samples != -1 else dataset["test"]

        # Define the data collator with padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./output",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
        )

        # Define the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Initialize Weights&Biases
        # wandb.init(project="sentiment-analysis", config=training_args)

        # Train the model
        trainer.train()

        # evaluate:
        eval_result = trainer.evaluate()

        # Retrieve the loss and predicted logits
        eval_loss = eval_result["eval_loss"]
        # Get model's logits
        model_predictions = trainer.predict(eval_dataset).predictions

        # Convert logits to predicted labels
        predicted_labels = np.argmax(model_predictions, axis=1)

        # Retrieve the ground truth labels
        labels = eval_dataset["labels"]

        # Calculate the accuracy
        accuracy = np.mean(predicted_labels == labels)

        model_accuracies.append(accuracy)

        # Finish the Weights&Biases run
        # wandb.finish()

    # Store the accuracies for the current model
    accuracies[model_name] = model_accuracies

# Compute mean and standard deviation of accuracies for each model
mean_accuracies = {}
std_accuracies = {}

for model_name, model_acc in accuracies.items():
    mean_accuracies[model_name] = np.mean(model_acc)
    std_accuracies[model_name] = np.std(model_acc)

# Print the results
for model_name in model_names:
    print(f"Model: {model_name}")
    print(f"Mean accuracy: {mean_accuracies[model_name]:.4f}")
    print(f"Standard deviation of accuracy: {std_accuracies[model_name]:.4f}")
