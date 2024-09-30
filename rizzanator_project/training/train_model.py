# train_model.py

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch

def main():
    # Load the dataset
    df = pd.read_csv('../datasets/custom_dataset.csv')
    dataset = Dataset.from_pandas(df)

    # Split the dataset into train and test sets
    dataset = dataset.train_test_split(test_size=0.1)

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

    # Tokenization function
    def tokenize_function(examples):
        inputs = ['context: ' + c for c in examples['context']]
        model_inputs = tokenizer(inputs, max_length=64, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['response'], max_length=64, truncation=True)

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        push_to_hub=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')

if __name__ == '__main__':
    main()
