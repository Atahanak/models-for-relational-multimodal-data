import argparse

import pandas as pd
from datasets import Dataset
from icecream import ic
from peft import LoraConfig, get_peft_model
from peft import TaskType as peftTaskType
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)

import evaluate

def read_dataset(args):
    df = pd.read_csv(args.root)
    df = df[["reviewText", "overall", "summary"]]
    df = df.dropna()
    df['text'] = df['summary'] + " " + df['reviewText']
    df = df.rename(columns={"overall": "label"})
    df['label'] = df['label'].astype(float)

    dataset = Dataset.from_pandas(df[['text', 'label']])
    if args.nrows < len(dataset):
        dataset = dataset.select(range(args.nrows))

    return dataset

def finetune_llm(args):
    dataset = read_dataset(args)

    model_name = args.text_model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)
    train_dataset = tokenized_datasets['train']
    test_dataset = tokenized_datasets['test']


    lora_config = LoraConfig(
        task_type=peftTaskType.SEQ_CLS, 
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        inference_mode=False, 
        lora_dropout=args.lora_dropout
    )
    model = get_peft_model(model, lora_config)  
    print(model.state_dict().keys())  

    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        run_name=args.name
    )
    mse_metric = evaluate.load("mse")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze(1)
        mse = mse_metric.compute(predictions=predictions, references=labels)
        return {"mse": mse}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    print(results)

def main():
    args = parse_args()
    finetune_llm(args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--nrows", type=int, default=100)
    parser.add_argument("--name", type=str, default="fashion")  
    parser.add_argument("--text_model", type=str, default="sentence-transformers/all-distilroberta-v1")
    parser.add_argument("--root", type=str)
    parser.add_argument("--lora_alpha", type=int, default=1)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_")

    return parser.parse_args()

if __name__ == "__main__":
    main()