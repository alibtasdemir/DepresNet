import torch
import os
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
from tqdm import tqdm
from torch.utils.data import DataLoader

from torchsummary import summary
tqdm.pandas()

DATAPATH = "/content/drive/MyDrive/BBM/CMP711/Project/data/data.csv"

dataset = pd.read_csv(DATAPATH)

train_bert = dataset.sample(frac=0.75)
validation_bert = dataset.drop(train_bert.index)

print(f'BERT Train Size: {train_bert.shape}')
print(f'BERT Validation Size: {validation_bert.shape}')

train = Dataset.from_pandas(train_bert, preserve_index=False)
validation = Dataset.from_pandas(validation_bert, preserve_index=False)

print(train)
print(validation)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_train_dataset = train.map(tokenize_function, batched=True)
tokenized_test_dataset = validation.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    "finetuned_bert",
    evaluation_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    num_train_epochs=2
    )
   # default arguments for fine-tuning
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # overwriting MLM roberta-base for sequence binary classification
model.cuda()

def compute_metrics(eval_preds):   # compute accuracy and f1-score
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(   # specifying trainer class
    model,
    training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()  # starts fine-tuning
trainer.save_model("finetunedModel")