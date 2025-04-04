import os
import sys

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from config import TOKENIZER_DIR, BASE_DIR
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam, SGD, Optimizer
import evaluate
from evaluate import EvaluationModule
import os
import gc

class TranslationDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.input_texts[idx], return_tensors="pt", padding=True, truncation=True)
        targets = self.tokenizer(self.target_texts[idx], return_tensors="pt", padding=True, truncation=True)
        return {**inputs, "labels": targets["input_ids"]}
    

def collate_fn(batch):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    input_ids = [item['input_ids'].squeeze() for item in batch]
    attention_mask = [item['attention_mask'].squeeze() for item in batch]
    labels = [item['labels'].squeeze() for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def evaluate_model_on_bleu(model: AutoModelForSeq2SeqLM, dataloader: DataLoader, tokenizer: AutoTokenizer, bleu_metric: EvaluationModule, 
                           device: str, fold_name: str=None) -> float:
    model.eval()
    all_inputs = []
    all_preds = []
    all_labels = []
    all_results = []

    with torch.no_grad():
        for batch in dataloader:
            batch_bleu = 0.0
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_length=100)
            predictions = [tokenizer.decode(g, skip_special_tokens=True) or "empty" for g in outputs]
            references = [tokenizer.decode(g, skip_special_tokens=True) or "empty" for g in batch["labels"]]
            inputs = [tokenizer.decode(g, skip_special_tokens=True) for g in batch["input_ids"]]

            all_preds.extend(predictions)
            all_labels.extend(references)
            all_inputs.extend(inputs)

            if predictions and references and all(predictions) and all(references):
                batch_bleu = bleu_metric.compute(predictions=predictions, references=references)['bleu']
            else:
                print("Warning: Empty predictions or references found. Skipping BLEU calculation.")
            all_results.append(batch_bleu)

    if fold_name:
        preds = pd.DataFrame({"inputs" : all_inputs, "predictions" : all_preds, "labels" : all_labels})
        filename = f"{fold_name}.csv"
        preds.to_csv(os.path.join(BASE_DIR, 'logs_cv', filename))
    final_bleu = np.mean(all_results)
    del outputs, predictions, references, inputs, batch
    gc.collect()
    torch.cuda.empty_cache()
    return np.round(final_bleu, 3)


def train(model: AutoModelForSeq2SeqLM, optimizer: Optimizer, train_dataloader: DataLoader, device: str, num_epochs: int) -> AutoModelForSeq2SeqLM:
    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def cross_validation_pt(model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, data: pd.DataFrame, device: str, hyperparams: dict, 
                        num_epochs: int=5, n_splits: int=10, batch_size: int=16) -> float:
    bleu_metric = evaluate.load("bleu")
    initial_state_dict = model.state_dict()
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    avg_bleu = 0
    train_data = TranslationDataset(data.pl, data.mig, tokenizer)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_data)):
        print(f"Fold {i + 1}")
        train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), collate_fn=collate_fn)
        test_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx), collate_fn=collate_fn)
        
        model.load_state_dict(initial_state_dict)
        model.to(device)            
        if hyperparams["optimizer"] == "Adam":
            optimizer = Adam(model.parameters(), lr=hyperparams["learning_rate"], betas=(hyperparams['beta1'], hyperparams['beta2']))
        elif hyperparams["optimizer"] == "SGD":
            optimizer = SGD(model.parameters(), lr=hyperparams["learning_rate"], momentum=hyperparams["momentum"])
        
        model = train(model, optimizer, train_dataloader, device, num_epochs)

        bleu_score = evaluate_model_on_bleu(model, test_dataloader, tokenizer, bleu_metric, device, f"Fold_{i + 1}")
        avg_bleu += bleu_score
        print(f"BLEU score: {bleu_score}")
        torch.cuda.empty_cache()
    gc.collect()
    
    return np.round(avg_bleu / (i + 1), 3)
