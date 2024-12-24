import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import evaluate
import os

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
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/")
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


def evaluate_model_on_bleu(model, dataloader, tokenizer, bleu_metric, device, fold_name=None):
    model.eval()
    all_inputs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
            references = [tokenizer.decode(g, skip_special_tokens=True) for g in batch["labels"]]
            inputs = [tokenizer.decode(g, skip_special_tokens=True) for g in batch["input_ids"]]

            all_preds.extend(predictions)
            all_labels.extend(references)
            all_inputs.extend(inputs)

    if fold_name:
        preds = pd.DataFrame({"inputs" : all_inputs, "predictions" : all_preds, "labels" : all_labels})
        filename = f"{fold_name}.csv"
        preds.to_csv(os.path.join('.', 'logs_cv', filename))
    bleu_score = np.round(bleu_metric.compute(predictions=all_preds, references=all_labels)['bleu'], 3)
    return bleu_score


def cross_validation_pt(model, tokenizer, data, device, num_epochs=5, n_splits=10, batch_size=16, trace=False):
    bleu_metric = evaluate.load("bleu")
    initial_state_dict = model.state_dict()
    kfold = KFold(n_splits=n_splits, shuffle=True)
    avg_bleu = 0
    train_data = TranslationDataset(data.pl, data.mig, tokenizer)

    if trace:
        for i, (train_idx, test_idx) in enumerate(kfold.split(train_data)):
            print(f"Fold {i + 1}")
            train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), collate_fn=collate_fn)
            test_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx), collate_fn=collate_fn)
            
            model.load_state_dict(initial_state_dict)
            model.to(device)
            lr = 5e-5
            optimizer = Adam(model.parameters(), lr=lr)
            model.train()

            with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
            record_shapes=True,
            with_stack=True
        ) as prof:
                for epoch in range(num_epochs):
                    for batch in tqdm(train_dataloader):
                        batch = {key: value.to(device) for key, value in batch.items()}

                        outputs = model(**batch)
                        loss = outputs.loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    prof.step()

            bleu_score = evaluate_model_on_bleu(model, test_dataloader, tokenizer, bleu_metric, device, f"Fold_{i + 1}")
            avg_bleu += bleu_score
            print(f"BLEU score: {bleu_score}")
            torch.cuda.empty_cache()
    else:
        for i, (train_idx, test_idx) in enumerate(kfold.split(train_data)):
            print(f"Fold {i + 1}")
            train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), collate_fn=collate_fn)
            test_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx), collate_fn=collate_fn)
            
            model.load_state_dict(initial_state_dict)
            model.to(device)
            lr = 5e-5
            optimizer = Adam(model.parameters(), lr=lr)
            model.train()
            for epoch in range(num_epochs):
                for batch in tqdm(train_dataloader):
                    batch = {key: value.to(device) for key, value in batch.items()}

                    outputs = model(**batch)
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            bleu_score = evaluate_model_on_bleu(model, test_dataloader, tokenizer, bleu_metric, device, f"Fold_{i + 1}")
            avg_bleu += bleu_score
            print(f"BLEU score: {bleu_score}")
            torch.cuda.empty_cache()
    
    return np.round(avg_bleu / (i + 1), 3)