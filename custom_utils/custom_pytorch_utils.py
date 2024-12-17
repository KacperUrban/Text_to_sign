import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


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


def evaluate_model_on_bleu(model, dataloader, tokenizer, bleu_metric, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
            references = [tokenizer.decode(g, skip_special_tokens=True) for g in batch["labels"]]

            all_preds.extend(predictions)
            all_labels.extend(references)

    # Tokenize predictions and references
    all_preds_tokenized = [pred for pred in all_preds]
    all_labels_tokenized = [[label] for label in all_labels]

    bleu_score = np.round(bleu_metric.compute(predictions=all_preds_tokenized, references=all_labels_tokenized)['bleu'], 3)
    return bleu_score