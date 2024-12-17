from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import neptune
from neptune.utils import stringify_unsupported
import evaluate
from custom_utils.custom_pytorch_utils import TranslationDataset, collate_fn, evaluate_model_on_bleu
from sklearn.model_selection import KFold

print(load_dotenv())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = AutoModelForSeq2SeqLM.from_pretrained("model/model")
tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/")

bleu_metric = evaluate.load("bleu")

data = pd.read_csv("data/final_data/all_data.csv")

initial_state_dict = model.state_dict()
kfold = KFold(n_splits=10)
avg_bleu = 0

for i, (train_idx, test_idx) in enumerate(kfold.split(data)):
    print(f"Fold {i + 1}")
    train_part = data.iloc[train_idx].reset_index(drop=True)
    test_part = data.iloc[test_idx].reset_index(drop=True)

    train_dataset = TranslationDataset(train_part.pl, train_part.mig, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    test_dataset = TranslationDataset(test_part.pl, test_part.mig, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    model.load_state_dict(initial_state_dict)
    model.to(device)
    lr = 5e-5
    num_epochs = 1
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

    bleu = evaluate_model_on_bleu(model, test_dataloader, tokenizer, bleu_metric, device)
    avg_bleu += bleu
    print(f"BLEU score: {bleu}")
    torch.cuda.empty_cache()
print(f"Avergae BLEU: {np.round(avg_bleu / (i + 1), 3)}")