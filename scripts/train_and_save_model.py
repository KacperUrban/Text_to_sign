import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from config import BASE_DIR, MODEL_DIR, TOKENIZER_DIR, DATA_DIR, FINAL_MODEL_DIR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import pandas as pd
from custom_utils import train, TranslationDataset, evaluate_model_on_bleu, collate_fn
from sklearn.model_selection import train_test_split
import evaluate

if __name__ == '__main__':
    load_dotenv(dotenv_path=BASE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bleu_metric = evaluate.load("bleu")

    hyperparams = {
         "learning_rate" : 0.000301242,
         "optimizer" : "Adam",
         "momentum" : 0.0,
         "beta1" : 0.791376,
         "beta2" : 0.760868,
         "epochs" : 20,
         "batch_size" : 64,
    }

    data = pd.read_csv(DATA_DIR + "/final_data/all_data.csv")

    train_data, test_data  = train_test_split(data, test_size=0.1)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    for params in model.model.encoder.layers.parameters():
                params.requires_grad = False

    optimizer = Adam(model.parameters(), lr=hyperparams["learning_rate"], betas=(hyperparams["beta1"], hyperparams["beta2"]))

    train_data = train_data.dropna().reset_index(drop=True)
    test_data = test_data.dropna().reset_index(drop=True)

    train_data = TranslationDataset(train_data.pl, train_data.mig, tokenizer)
    test_data = TranslationDataset(test_data.pl, test_data.mig, tokenizer)

    train_dataloader = DataLoader(train_data, batch_size=hyperparams["batch_size"], collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=hyperparams["batch_size"], collate_fn=collate_fn)

    model.to(device)
    model = train(model, optimizer, train_dataloader, device, hyperparams["epochs"])
    score = evaluate_model_on_bleu(model, test_dataloader, tokenizer, bleu_metric, device)
    print(f"BLEU score: {score}")

    model.save_pretrained(FINAL_MODEL_DIR, safe_serialization=True)
