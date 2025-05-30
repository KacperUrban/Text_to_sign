import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from config import (
    BASE_DIR,
    MODEL_DIR,
    TOKENIZER_DIR,
    DATA_DIR,
    FINAL_MODEL_DIR,
    TOKENIZER_DE_DIR,
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import pandas as pd
from custom_utils import (
    train,
    TranslationDataset,
    evaluate_model_on_bleu,
    collate_fn,
    collate_fn_de,
    train_amp,
)
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np
import random
import neptune

if __name__ == "__main__":
    load_dotenv(dotenv_path=BASE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bleu_metric = evaluate.load("bleu")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    hyperparams = {
        "learning_rate": 0.0000119347,
        "optimizer": "Adam",
        "momentum": 0.0,
        "beta1": 0.332396,
        "beta2": 0.416326,
        "epochs": 15,
        "batch_size": 32,
    }

    # data = pd.read_csv(DATA_DIR + "/final_data/all_data.csv")

    # train_data, test_data  = train_test_split(data, test_size=0.1)

    train_data = pd.read_csv(DATA_DIR + "/final_data/de/train_data.csv")
    dev_data = pd.read_csv(DATA_DIR + "/final_data/de/test_data.csv")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR + "_de")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DE_DIR)

    #     for params in model.model.encoder.layers.parameters():
    #                 params.requires_grad = False

    optimizer = Adam(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        betas=(hyperparams["beta1"], hyperparams["beta2"]),
    )

    train_data = train_data.dropna().reset_index(drop=True)
    test_data = dev_data.dropna().reset_index(drop=True)

    train_data = TranslationDataset(train_data.de, train_data.mig, tokenizer)
    test_data = TranslationDataset(test_data.de, test_data.mig, tokenizer)

    train_dataloader = DataLoader(
        train_data, batch_size=hyperparams["batch_size"], collate_fn=collate_fn_de
    )
    test_dataloader = DataLoader(
        test_data, batch_size=hyperparams["batch_size"], collate_fn=collate_fn_de
    )
    with neptune.init_run(tags=["best params", "de", "unfrozen", "test_test"]) as run:
        model.to(device)
        model = train_amp(model, optimizer, train_dataloader, device, hyperparams["epochs"])
        score = evaluate_model_on_bleu(
            model, test_dataloader, tokenizer, bleu_metric, device
        )
        print(f"BLEU score: {score}")

        run["hyperparameters"] = hyperparams
        run["score/BLEU"] = score

        model.save_pretrained(FINAL_MODEL_DIR + "_de", safe_serialization=True)
