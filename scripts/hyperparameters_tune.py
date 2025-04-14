import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from config import DATA_DIR, MODEL_DIR, TOKENIZER_DIR, BASE_DIR
from custom_utils import (
    cross_validation_pt,
    TranslationDataset,
    collate_fn,
    train,
    evaluate_model_on_bleu,
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import evaluate
from dotenv import load_dotenv
import pandas as pd
import optuna
import time
from tqdm import tqdm
import numpy as np
import neptune
from collections import defaultdict


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()

        time_in_sec = round(stop - start, 3)

        minutes = time_in_sec // 60
        seconds = np.round(time_in_sec - (minutes * 60), 2)

        print(
            f"Function: {func.__name__}, took: {minutes} minutes and {seconds} seconds"
        )
        return result

    return wrapper


def objective(trail):
    with neptune.init_run(tags=["frozen"]) as run:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

        for params in model.model.encoder.layers.parameters():
            params.requires_grad = False

        hyperparams = defaultdict(float)
        hyperparams["batch_size"] = 64
        hyperparams["epcohs"] = 5
        hyperparams["learning_rate"] = trail.suggest_float(
            "lr", low=5e-8, high=5e-3, log=True
        )
        hyperparams["optimizer"] = trail.suggest_categorical(
            "optimizer", ["Adam", "SGD"]
        )

        if hyperparams["optimizer"] == "SGD":
            hyperparams["momentum"] = trail.suggest_float("momentum", low=0.0, high=1.0)
        else:
            hyperparams["beta1"] = trail.suggest_float("beta1", low=0.0, high=1.0)
            hyperparams["beta2"] = trail.suggest_float("beta2", low=0.0, high=1.0)

        run["hyperparameters"] = hyperparams
        score = cross_validation_pt(
            model,
            tokenizer,
            data,
            device,
            hyperparams,
            n_splits=10,
        )
        run["score/BLEU"] = score
    return score


def tune_number_of_epochs(data: pd.DataFrame, device: str) -> None:
    train_indices = data.sample(frac=0.85, random_state=42).index
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    train_data = data.iloc[train_indices].reset_index(drop=True)
    valid_data = data.drop(train_indices).reset_index(drop=True)

    train_dataset = TranslationDataset(train_data.pl, train_data.mig, tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )

    valid_dataset = TranslationDataset(valid_data.pl, valid_data.mig, tokenizer)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )
    bleu_metric = evaluate.load("bleu")

    for epoch in tqdm(range(15, 25)):
        with neptune.init_run(tags=["frozen", "epochs"]) as run:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
            model.to(device)
            optimizer = Adam(
                model.parameters(), lr=0.000301242, betas=(0.791376, 0.760868)
            )

            for params in model.model.encoder.layers.parameters():
                params.requires_grad = False

            model = train(model, optimizer, train_dataloader, device, epoch)
            score = evaluate_model_on_bleu(
                model, valid_dataloader, tokenizer, bleu_metric, device
            )

            run["hyperparameters/learning_rate"] = 0.000301242
            run["hyperparameters/optimizer"] = "Adam"
            run["hyperparameters/beta1"] = 0.791376
            run["hyperparameters/beta2"] = 0.760868
            run["hyperparameters/momentum"] = 0.0
            run["hyperparameters/epochs"] = epoch
            run["score/BLEU"] = score


@timeit
def optimize_with_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)


if __name__ == "__main__":
    load_dotenv(dotenv_path=BASE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = pd.read_csv(DATA_DIR + "/final_data/augmented_data.csv")

    tune_number_of_epochs(data, device)
    # optimize_with_optuna()
