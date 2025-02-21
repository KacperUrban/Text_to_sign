import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from config import DATA_DIR, MODEL_DIR, TOKENIZER_DIR, BASE_DIR
from custom_utils import cross_validation_pt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import pandas as pd
import optuna
import time
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


def log_params(run, hyperparams):
    run["hyperparameters/learning_rate"] = hyperparams["lr"]
    run["hyperparameters/optimizer"] = hyperparams["optimizer_name"]
    run["hyperparameters/beta1"] = hyperparams["beta1"]
    run["hyperparameters/beta2"] = hyperparams["beta2"]
    run["hyperparameters/momentum"] = hyperparams["momentum"]


def objective(trail):
    with neptune.init_run(tags=["unfrozen"]) as run:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

        # for params in model.model.encoder.layers.parameters():
        #     params.requires_grad = False

        hyperparams = defaultdict(float)
        hyperparams["lr"] = trail.suggest_float("lr", low=5e-8, high=5e-3, log=True)
        hyperparams["optimizer_name"] = trail.suggest_categorical(
            "optimizer", ["Adam", "SGD"]
        )

        if hyperparams["optimizer_name"] == "SGD":
            hyperparams["momentum"] = trail.suggest_float("momentum", low=0.0, high=1.0)
        else:
            hyperparams["beta1"] = trail.suggest_float("beta1", low=0.0, high=1.0)
            hyperparams["beta2"] = trail.suggest_float("beta2", low=0.0, high=1.0)
        
        log_params(run, hyperparams)
        score = cross_validation_pt(
            model,
            tokenizer,
            data,
            device,
            hyperparams,
            num_epochs=5,
            n_splits=10,
            batch_size=64,
        )
        run["score/BLEU"] = score
    return score


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

    optimize_with_optuna()
