from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import pandas as pd
from custom_utils.custom_pytorch_utils import cross_validation_pt
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
    if hyperparams["optimizer_name"] == "Adam":
        run["hyperparameters/beta1"] = hyperparams["Beta1"]
        run["hyperparameters/beta2"] = hyperparams["Beta2"]
    if hyperparams["optimizer_name"] == "SGD":
        run["hyperparameters/momentum"] = hyperparams["momentum"]


def objective(trail):
    with neptune.init_run(tags=["Test"]) as run:
        model = AutoModelForSeq2SeqLM.from_pretrained("model/model")
        tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/")

        # for params in model.model.encoder.layers.parameters():
        #   params.requires_grad = False

        hyperparams = defaultdict(int)
        hyperparams["lr"] = trail.suggest_float("lr", low=5e-9, high=5e-4)
        hyperparams["optimizer_name"] = trail.suggest_categorical(
            "optimizer", ["Adam", "SGD"]
        )
        hyperparams["Beta1"] = trail.suggest_float("Beta1", low=0.0, high=1.0)
        hyperparams["Beta2"] = trail.suggest_float("Beta2", low=0.0, high=1.0)

        if hyperparams["optimizer_name"] == "SGD":
            hyperparams["momentum"] = trail.suggest_float("momentum", low=0.0, high=1.0)

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
    study.optimize(objective, n_trials=2)

    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)


if __name__ == "__main__":
    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = pd.read_csv("data/final_data/augmented_data.csv")

    optimize_with_optuna()
