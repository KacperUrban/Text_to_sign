from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import pandas as pd
from custom_utils.custom_pytorch_utils import cross_validation_pt
import optuna
import time
import numpy as np
import neptune

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()

        time_in_sec = round(stop - start, 3)

        minutes = time_in_sec // 60
        seconds = np.round(time_in_sec - (minutes * 60), 2)

        print(f"Function: {func.__name__}, took: {minutes} minutes and {seconds} seconds")
        return result
    return wrapper


def objective(trail):
    model = AutoModelForSeq2SeqLM.from_pretrained("model/model")
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/")

    for params in model.model.encoder.layers.parameters():
        params.requires_grad = False

    lr = trail.suggest_float("lr", low=5e-9, high=5e-4)
    optimizer = trail.suggest_categorical("optimizer", ["Adam", "SGD"])
    momentum = trail.suggest_float("momentum", low = 0.0, high=1.0)
    run["hyperparameters/learning_rate"] = lr
    if optimizer == "SGD":
        run["hyperparameters/momentum"] = momentum
    run["hyperparameters/optimizer"] = optimizer
    
    score = cross_validation_pt(model, tokenizer, data, device, optimizer=optimizer, momentum=momentum, num_epochs=5, n_splits=10, batch_size=32, lr=lr)
    run["score/BLEU"] = score
    return score

@timeit
def optimize_with_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=0)

    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)


if __name__ == '__main__':
    load_dotenv()
    run = neptune.init_run(tags="Experiments")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = pd.read_csv("data/final_data/augmented_data.csv")

    optimize_with_optuna()
    run.stop()

    