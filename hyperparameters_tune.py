from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import pandas as pd
from custom_utils.custom_pytorch_utils import cross_validation_pt
import optuna
import time


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()

        print(f"Function: {func.__name__}, took: {round(stop - start, 3)} sec")
        return result
    return wrapper


def objective(trail):
    model = AutoModelForSeq2SeqLM.from_pretrained("model/model")
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/")
    lr = trail.suggest_float("lr", low=5e-9, high=5e-4)
    
    score = cross_validation_pt(model, tokenizer, data, device, num_epochs=5, n_splits=10, batch_size=8, lr=lr)
    return score

@timeit
def optimize_with_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)


if __name__ == '__main__':
    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = pd.read_csv("data/final_data/all_data.csv")

    optimize_with_optuna()

    