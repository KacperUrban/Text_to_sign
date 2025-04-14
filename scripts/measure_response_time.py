import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from config import DATA_DIR, MODEL_DIR, TOKENIZER_DIR, BASE_DIR
from custom_utils import TranslationDataset, collate_fn, train, evaluate_model_on_bleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dotenv import load_dotenv
import pandas as pd
import time
import numpy as np


def measure_response_time(
    data: pd.DataFrame,
    device: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    hyperparams: dict,
):
    train_data = data
    test_data = data.copy()

    train_data = TranslationDataset(train_data.pl, train_data.mig, tokenizer)
    test_data = TranslationDataset(test_data.pl, test_data.mig, tokenizer)

    train_dataloader = DataLoader(
        train_data, batch_size=hyperparams["batch_size"], collate_fn=collate_fn
    )
    test_dataloader = DataLoader(test_data, batch_size=1, collate_fn=collate_fn)

    for params in model.model.encoder.layers.parameters():
        params.requires_grad = False

    model.to(device)
    optimizer = optimizer = Adam(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        betas=(hyperparams["beta1"], hyperparams["beta2"]),
    )
    model = train(model, optimizer, train_dataloader, device, hyperparams["epochs"])

    times_list = []
    length_sentence = []
    for batch in test_dataloader:
        start_time = time.time()
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=100,
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        end_time = time.time()
        length_sentence.append(len(outputs[0]))
        times_list.append(end_time - start_time)
    return (
        np.mean(times_list),
        np.std(times_list),
        np.mean(length_sentence),
        np.std(length_sentence),
    )


if __name__ == "__main__":
    load_dotenv(dotenv_path=BASE_DIR)
    data = pd.read_csv(DATA_DIR + "/final_data/augmented_data.csv")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    best_params = {
        "learning_rate": 0.000301242,
        "optimizer": "Adam",
        "momentum": 0.0,
        "beta1": 0.791376,
        "beta2": 0.760868,
        "epochs": 20,
        "batch_size": 64,
    }
    print(measure_response_time(data, "cuda", model, tokenizer, best_params))
