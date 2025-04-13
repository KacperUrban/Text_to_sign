import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from config import DATA_DIR, MODEL_DIR, TOKENIZER_DIR, BASE_DIR
from custom_utils import cross_validation_pt, TranslationDataset, collate_fn, train, evaluate_model_on_bleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dotenv import load_dotenv
import pandas as pd
import neptune
import evaluate
from tqdm import tqdm

def load_files(filepaths: list[str]) -> pd.DataFrame:
    final_df = pd.read_csv(DATA_DIR + "/final_data/" + filepaths[0])
    for i in range(1, len(filepaths)):
        tmp_df = pd.read_csv(DATA_DIR + "/final_data/" + filepaths[i], names=["pl", "mig"], header=None)
        final_df = pd.concat([final_df, tmp_df], axis=0)
    return final_df.reset_index(drop=True)


def train_and_test(best_params: dict, data: pd.DataFrame, device: str) -> None:
        with neptune.init_run(tags=["test", "best params", "frozen", "augmented"]) as run:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

            for params in model.model.encoder.layers.parameters():
                params.requires_grad = False
            
            score = cross_validation_pt(
                model,
                tokenizer,
                data,
                device,
                best_params,
                n_splits=10,
            )
            run["hyperparameters"] = best_params
            run["score/BLEU"] = score
            print(f"{score}")

def cross_valid_diff_files(filepaths: list[str], device: str, hyperparams: dict) -> None:
    final_score = 0
    bleu_metric = evaluate.load("bleu")
    with neptune.init_run(tags=["cvtest", "frozen", "augmented", "best params"]) as run:
        for i in tqdm(range(len(filepaths))):
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

            for params in model.model.encoder.layers.parameters():
                    params.requires_grad = False

            optimizer = optimizer = Adam(model.parameters(), lr=hyperparams["learning_rate"], betas=(hyperparams["beta1"], hyperparams["beta2"]))

            train_data = load_files(filepaths[:i] + filepaths[i + 1:])
            test_data = pd.read_csv(DATA_DIR + "/final_data/" + filepaths[i], names=["pl", "mig"], header=None)
            tag_name = filepaths[i].split('_')[0]

            train_data = train_data.dropna().reset_index(drop=True)
            test_data = test_data.dropna().reset_index(drop=True)

            train_data = TranslationDataset(train_data.pl, train_data.mig, tokenizer)
            test_data = TranslationDataset(test_data.pl, test_data.mig, tokenizer)

            train_dataloader = DataLoader(train_data, batch_size=hyperparams["batch_size"], collate_fn=collate_fn)
            test_dataloader = DataLoader(test_data, batch_size=hyperparams["batch_size"], collate_fn=collate_fn)

            model.to(device)
            model = train(model, optimizer, train_dataloader, device, hyperparams["epochs"])
            score = evaluate_model_on_bleu(model, test_dataloader, tokenizer, bleu_metric, device)

            run["hyperparameters"] = hyperparams
            run[f"score/{tag_name}/BLEU"] = score
            final_score += score 
        run["score/BLEU"] = round(final_score / len(filepaths), 4)
        

def test_on_specific_domain(filepaths: list[str], device: str, hyperparams: dict) -> None:
    for filepath in filepaths:
        tag = filepath.split('_')[0]
        with neptune.init_run(tags=["cvtest", "frozen", "augmented", "best params", tag]) as run:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

            for params in model.model.encoder.layers.parameters():
                params.requires_grad = False

            data = pd.read_csv(DATA_DIR + "/final_data/" + filepath, names=["pl", "mig"], header=None)
            data = data.dropna().reset_index(drop=True)

            score = cross_validation_pt(
                model,
                tokenizer,
                data,
                device,
                hyperparams,
                n_splits=10,
            )
            run["hyperparameters"] = hyperparams
            run["score/BLEU"] = score
            print(f"{score}")

          

if __name__ == "__main__":
    load_dotenv(dotenv_path=BASE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = pd.read_csv(DATA_DIR + "/final_data/augmented_data.csv")
    filepath_list = ["policja_data_augmented.csv", "ratownictwo_data_augmented.csv", "urzad_dowod_augmented.csv", "urzedy_data_augmented.csv", 
                     "zus_data_augmented.csv"]
    best_params = {
         "learning_rate" : 0.000301242,
         "optimizer" : "Adam",
         "momentum" : 0.0,
         "beta1" : 0.791376,
         "beta2" : 0.760868,
         "epochs" : 20,
         "batch_size" : 64,
    }
    
    test_on_specific_domain(filepath_list, device, best_params)
    # train_and_test(best_params, data, device)
