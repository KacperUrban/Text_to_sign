import pandas as pd
from datasets import Dataset, DatasetDict
import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from config import DATA_DIR
from custom_utils import (
    capitalize_sentence,
    load_data,
    split_data_from_list,
)

def preprocess_pl_data(PATH: str, PATH_FINAL: str) -> pd.DataFrame:
    """This function load data from csv file, then preprocess it (capitalize letters, remove duplicated etc.). After this
    function return Dataframe and saved it to particular file. It is only applies on our custom dataset in Polish language
    for translation from text to glosses.

    Args:
        PATH (str): path to csv file, where data are stored
        PATH_FINAL (str): path to target place, where data will be stored

    Returns:
        pd.DataFrame: preprocessed data for translation problem
    """
    print("Preprocessing data...")
    if PATH == "/urzad_dowody.odt":
        data = load_data(DATA_DIR + PATH)
        data = split_data_from_list(data)
    else:
        data = pd.read_csv(DATA_DIR + PATH)

    data["pl"] = data["pl"].apply(capitalize_sentence)
    data["mig"] = data["mig"].apply(capitalize_sentence)

    if sum(data.duplicated()) > 0:
        print(f"Number of duplicates: {sum(data.duplicated())}")
        data.drop_duplicates(inplace=True)
    print(f"Saving data to {DATA_DIR + PATH_FINAL}")
    data.to_csv(DATA_DIR + PATH_FINAL, index=False)
    return data

def preprocess_de_data(PATH: str, PATH_FINAL: str) -> None:
    """This function preprocess data for furthed training. It's apply to existed dataset PHOENIX-Weather.
    In this function mainly strucutre of file is changed not data itself. Only capitalization is applied.

    Args:
        PATH (str): path to csv file, where data are stored
        PATH_FINAL (str): path to target place, where data will be stored
    """
    print("Preprocessing data...")
    data = pd.read_csv(DATA_DIR + PATH, sep="|")

    data.drop(
        ["name", "video", "start", "end", "speaker"], inplace=True, axis=1
    )

    data = data.rename(columns={"orth": "mig", "translation": "de"})
    data.mig = data.mig.str.capitalize()
    data.de = data.de.str.capitalize()
    data = data[["de", "mig"]]

    print(f"Saving data to {DATA_DIR + PATH_FINAL}")
    data.to_csv(DATA_DIR + PATH_FINAL, index=False)


if __name__ == "__main__":
    # preprocess Polish data
    zus_data = preprocess_pl_data("/Praca_Socjalna_ZUS_Clean.txt", "/final_data/zus_data.csv")
    ratow_data = preprocess_pl_data("/Ratownictwo_Medyczne_Clean.txt", "/final_data/ratownictwo_data.csv")
    urzed_data = preprocess_pl_data("/Urzedy_Instytucje_Administracja_Clean.txt", "/final_data/urzedy_data.csv")
    policja_data = preprocess_pl_data("/Policja_Straz_Pozarna_Clean.txt", "/final_data/policja_data.csv")
    urzad_dowod_data = preprocess_pl_data("/urzad_dowody.odt", "/final_data/urzad_dowod.csv")
    augmented_data = preprocess_pl_data("/augmented_data_draft.csv", "/final_data/augmented_data.csv")

    all_data = pd.concat(
        [zus_data, ratow_data, urzed_data, policja_data, urzad_dowod_data],
        ignore_index=True,
    )

    if sum(all_data.duplicated()) > 0:
        all_data.drop_duplicates(inplace=True)

    all_data.to_csv(DATA_DIR + "/final_data/all_data.csv", index=False)

    raw_dataset_list = []
    for i in range(0, len(all_data)):
        raw_dataset_list.append(
            {
                "translation": {
                    "pl": all_data.iloc[i]["pl"],
                    "mig": all_data.iloc[i]["mig"],
                }
            }
        )

    raw_dataset = Dataset.from_list(raw_dataset_list)

    train_test = raw_dataset.train_test_split(test_size=0.2, seed=42)
    valid_test = train_test["test"].train_test_split(test_size=0.5, seed=42)
    train_test_dataset = DatasetDict(
        {
            "train": train_test["train"],
            "valid": valid_test["train"],
            "test": valid_test["test"],
        }
    )

    train_test_dataset.save_to_disk(DATA_DIR + "/final_data")

    # preprocess German data
    preprocess_de_data("/PHOENIX-2014-T.train.corpus.csv", "/final_data/de/train_data.csv")
    preprocess_de_data("/PHOENIX-2014-T.dev.corpus.csv", "/final_data/de/dev_data.csv")
    preprocess_de_data("/PHOENIX-2014-T.test.corpus.csv", "/final_data/de/test_data.csv")
