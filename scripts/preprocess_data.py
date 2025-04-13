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



if __name__ == "__main__":

    # Load data
    zus_data = pd.read_csv(DATA_DIR + "/Praca_Socjalna_ZUS_Clean.txt")
    ratow_data = pd.read_csv(DATA_DIR + "/Ratownictwo_Medyczne_Clean.txt")
    urzed_data = pd.read_csv(DATA_DIR + "/Urzedy_Instytucje_Administracja_Clean.txt")
    policja_data = pd.read_csv(DATA_DIR + "/Policja_Straz_Pozarna_Clean.txt")
    raw_data = load_data(DATA_DIR + "/urzad_dowody.odt")
    urzad_dowod_data = split_data_from_list(raw_data)
    augmented_data = pd.read_csv(DATA_DIR + "/augmented_data_draft.csv")
    # Capitaliaze all sentences
    augmented_data["pl"] = augmented_data["pl"].apply(capitalize_sentence)
    augmented_data["mig"] = augmented_data["mig"].apply(capitalize_sentence)

    if sum(augmented_data.duplicated()) > 0:
        print(f"Number of duplicates: {sum(augmented_data.duplicated())}")
        augmented_data.drop_duplicates(inplace=True)
    augmented_data.to_csv(DATA_DIR + "/final_data/augmented_data.csv", index=False)

    zus_data["pl"] = zus_data["pl"].apply(capitalize_sentence)
    zus_data["mig"] = zus_data["mig"].apply(capitalize_sentence)
    if sum(zus_data.duplicated()) > 0:
        zus_data.drop_duplicates(inplace=True)
    zus_data.to_csv(DATA_DIR + "/final_data/zus_data.csv", index=False)

    ratow_data["pl"] = ratow_data["pl"].apply(capitalize_sentence)
    ratow_data["mig"] = ratow_data["mig"].apply(capitalize_sentence)
    if sum(ratow_data.duplicated()) > 0:
        ratow_data.drop_duplicates(inplace=True)
    ratow_data.to_csv(DATA_DIR + "/final_data/ratownictwo_data.csv", index=False)

    urzed_data["pl"] = urzed_data["pl"].apply(capitalize_sentence)
    urzed_data["mig"] = urzed_data["mig"].apply(capitalize_sentence)
    if sum(urzed_data.duplicated()) > 0:
        urzed_data.drop_duplicates(inplace=True)
    urzed_data.to_csv(DATA_DIR + "/final_data/urzedy_data.csv", index=False)

    policja_data["pl"] = policja_data["pl"].apply(capitalize_sentence)
    policja_data["mig"] = policja_data["mig"].apply(capitalize_sentence)
    if sum(policja_data.duplicated()) > 0:
        policja_data.drop_duplicates(inplace=True)
    policja_data.to_csv(DATA_DIR + "/final_data/policja_data.csv", index=False)

    urzad_dowod_data["pl"] = urzad_dowod_data["pl"].apply(capitalize_sentence)
    urzad_dowod_data["mig"] = urzad_dowod_data["mig"].apply(capitalize_sentence)
    if sum(urzad_dowod_data.duplicated()) > 0:
        urzad_dowod_data.drop_duplicates(inplace=True)
    urzad_dowod_data.to_csv(DATA_DIR + "/final_data/urzad_dowod.csv", index=False)

    # Concatanate all data
    all_data = pd.concat(
        [zus_data, ratow_data, urzed_data, policja_data, urzad_dowod_data], ignore_index=True
    )

    # Check if duplicate rows exists
    if sum(all_data.duplicated()) > 0:
        all_data.drop_duplicates(inplace=True)

    all_data.to_csv(DATA_DIR + "/final_data/all_data.csv", index=False)

    # Create a dictionary for further preprocessing
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

    # From dictionary create Dataset
    raw_dataset = Dataset.from_list(raw_dataset_list)

    # Split data into three datasets
    train_test = raw_dataset.train_test_split(test_size=0.2, seed=42)
    valid_test = train_test["test"].train_test_split(test_size=0.5, seed=42)
    train_test_dataset = DatasetDict(
        {
            "train": train_test["train"],
            "valid": valid_test["train"],
            "test": valid_test["test"],
        }
    )

    # Save data
    train_test_dataset.save_to_disk(DATA_DIR + "/final_data")

    # preprocess German data
    data_de_train = pd.read_csv(DATA_DIR + "/PHOENIX-2014-T.train.corpus.csv", sep="|")
    data_de_dev = pd.read_csv(DATA_DIR + "/PHOENIX-2014-T.dev.corpus.csv", sep="|")
    data_de_test = pd.read_csv(DATA_DIR + "/PHOENIX-2014-T.test.corpus.csv", sep="|")

    data_de_train.drop(["name", "video", "start", "end", "speaker"], inplace=True, axis=1)
    data_de_dev.drop(["name", "video", "start", "end", "speaker"], inplace=True, axis=1)
    data_de_test.drop(["name", "video", "start", "end", "speaker"], inplace=True, axis=1)

    data_de_train = data_de_train.rename(columns={"orth" : "mig", "translation" : "de"})
    data_de_dev = data_de_dev.rename(columns={"orth" : "mig", "translation" : "de"})
    data_de_test = data_de_test.rename(columns={"orth" : "mig", "translation" : "de"})

    data_de_train.to_csv(DATA_DIR + "/final_data/de/train_data.csv")
    data_de_dev.to_csv(DATA_DIR + "/final_data/de/dev_data.csv")
    data_de_test.to_csv(DATA_DIR + "/final_data/de/test_data.csv")
