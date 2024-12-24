import pandas as pd
from datasets import Dataset, DatasetDict
from custom_utils.custom_preprocessing import (
    capitalize_sentence,
    load_data,
    split_data_from_list,
)


if __name__ == "__main__":
    # Load data
    zus_data = pd.read_csv("data/Praca_Socjalna_ZUS_Clean.txt")
    ratow_data = pd.read_csv("data/Ratownictwo_Medyczne_Clean.txt")
    urzed_data = pd.read_csv("data/Urzedy_Instytucje_Administracja_Clean.txt")
    policja_data = pd.read_csv("data/Policja_Straz_Pozarna_Clean.txt")
    raw_data = load_data("data/wypowiedzi.odt")
    wyp_data = split_data_from_list(raw_data)

    # Capitaliaze all sentences
    zus_data["pl"] = zus_data["pl"].apply(capitalize_sentence)
    zus_data["mig"] = zus_data["mig"].apply(capitalize_sentence)
    if sum(zus_data.duplicated()) > 0:
        zus_data.drop_duplicates(inplace=True)
    zus_data.to_csv("data/final_data/zus_data.csv", index=False)

    ratow_data["pl"] = ratow_data["pl"].apply(capitalize_sentence)
    ratow_data["mig"] = ratow_data["mig"].apply(capitalize_sentence)
    if sum(ratow_data.duplicated()) > 0:
        ratow_data.drop_duplicates(inplace=True)
    ratow_data.to_csv("data/final_data/ratownictwo_data.csv", index=False)

    urzed_data["pl"] = urzed_data["pl"].apply(capitalize_sentence)
    urzed_data["mig"] = urzed_data["mig"].apply(capitalize_sentence)
    if sum(urzed_data.duplicated()) > 0:
        urzed_data.drop_duplicates(inplace=True)
    urzed_data.to_csv("data/final_data/urzedy_data.csv", index=False)

    policja_data["pl"] = policja_data["pl"].apply(capitalize_sentence)
    policja_data["mig"] = policja_data["mig"].apply(capitalize_sentence)
    if sum(policja_data.duplicated()) > 0:
        policja_data.drop_duplicates(inplace=True)
    policja_data.to_csv("data/final_data/policja_data.csv", index=False)

    wyp_data["pl"] = wyp_data["pl"].apply(capitalize_sentence)
    wyp_data["mig"] = wyp_data["mig"].apply(capitalize_sentence)
    if sum(wyp_data.duplicated()) > 0:
        wyp_data.drop_duplicates(inplace=True)
    wyp_data.to_csv("data/final_data/wypowiedzi_data.csv", index=False)

    # Concatanate all data
    all_data = pd.concat(
        [zus_data, ratow_data, urzed_data, policja_data, wyp_data], ignore_index=True
    )

    # Check if duplicate rows exists
    if sum(all_data.duplicated()) > 0:
        all_data.drop_duplicates(inplace=True)

    all_data.to_csv("data/final_data/all_data.csv", index=False)

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
    train_test_dataset.save_to_disk("data/final_data")
