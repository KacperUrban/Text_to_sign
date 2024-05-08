import pandas as pd
from odf.opendocument import load
from odf import text, teletype
from datasets import Dataset, DatasetDict


def capitalize_sentence(sentence: str) -> str:
    """This function captialize first word in sentence and lowercase rest of the sentence. In
    this function was added some conditions to lowercasing and captializing.

    Args:
        sentence (str): sentence in polish or sign language

    Returns:
        str: preprocessed sentence
    """
    cap_words = ["pani", "pana", "panu", "pan", "panią"]
    up_words = ["I", "II", "III"]
    words = sentence.split()
    if words[0] not in up_words:
        words[0] = words[0].capitalize()
    words[1:] = [
        word.capitalize()
        if word.lower() in cap_words
        else word.upper()
        if word in up_words
        else word.lower()
        for word in words[1:]
    ]
    return " ".join(words)


def load_data(filepath: str) -> list[str]:
    """The function has to load data from odf format and return list of file lines.

    Args:
        filepath (str): path to the file

    Returns:
        list[str]: list of the file lines
    """
    raw_data = []
    text_doc = load(filepath)
    all_params = text_doc.getElementsByType(text.P)
    for line in all_params:
        raw_data.append(teletype.extractText(line))
    return raw_data


def split_data_from_list(raw_data: list[str]) -> pd.DataFrame:
    """That function process data from list and splits data into pair of examples. In the left column is sentence
    in polish language and in the right column is sentence in polish sign language.

    Args:
        raw_data (list[str]): list of the sentence in polish and sign language

    Returns:
        pd.DataFrame: DataFrame with sentence in polish language and in sign language
    """
    pl_sentence = []
    sentence = []
    i = 0
    while i < len(raw_data):
        if raw_data[i + 1] == "[":
            value = raw_data[i]
            i += 2
            while i < len(raw_data):
                if raw_data[i] == "]":
                    break
                pl_sentence.append(value[1:])
                sentence.append(raw_data[i])
                i += 1
        i += 1
    data = pd.DataFrame({"pl": pl_sentence, "mig": sentence})
    return data


# Load data
zus_data = pd.read_csv("data/Praca_Socjalna_ZUS_Clean.txt")
ratow_data = pd.read_csv("data/Ratownictwo_Medyczne_Clean.txt")
urzed_data = pd.read_csv("data/Urzedy_Instytucje_Administracja_Clean.txt")
policja_data = pd.read_csv("data/Policja_Straz_Pozarna_Clean.txt")
raw_data = load_data("data/wypowiedzi.odt")
wyp_data = split_data_from_list(raw_data)

# Concatanate all data
all_data = pd.concat(
    [zus_data, ratow_data, urzed_data, policja_data, wyp_data], ignore_index=True
)

# Capitaliaze all sentences
all_data["pl"] = all_data["pl"].apply(capitalize_sentence)
all_data["mig"] = all_data["mig"].apply(capitalize_sentence)

# Check if duplicate rows exists
if sum(all_data.duplicated()) > 0:
    all_data.drop_duplicates(inplace=True)

# Create a dictionary for further preprocessing
raw_dataset_list = []
for i in range(0, len(all_data)):
    raw_dataset_list.append(
        {"translation": {"pl": all_data.iloc[i]["pl"], "mig": all_data.iloc[i]["mig"]}}
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
