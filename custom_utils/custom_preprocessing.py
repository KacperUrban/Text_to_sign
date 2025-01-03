import pandas as pd
from odf.opendocument import load
from odf import text, teletype
from odf.opendocument import load


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
        (
            word.capitalize()
            if word.lower() in cap_words
            else word.upper() if word in up_words else word.lower()
        )
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
        try:
            int(raw_data[i])
            pl_sentence.append(raw_data[i + 1])
            sentence.append(raw_data[i + 2])
            i += 3
        except (ValueError, IndexError):
            i += 1

    data = pd.DataFrame({"pl": pl_sentence, "mig": sentence})
    return data
