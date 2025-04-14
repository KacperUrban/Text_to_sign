import os
import sys

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)


from config import MODEL_DIR, TOKENIZER_DIR, TOKENIZER_DE_DIR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline


def load_save_check_model(
    PATH_R: str, PATH_M_F: str, PATH_T_F: str, sentence_to_translation: str
) -> None:
    """This function load model and tokenizer from remote HuggingFace repository. Then save model to the local folder
    and do translation on specified sentence. The process have to ensure that base model was properly fetched and saved.

    Args:
        PATH_R (str): path to remote model project (e. g. Helsinki-NLP/opus-mt-pl-en)
        PATH_M_F (str): path where will be model stored
        PATH_T_F (str): path where will be tokenizer stored
        sentence_to_translation (str): as name says this sentence wil be translated
    """
    print("Model and tokenizer are loading...")
    tokenizer = AutoTokenizer.from_pretrained(PATH_R)
    model = AutoModelForSeq2SeqLM.from_pretrained(PATH_R)

    tokenizer.save_pretrained(PATH_T_F)
    model.save_pretrained(PATH_M_F, safe_serialization=True)

    print(f"Filepath to tokenizer: {PATH_T_F}")
    print(f"Filepath to model: {PATH_M_F}")
    print("Succesfully saved!")

    tokenizer = AutoTokenizer.from_pretrained(PATH_T_F)
    model = AutoModelForSeq2SeqLM.from_pretrained(PATH_M_F)

    print("First translation:")
    translation = pipeline("translation", model=model, tokenizer=tokenizer)
    print(translation(sentence_to_translation))


if __name__ == "__main__":

    PATHS = {"pl": "Helsinki-NLP/opus-mt-pl-en", "de": "Helsinki-NLP/opus-mt-de-en"}
    load_save_check_model(PATHS["pl"], MODEL_DIR, TOKENIZER_DIR, "Lubie jesc jablka!")
    load_save_check_model(
        PATHS["de"], MODEL_DIR + "_de", TOKENIZER_DE_DIR, "Ich esse gerne Äpfel!"
    )
