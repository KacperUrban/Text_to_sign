import os
import sys

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)


from config import MODEL_DIR, TOKENIZER_DIR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline


if __name__ == "__main__":
    # load and save model
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-pl-en")

    print("Model and tokenizer are loading...")

    tokenizer.save_pretrained(TOKENIZER_DIR)
    model.save_pretrained(MODEL_DIR)

    print("Succesfull saving!")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print("First translation:")

    translation = pipeline("translation", model=model, tokenizer=tokenizer)
    print(translation("Lubie jesc jablka!"))
