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

    tokenizer_de = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    model_de = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    print("Model and tokenizer are loading...")

    tokenizer.save_pretrained(TOKENIZER_DIR)
    model.save_pretrained(MODEL_DIR, safe_serialization=True)

    tokenizer_de.save_pretrained(TOKENIZER_DIR + "_de")
    model_de.save_pretrained(MODEL_DIR + "_de", safe_serialization=True)

    print("Succesfull saving!")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    model_de = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR + "_de")
    tokenizer_de = AutoTokenizer.from_pretrained(TOKENIZER_DIR + "_de")
    print("First translation:")

    translation = pipeline("translation", model=model, tokenizer=tokenizer)
    print(translation("Lubie jesc jablka!"))

    translation_de = pipeline("translation", model=model_de, tokenizer=tokenizer_de)
    print(translation_de("Ich esse gerne Äpfel!"))