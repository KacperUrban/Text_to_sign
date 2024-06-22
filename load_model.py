from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline


if __name__ == "__main__":
    # load and save model
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-pl-en")

    print("Model and tokenizer are loading...")

    tokenizer.save_pretrained("model/tokenizer/")
    model.save_pretrained("model/model")

    print("Succesfull saving!")

    model = AutoModelForSeq2SeqLM.from_pretrained("model/model")
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/")
    print("First translation:")

    translation = pipeline("translation", model=model, tokenizer=tokenizer)
    print(translation("Lubie jesc jablka!"))
