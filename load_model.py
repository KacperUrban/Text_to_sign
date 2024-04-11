from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from transformers import pipeline

# load and save model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
model = TFAutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-pl-en")

tokenizer.save_pretrained("model/tokenizer/")
model.save_pretrained("model/model")

print("Succesfull saving!")

model = TFAutoModelForSeq2SeqLM.from_pretrained("model/model")
tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/")
print("First translation:")

translation = pipeline("translation", model=model, tokenizer=tokenizer)
print(translation("Lubie jesc jablka!"))
