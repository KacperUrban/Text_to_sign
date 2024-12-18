from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import pandas as pd
from custom_utils.custom_pytorch_utils import cross_validation_pt


if __name__ == '__main__':
    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSeq2SeqLM.from_pretrained("model/model")
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer/")

    data = pd.read_csv("data/final_data/all_data.csv")

    print(f"Average BLEU score: {cross_validation_pt(model, tokenizer, data, device)}")
