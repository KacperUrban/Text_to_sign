from nltk.translate.bleu_score import corpus_bleu
from pandas import DataFrame
from transformers import pipeline
import pandas as pd
from typing import Tuple
from datasets import DatasetDict
import numpy as np


class CustomBleu:
    """_summary_
    """
    def __init__(self, data:DataFrame, translator:pipeline) -> None:
        """_summary_

        Args:
            data (DataFrame): _description_
            translator (pipeline): _description_
        """
        self.data = data
        self.translator = translator

    def prepare_label_to_bleu(self) -> dict:
        """_summary_

        Returns:
            dict: _description_
        """
        hashmap = {}
        for index, row in self.data.iterrows():
            polish_sentence = row['pl']
            sign_language_sentence = row['mig']
            
            if polish_sentence in hashmap:
                hashmap[polish_sentence].append(sign_language_sentence)
            else:
                hashmap[polish_sentence] = [sign_language_sentence]
        return hashmap

    def score(self, dataset:DatasetDict) -> Tuple[float, DataFrame]:
        """_summary_

        Args:
            dataset (DatasetDict): _description_

        Returns:
            Tuple[float, DataFrame]: _description_
        """
        ref_sent = self.prepare_label_to_bleu()
        translation_corpus = []
        reference_corpus = []
        for i in range(0, len(dataset["test"]["translation"])):
            value = ref_sent[dataset["test"]["translation"][i]["pl"]]
            translation = self.translator(dataset["test"]["translation"][i]["pl"])
            translation_corpus.append(translation[0]["translation_text"].split())
            if len(value) > 1:
                reference_corpus.append([j.split() for j in value])
            else:
                reference_corpus.append(value[0].split())
        b_score = corpus_bleu(reference_corpus, translation_corpus)
        compare_ref_trans = pd.DataFrame(
            {"reference": reference_corpus, "translation": translation_corpus}
        )
        return np.round(b_score,2), compare_ref_trans
