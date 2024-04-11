from nltk.translate.bleu_score import corpus_bleu
from pandas import DataFrame
from transformers import pipeline
import pandas as pd
from typing import Tuple
from datasets import DatasetDict


class CustomBleu:
    def __init__(self, data:DataFrame, translator:pipeline) -> None:
        self.data = data
        self.translator = translator

    def prepare_label_to_bleu(self) -> dict:
        hashmap = {}
        for i in range(0, self.data.shape[0]):
            if self.data.iloc[i, 0] in hashmap:
                hashmap[self.data.iloc[i, 0]] = hashmap[self.data.iloc[i, 0]].append(
                    self.data.iloc[i, 1]
                )
            else:
                hashmap[self.data.iloc[i, 0]] = [self.data.iloc[i, 1]]
        return hashmap

    def score(self, dataset:DatasetDict) -> Tuple[float, DataFrame]:
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
        return b_score, compare_ref_trans
