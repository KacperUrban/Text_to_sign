from nltk.translate.bleu_score import corpus_bleu
from pandas import DataFrame
from transformers import pipeline
import pandas as pd
from typing import Tuple
from datasets import DatasetDict
import numpy as np


class CustomBleu:
    """That class is modified version of common functin BLEU from nltk library. In this class was added
    preprocessing phase on data and operate on DatasetDict from datasets library.
    """
    def __init__(self, data:DatasetDict, translator:pipeline) -> None:
        """Init function

        Args:
            data (DatasetDict): DatasetDict with all examples.
            translator (pipeline): Transformers pipeline
        """
        self.data = data
        self.translator = translator

    def prepare_label_to_bleu(self) -> dict:
        """This function build hashmap with sentence in polish and its referneces in sign polish language

        Returns:
            dict: Return dictionary with sentence in base language and its translation sentences.
        """
        hashmap = {}
        
        for dataset in ['train', 'valid', 'test']:
            for i in range(len(self.data[dataset])):
                polish_sentence = self.data[dataset][i]['translation']['pl']
                sign_language_sentence = self.data[dataset][i]['translation']['mig']
                if polish_sentence in hashmap:
                    hashmap[polish_sentence].append(sign_language_sentence)
                else:
                    hashmap[polish_sentence] = [sign_language_sentence]
        return hashmap

    def score(self) -> Tuple[float, DataFrame]:
        """Build a hashmap of sentences and its translations. The calculate BLEU score.
        
        Returns:
            Tuple[float, DataFrame]: Return BLEU score and DataFrame with references and translated sentences
        """
        ref_sent = self.prepare_label_to_bleu()
        translation_corpus = []
        reference_corpus = []
        for i in range(0, len(self.data["test"]["translation"])):
            value = ref_sent[self.data["test"]["translation"][i]["pl"]]
            translation = self.translator(self.data["test"]["translation"][i]["pl"])
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
