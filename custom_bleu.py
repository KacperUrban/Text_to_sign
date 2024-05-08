from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from pandas import DataFrame
from transformers import pipeline
import pandas as pd
from typing import Tuple
from datasets import DatasetDict
import numpy as np
from tqdm import tqdm
import os


class CustomBleu:
    """That class is modified version of common functin BLEU from nltk library. In this class was added
    preprocessing phase on data and operate on DatasetDict from datasets library. Additionaly this class
    enable users get a summary dataframe.
    """

    def __init__(self, data: DatasetDict, translator: pipeline) -> None:
        """Init function

        Args:
            data (DatasetDict): DatasetDict with all examples.
            translator (pipeline): Transformers pipeline.

        Examples:
        Below it is an example of correct DatasetDict structure:
        DatasetDict({
            train: Dataset({
                features: ['translation'],
                num_rows: 968
            })
            valid: Dataset({
                features: ['translation'],
                num_rows: 121
            })
            test: Dataset({
                features: ['translation'],
                num_rows: 122
            })
        })

        Below also it is example of how to create transfomers pipeline for translation problem:

        from transformers import pipeline
        translator = pipeline('translation', model=model, tokenizer=tokenizer)
        """
        self.data = data
        self.translator = translator

    def prepare_label_to_bleu(self) -> dict:
        """This function build hashmap with sentence in polish and its referneces in sign polish language.

        Returns:
            dict: Return dictionary with sentence in base language and its translation sentences.
        """
        hashmap = {}

        for dataset in ["train", "valid", "test"]:
            for i in range(len(self.data[dataset])):
                polish_sentence = self.data[dataset][i]["translation"]["pl"]
                sign_language_sentence = self.data[dataset][i]["translation"]["mig"]
                if polish_sentence in hashmap:
                    hashmap[polish_sentence].append(sign_language_sentence)
                else:
                    hashmap[polish_sentence] = [sign_language_sentence]
        return hashmap

    def score(
        self, dataset_name: str, generate_xlsx: bool = False
    ) -> Tuple[float, DataFrame]:
        """Build a hashmap of the sentences and its translations. Hashmap is build in prepare_label_to_bleu.
        The calculate BLEU score and some summary in DataFrame.
        Args:
        dataset_name (str): This parameter describe, which dataset we will be using to calculate BLEU score.
        generate_xlsx (str): Despite returned DataFrame, user can get the excel file with summary DataFrame. If you
        want this, set this parameter to True.

        Returns:
            Tuple[float, DataFrame]: Return BLEU score and DataFrame with summary like BLEU score for every sentence,
            translation sentence, refernece sentence, sentence in polish etc.
        """
        smoothing_func = SmoothingFunction()
        ref_sent = self.prepare_label_to_bleu()
        sum_bleu = 0
        bleu_list = []
        translation_corpus = []
        reference_corpus_to_display = []
        polish_sentences = []
        identity_check = []
        for i in tqdm(range(len(self.data[dataset_name]["translation"]))):
            value = ref_sent[self.data[dataset_name]["translation"][i]["pl"]]
            translation = self.translator(
                self.data[dataset_name]["translation"][i]["pl"]
            )
            translation_processed = translation[0]["translation_text"].split()
            translation_corpus.append(translation[0]["translation_text"])
            polish_sentences.append(self.data[dataset_name]["translation"][i]["pl"])
            if len(value) > 1:
                tmp_var = [j.split() for j in value]
                bleu = sentence_bleu(
                    tmp_var,
                    translation_processed,
                    smoothing_function=smoothing_func.method1,
                )
                sum_bleu += bleu * 100
                bleu_list.append(np.round(bleu * 100, 2))
                reference_corpus_to_display.append(", ".join(value))
                identity_check.append(translation_processed in tmp_var)
            else:
                tmp_var = value[0].split()
                bleu = sentence_bleu(
                    tmp_var,
                    translation_processed,
                    smoothing_function=smoothing_func.method1,
                )
                sum_bleu += bleu * 100
                bleu_list.append(np.round(bleu * 100, 2))
                reference_corpus_to_display.append("".join(value))
                identity_check.append(translation_processed == tmp_var)
        summary_df = pd.DataFrame(
            {
                "polish sentence": polish_sentences,
                "reference": reference_corpus_to_display,
                "translation": translation_corpus,
                "is_identical": identity_check,
                "bleu": bleu_list,
            }
        )

        if generate_xlsx:
            path = "summary.xlsx"
            if os.path.exists(path):
                with pd.ExcelWriter(
                    path, mode="a", if_sheet_exists="overlay"
                ) as writer:
                    summary_df.to_excel(writer, sheet_name=dataset_name)
            else:
                with pd.ExcelWriter(path, mode="w") as writer:
                    summary_df.to_excel(writer, sheet_name=dataset_name)

        return (
            np.round(sum_bleu / len(self.data[dataset_name]["translation"]), 2),
            summary_df,
        )
