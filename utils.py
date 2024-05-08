from abc import ABC
from typing import Any
import torch
import numpy as np
import pandas as pd
import logging
import voyageai
from openai import OpenAI
from mteb import MTEB
from mteb.tasks import TatoebaBitextMining
import os
import json


def get_logger() -> logging.Logger:
    """Returns a configured logger"""
    logging.basicConfig(
        filename="mteb.log",
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger("main")


logger = get_logger()


class ClientManager:
    """Manages the clients for the external models."""

    def __init__(self):
        self.oai_client = None
        self.voyage_client = None

    def get_oai_client(self):
        if self.oai_client is None:
            self.oai_client = OpenAI()
        return self.oai_client

    def get_voyage_client(self):
        if self.voyage_client is None:
            self.voyage_client = voyageai.Client()
        return self.voyage_client


class ExternalModel(ABC):
    """Abstract class for models accessable via external API"""

    def __init__(self, model_name: str, client: Any, **kwargs):
        self.model_name = model_name
        self.client = client

    def encode(
        self, sentences: list[str], batch_size: int, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            batch_size (`int`): Batch size for the encoding
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        pass


class VoyageAPIModel(ExternalModel):
    """Voyage API model for encoding sentences."""

    def encode(self, sentences: list[str], batch_size=128, **kwargs: Any) -> np.ndarray:

        embeddings = []
        for i in range(0, len(sentences), batch_size):
            embeddings += self.client.embed(
                sentences[i : i + batch_size],
                model=self.model_name,
                input_type="document",
            ).embeddings

        logger.info(
            f"Embeddings shape: {np.array(embeddings).shape}, i: {i}/{len(sentences)}"
        )
        return np.array(embeddings)


class OpenAIAPIModel(ExternalModel):
    """OpenAI API model for encoding sentences."""

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        logger.log(logging.INFO, f"Encoding {len(sentences)} sentences")
        return np.array(
            [
                self.client.embeddings.create(input=sentence, model=self.model_name)
                .data[0]
                .embedding
                for sentence in sentences
            ]
        )


def run_law_benchmarks(model: ExternalModel, logger: logging.Logger):
    """Runs the law benchmarks for the given model via API"""

    model_name = model.model_name

    TASK_LIST_RETRIEVAL_LAW = [
        "LegalSummarization",  # Contracts and their summarizations
        "LegalBenchConsumerContractsQA",  # Questions and answers on contracts
        "LegalBenchCorporateLobbying",  # Corporate lobbying bill titles and summaries
        # "LegalQuAD",  # German legal questions and relevant cases, very long context [ (oai) 34k fail / 8k limit  ; (voyage) 1M fail / 120k limit ]
        # "GerDaLIRSmall",  # German legal cases, p2p!, 2M token limit on voyage
    ]

    for task in TASK_LIST_RETRIEVAL_LAW:
        logger.info(f"Running task: {task} for model: {model_name}")
        eval_splits = ["test"]
        evaluation = MTEB(tasks=[task])
        evaluation.run(
            model, output_folder=f"results/{model_name}", eval_splits=eval_splits
        )


def run_cross_lingual_benchmarks(model: ExternalModel, logger: logging.Logger):
    """Runs the cross-lingual benchmarks for the given model via API"""

    model_name = model.model_name

    # MULTILINGUAL_TASKS = ["BUCC", "Tatoeba"]
    # MULTILINGUAL_TASKS = ["Tatoeba"]

    # for task in MULTILINGUAL_TASKS:
    #     logger.info(f"Running task: {task} for model: {model_name}")
    #     eval_splits = ["test"]
    #     evaluation = MTEB(tasks=[task])
    #     evaluation.run(
    #         model, output_folder=f"results/{model_name}", eval_splits=eval_splits
    #     )

    eval_splits = ["test"]
    lang_pairs = [
        "swe-eng",
        "spa-eng",
        "dan-eng",
        "fin-eng",
        "deu-eng",
        "fra-eng",
        "por-eng",
        "pol-eng",
        "nld-eng",
        "nno-eng",
    ]

    for lang_pair in lang_pairs:
        logger.info(
            f"Running task: Tatoeba, with lang: {lang_pair}, for model: {model_name}"
        )
        evaluation = MTEB(tasks=[TatoebaBitextMining(langs=[lang_pair])])
        evaluation.tasks[0].metadata.name = f"Tatoeba_{lang_pair}"
        evaluation.run(
            model,
            output_folder=f"results/{model_name}",
            eval_splits=eval_splits,
        )


# Results data wrangling


def get_partial_results_json(path: str) -> dict:
    """Returns a dictionary with the results of the vanilla benchmarks"""
    with open(path, "r") as f:
        results = json.load(f)
    return {
        "mteb_dataset_name": results["mteb_dataset_name"],
        "evaluation_time": results["test"]["evaluation_time"],
        "ndcg@10": results["test"]["ndcg_at_10"],
    }


def get_cross_lingual_results_json(path: str) -> dict:
    """Returns a dictionary with the results of the cross-lingual benchmarks"""
    with open(path, "r") as f:
        results = json.load(f)
    scores = extract_bitext_scores(results["test"])
    return {
        "mteb_dataset_name": results["mteb_dataset_name"],
        "evaluation_time": results["test"]["evaluation_time"],
        "f1": scores["f1"],
    }


def get_all_results():
    """Returns a both dictionaries with the results of the vanilla and cross-lingual benchmarks"""

    all_results = []
    cross_lingual_results = []
    for model_name in os.listdir("results/"):
        path = f"results/{model_name}"
        if os.path.isdir(path):
            for file in os.listdir(path):
                file_path = f"{path}/{file}"
                if not file.startswith("Tatoeba"):
                    d = get_partial_results_json(file_path)
                    d["model_name"] = model_name
                    all_results.append(d)
                else:
                    d = get_cross_lingual_results_json(file_path)
                    d["model_name"] = model_name
                    cross_lingual_results.append(d)
    return all_results, cross_lingual_results


def extract_bitext_scores(test_dict: dict) -> dict:
    """Extracts the scores from the bitext mining results"""

    for key, value in test_dict.items():
        if not key.startswith("evaluation_time"):
            return value


def get_vanilla_results_df(values=["ndcg@10", "evaluation_time"]) -> pd.DataFrame:
    """Returns a DataFrame with the vanilla results"""

    df = pd.DataFrame(get_all_results()[0])
    pivot_df = df.pivot_table(
        columns="mteb_dataset_name",
        index="model_name",
        values=values,
        aggfunc="mean",
        margins=True,
        margins_name="Average",
    )
    return pivot_df


def get_cross_lingual_results_df(values=["scores", "evaluation_time"]) -> pd.DataFrame:
    """Returns a DataFrame with the cross-lingual results"""

    df = pd.DataFrame(get_all_results()[1])
    pivot_df = df.pivot_table(
        columns="mteb_dataset_name",
        index="model_name",
        values=values,
        aggfunc="mean",
        margins=True,
        margins_name="Average",
    )
    return pivot_df


def get_weighted_average_df(a: float = 1) -> pd.DataFrame:
    """Returns a DataFrame with the weighted average of the cross-lingual results"""

    f1 = get_cross_lingual_results_df(["f1"])["f1"].copy()
    order = [
        "Tatoeba_swe-eng",
        "Tatoeba_spa-eng",
        "Tatoeba_dan-eng",
        "Tatoeba_fin-eng",
        "Tatoeba_deu-eng",
        "Tatoeba_fra-eng",
        "Tatoeba_por-eng",
        "Tatoeba_pol-eng",
        "Tatoeba_nld-eng",
        "Tatoeba_nno-eng",
    ]

    weights = [(-a * x + len(order)) / len(order) for x in range(len(order))]
    f1["W_Average"] = np.average(f1[order], weights=weights, axis=1)
    order += ["Average", "W_Average"]
    return f1[order]
