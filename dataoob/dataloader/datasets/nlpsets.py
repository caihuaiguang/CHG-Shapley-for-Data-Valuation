"""NLP data sets.

Run ``make install-extra`` as
`transformers <https://huggingface.co/docs/transformers/index>`_. is an optional
dependency.
"""
from typing import Callable

import numpy as np
import pandas as pd
import torch

from dataoob.dataloader.register import Register, cache
from dataoob.dataloader.util import ListDataset

MAX_DATASET_SIZE = 1000
"""Data Valuation algorithms can take a long time for large data sets, thus cap size."""


def bert_embeddings(func: Callable[[str, bool], tuple[ListDataset, np.ndarray]]):
    """Convert text data into pooled embeddings with DistilBERT model.

    Given a data set with a list of string, such as NLP data set function (see below),
    converts the sentences into strings. It is the equivalent of training a downstream
    task with bert but all the BERT layers are frozen. It is advised to just
    train with the raw strings with a BERT model located in models/bert.py or defining
    your own model. DistilBERT is just a faster version of BERT

    References
    ----------
    .. [1] J. Devlin, M.W. Chang, K. Lee, and K. Toutanova,
        BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        arXiv.org, 2018. Available: https://arxiv.org/abs/1810.04805.
    .. [2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf,
        DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
        arXiv.org, 2019. Available: https://arxiv.org/abs/1910.01108.
    """

    def wrapper(
        cache_dir: str, force_download: bool, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        from transformers import DistilBertModel, DistilBertTokenizerFast

        BERT_PRETRAINED_NAME = "distilbert-base-uncased"  # TODO update this

        tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PRETRAINED_NAME)
        bert_model = DistilBertModel.from_pretrained(BERT_PRETRAINED_NAME)

        dataset, labels = func(cache_dir, force_download, **kwargs)
        entries = [entry for entry in dataset[:MAX_DATASET_SIZE]]
        res = tokenizer.__call__(
            entries, max_length=250, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            pooled_embeddings = bert_model(res.input_ids, res.attention_mask)[0][:, 0]
        return pooled_embeddings.numpy(force=True), np.array(labels)

    return wrapper


@Register("bbc", cacheable=True, categorical=True)
def download_bbc(cache_dir: str, force_download: bool):
    """Classification data set registered as ``"bbc"``.

    Predicts type of article from the article. Used in NLP data valuation tasks.

    References
    ----------
    .. [1] D. Greene and P. Cunningham,
        Practical Solutions to the Problem of Diagonal Dominance in
        Kernel Document Clustering, Proc. ICML 2006.
    """
    github_url = (
        "https://raw.githubusercontent.com/"
        "mdsohaib/BBC-News-Classification/master/bbc-text.csv"
    )
    cache(github_url, cache_dir, "bbc-text.csv", force_download)
    df = pd.read_csv(cache_dir + "bbc-text.csv")

    label_dict = {
        "business": 0,
        "entertainment": 1,
        "sport": 2,
        "tech": 3,
        "politics": 4,
    }
    labels = np.fromiter((label_dict[label] for label in df["category"]), dtype=int)

    return ListDataset(df["text"].values), labels


@Register("imdb", cacheable=True, categorical=True)
def download_imdb(cache_dir: str, force_download: bool):
    """Binary category sentiment analysis data set registered as ``"imdb"``.

    Predicts sentiment analysis of the review as either positive (1) or negative (0).
    Used in NLP data valuation tasks.

    References
    ----------
    .. [1] A. Maas, R. Daly, P. Pham, D. Huang, A. Ng, and C. Potts.
        Learning Word Vectors for Sentiment Analysis.
        The 49th Annual Meeting of the Association for Computational Linguistics (2011).
    """
    github_url = (
        "https://raw.githubusercontent.com/"
        "Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    )
    cache(github_url, cache_dir, "imdb.csv", force_download)
    df = pd.read_csv(cache_dir + "imdb.csv")

    label_dict = {"negative": 0, "positive": 1}
    labels = np.fromiter((label_dict[label] for label in df["sentiment"]), dtype=int)

    return ListDataset(df["review"].values), labels


bbc_embedding = Register("bbc-embeddings", True, True)(bert_embeddings(download_bbc))
"""Classification data set registered as ``"bbc-embeddings"``, BERT text embeddings."""

imdb_embedding = Register("imdb-embeddings", True, True)(bert_embeddings(download_imdb))
"""Classification data set registered as ``"imdb-embeddings"``, BERT text embeddings."""
