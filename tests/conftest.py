import pandas as pd
import pytest
import torch
from transformers import ModernBertConfig, ModernBertForSequenceClassification

from arbiter import config


@pytest.fixture(scope="session")
def study():
    return config.load()


@pytest.fixture
def tiny_model():
    cfg = ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        num_labels=1,
        classifier_pooling="mean",
        pad_token_id=0,
        global_attn_every_n_layers=1,
        local_attention=16,
    )
    torch.manual_seed(0)
    return ModernBertForSequenceClassification(cfg).float()


@pytest.fixture
def toy_qrels():
    return pd.DataFrame({"qid": ["a", "a", "a", "b"], "docid": ["d1", "d2", "d3", "e1"], "rel": [3, 1, 0, 1]})
