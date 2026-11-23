import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForSequenceClassification

from arbiter.arms import arm_state
from arbiter.config import Study


def tokenizer(study: Study):
    return AutoTokenizer.from_pretrained(study.path("models") / study.backbone)


def build(study: Study, arm: str, seed: int, pooling: str | None = None) -> tuple[ModernBertForSequenceClassification, int]:
    """cross-encoder with the arm's encoder weights; the head depends only on the seed."""
    cfg = ModernBertConfig.from_pretrained(study.path("models") / study.backbone)
    cfg.num_labels = 1
    cfg.classifier_pooling = pooling or study.train["pooling"]
    torch.manual_seed(seed)
    model = ModernBertForSequenceClassification(cfg)
    sd = arm_state(study, arm)
    target = model.model.state_dict()
    bad = [k for k in target if k not in sd or sd[k].shape != target[k].shape]
    if bad:
        raise ValueError(f"{arm}: {len(bad)} encoder tensors missing or mis-shaped, e.g. {bad[:3]}")
    model.model.load_state_dict({k: sd[k] for k in target}, strict=True)
    return model, len(target)


def load_trained(study: Study, path, pooling: str | None = None) -> ModernBertForSequenceClassification:
    cfg = ModernBertConfig.from_pretrained(study.path("models") / study.backbone)
    cfg.num_labels = 1
    cfg.classifier_pooling = pooling or study.train["pooling"]
    model = ModernBertForSequenceClassification(cfg)
    model.load_state_dict(load_file(str(path)), strict=True)
    return model
