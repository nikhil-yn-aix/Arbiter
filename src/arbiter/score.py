import numpy as np
import pandas as pd
import torch


def encode_pairs(tok, queries: list[str], docs: list[str], max_len: int) -> list[list[int]]:
    enc = tok(queries, docs, truncation="only_second", max_length=max_len, add_special_tokens=True)
    return enc["input_ids"]


def token_batches(lengths: np.ndarray, max_tokens: int) -> list[np.ndarray]:
    """length-sorted batches whose padded size stays within max_tokens."""
    order = np.argsort(lengths, kind="stable")
    batches, cur, longest = [], [], 0
    for i in order:
        n = int(lengths[i])
        if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
            batches.append(np.array(cur))
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        batches.append(np.array(cur))
    return batches


def pad(ids: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(x) for x in ids)
    out = torch.full((len(ids), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), n), dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = 1
    return out.to(device, non_blocking=True), mask.to(device, non_blocking=True)


@torch.inference_mode()
def score(model, tok, cands: pd.DataFrame, max_len: int, max_tokens: int) -> np.ndarray:
    device = next(model.parameters()).device
    enc = tok(cands["query"].tolist(), cands["text"].tolist(), truncation="only_second", max_length=max_len)
    ids = enc["input_ids"]
    types = enc.get("token_type_ids") if getattr(model.config, "type_vocab_size", 1) > 1 else None
    lengths = np.array([len(x) for x in ids])
    out = np.zeros(len(ids), dtype=np.float32)
    was_training = model.training
    model.eval()
    for b in token_batches(lengths, max_tokens):
        x, m = pad([ids[i] for i in b], tok.pad_token_id, device)
        kw = {"token_type_ids": pad([types[i] for i in b], 0, device)[0]} if types is not None else {}
        logits = model(input_ids=x, attention_mask=m, **kw).logits.float()
        out[b] = logits[:, 0].cpu().numpy() if logits.shape[1] == 1 else (logits[:, -1] - logits[:, 0]).cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite scores")
    model.train(was_training)
    return out
