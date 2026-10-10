import numpy as np
import torch

from ..settings import load_config, model_dir
from .base import Pair, Reranker, Scored, length_order, restore, sigmoid

QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
TOKEN_BUDGET = 16384


def half_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


class EncoderReranker(Reranker):
    def __init__(self, model_id: str, device: str):
        super().__init__(model_id, device)
        from sentence_transformers import CrossEncoder

        self.spec = load_config().models[model_id]
        self.model = CrossEncoder(
            str(model_dir(model_id)),
            device=device,
            max_length=self.spec.context,
            activation_fn=torch.nn.Identity(),
            model_kwargs={"torch_dtype": half_dtype(device)},
        )

    def score_pairs(self, pairs: list[Pair], max_tokens: int | None = None, batch_size: int | None = None) -> Scored:
        if not pairs:
            return Scored(np.zeros(0), np.zeros(0))
        self.model.max_length = min(max_tokens or self.spec.context, self.spec.context)
        order = length_order(pairs)
        logits = self.model.predict(
            [pairs[i] for i in order],
            batch_size=batch_size or self.spec.batch,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        logits = np.asarray(logits, dtype=np.float64).reshape(len(pairs))
        return Scored(restore(order, logits), restore(order, sigmoid(logits)))

    def close(self) -> None:
        del self.model


class QwenReranker(Reranker):
    def __init__(self, model_id: str, device: str, instruction: str | None = QWEN_INSTRUCTION):
        super().__init__(model_id, device)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = load_config().models[model_id]
        path = str(model_dir(model_id))
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device).eval()
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix = self.tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)

    def render(self, query: str, doc: str) -> str:
        if self.instruction is None:
            return f"<Query>: {query}\n<Document>: {doc}"
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    @torch.no_grad()
    def score_pairs(self, pairs: list[Pair], max_tokens: int | None = None, batch_size: int | None = None) -> Scored:
        if not pairs:
            return Scored(np.zeros(0), np.zeros(0))
        limit = min(max_tokens or self.spec.context, self.spec.context)
        body = limit - len(self.prefix) - len(self.suffix)
        order = length_order(pairs)
        texts = [self.render(*pairs[i]) for i in order]
        step = batch_size or self.spec.batch
        encoded = self.tokenizer(
            texts, padding=False, truncation="longest_first", return_attention_mask=False, max_length=body
        )["input_ids"]
        sequences = [self.prefix + ids + self.suffix for ids in encoded]
        out = []
        start = 0
        while start < len(sequences):
            end = start + 1
            widest = len(sequences[start])
            while end < len(sequences) and end - start < step:
                widest = max(widest, len(sequences[end]))
                if widest * (end - start + 1) > TOKEN_BUDGET:
                    break
                end += 1
            batch = self.tokenizer.pad({"input_ids": sequences[start:end]}, padding=True, return_tensors="pt")
            batch = batch.to(self.device)
            logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :].float()
            pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
            out.append(torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().numpy())
            start = end
        probs = np.concatenate(out).astype(np.float64)
        return Scored(restore(order, probs), restore(order, probs))

    def close(self) -> None:
        del self.model


class JinaReranker(Reranker):
    def __init__(self, model_id: str, device: str, instruction: str | None = None):
        super().__init__(model_id, device)
        from transformers import AutoModelForSequenceClassification

        self.spec = load_config().models[model_id]
        self.instruction = instruction
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                str(model_dir(model_id)),
                torch_dtype=str(half_dtype(device)).removeprefix("torch."),
                trust_remote_code=True,
                use_flash_attn=False,
            )
            .to(device)
            .eval()
        )

    def score_pairs(self, pairs: list[Pair], max_tokens: int | None = None, batch_size: int | None = None) -> Scored:
        if not pairs:
            return Scored(np.zeros(0), np.zeros(0))
        limit = min(max_tokens or self.spec.context, self.spec.context)
        order = length_order(pairs)
        prefix = f"{self.instruction}\n" if self.instruction else ""
        ordered = [[prefix + pairs[i][0], pairs[i][1]] for i in order]
        with torch.no_grad():
            probs = self.model.compute_score(
                ordered, max_length=limit, batch_size=max(1, min(batch_size or self.spec.batch, 16384 // limit))
            )
        probs = np.asarray(probs, dtype=np.float64)
        return Scored(restore(order, probs), restore(order, probs))

    def close(self) -> None:
        del self.model
