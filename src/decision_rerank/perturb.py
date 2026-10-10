import json
import random
import re

import nltk
import torch
from nltk.corpus import wordnet
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import release
from .settings import CACHE_DIR, DATA_DIR, MODELS_DIR, ensure_dirs

nltk.data.path.insert(0, str(DATA_DIR / "nltk"))
KEYBOARD = "qwertyuiopasdfghjklzxcvbnm"
PARAPHRASE_PROMPT = (
    "Rewrite the following search query so that it keeps exactly the same meaning but uses different "
    "wording. Reply with the rewritten query only.\n\nQuery: {query}"
)


def add_typos(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    eligible = [i for i, w in enumerate(words) if len(re.sub(r"\W", "", w)) >= 4]
    if not eligible:
        return text
    chosen = rng.sample(eligible, max(1, round(0.2 * len(eligible))))
    for i in chosen:
        word = words[i]
        spot = rng.randrange(1, len(word) - 1)
        action = rng.choice(["swap", "drop", "double", "replace"])
        if action == "swap":
            word = word[:spot] + word[spot + 1] + word[spot] + word[spot + 2 :]
        elif action == "drop":
            word = word[:spot] + word[spot + 1 :]
        elif action == "double":
            word = word[:spot] + word[spot] + word[spot:]
        else:
            word = word[:spot] + rng.choice(KEYBOARD) + word[spot + 1 :]
        words[i] = word
    return " ".join(words)


def swap_synonyms(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    options: dict[int, list[str]] = {}
    for i, word in enumerate(words):
        bare = re.sub(r"\W", "", word)
        if len(bare) < 4 or not bare.isalpha():
            continue
        found = {
            lemma.name().replace("_", " ")
            for synset in wordnet.synsets(bare.lower())
            for lemma in synset.lemmas()
            if lemma.name().lower() != bare.lower() and "_" not in lemma.name()
        }
        if found:
            options[i] = sorted(found)
    if not options:
        return text
    for i in rng.sample(sorted(options), min(2, len(options))):
        words[i] = rng.choice(options[i])
    return " ".join(words)


def paraphrase_queries(queries: list[str], count: int, seed: int, device: str) -> dict[str, list[str]]:
    path = MODELS_DIR / "paraphraser"
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device).eval()
    torch.manual_seed(seed)
    out: dict[str, list[str]] = {}
    for query in queries:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": PARAPHRASE_PROMPT.format(query=query)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=64,
                num_return_sequences=count,
                pad_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        out[query] = [t.strip().strip('"').split("\n")[0].strip() or query for t in texts]
    del model
    release(device)
    return out


def build_perturbations(queries: dict[str, str], count: int, seed: int, device: str) -> dict[str, dict[str, list[str]]]:
    ensure_dirs()
    path = CACHE_DIR / "perturbations.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if set(cached) == set(queries):
            return cached
    paraphrases = paraphrase_queries(list(queries.values()), count, seed, device)
    out: dict[str, dict[str, list[str]]] = {}
    for qid, text in queries.items():
        rng = random.Random(f"{seed}:{qid}")
        typos, synonyms = [], []
        for _ in range(count):
            for _ in range(10):
                candidate = add_typos(text, rng)
                if candidate != text:
                    break
            typos.append(candidate)
            for _ in range(10):
                candidate = swap_synonyms(text, rng)
                if candidate != text:
                    break
            synonyms.append(candidate)
        out[qid] = {"typos": typos, "synonyms": synonyms, "paraphrase": paraphrases[text]}
    path.write_text(json.dumps(out))
    return out
