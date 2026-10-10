import csv
import functools
import gzip
import json
import random
from collections import defaultdict
from dataclasses import dataclass

from .settings import DATA_DIR, load_config


@dataclass
class Dataset:
    name: str
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]]


def read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            yield json.loads(line)


def load_beir(name: str) -> Dataset:
    root = DATA_DIR / name
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(root / "qrels" / "test.tsv") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for qid, did, grade in reader:
            qrels[qid][did] = int(grade)
    queries = {row["_id"]: row["text"] for row in read_jsonl(root / "queries.jsonl") if row["_id"] in qrels}
    corpus = {}
    for row in read_jsonl(root / "corpus.jsonl"):
        title = row.get("title", "").strip()
        corpus[row["_id"]] = f"{title}. {row['text']}" if title else row["text"]
    judged = {qid: grades for qid, grades in qrels.items() if qid in queries and any(grades.values())}
    return Dataset(name, {q: queries[q] for q in judged}, corpus, judged)


def load_trec(name: str) -> Dataset:
    root = DATA_DIR / name
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(root / "qrels.txt") as handle:
        for line in handle:
            qid, _, pid, grade = line.split()
            qrels[qid][pid] = int(grade)
    judged = {qid for qid, grades in qrels.items() if any(grade >= 1 for grade in grades.values())}
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    with gzip.open(root / "top1000.tsv.gz", "rt") as handle:
        for line in handle:
            qid, pid, query, passage = line.rstrip("\n").split("\t")
            if qid in judged:
                queries[qid] = query
                corpus[pid] = passage
    kept = {qid: dict(qrels[qid]) for qid in queries}
    return Dataset(name, queries, corpus, kept)


@functools.cache
def load_dataset(name: str) -> Dataset:
    spec = load_config().datasets[name]
    return load_beir(name) if spec.source == "beir" else load_trec(name)


@dataclass
class Passage:
    pid: str
    text: str
    turns: list[str]


@dataclass
class LongCorpus:
    passages: list[Passage]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def conversation_turns(conversation: dict) -> list[tuple[str, str]]:
    turns = []
    index = 1
    while f"session_{index}" in conversation:
        stamp = conversation.get(f"session_{index}_date_time", "")
        for turn in conversation[f"session_{index}"]:
            line = f"{turn['speaker']}: {turn['text']}"
            if turn.get("blip_caption"):
                line += f" [shared image: {turn['blip_caption']}]"
            turns.append((turn["dia_id"], f"[{stamp}] {line}" if stamp else line))
        index += 1
    return turns


def load_locomo(chunk_tokens: int, token_length, sample: int, seed: int) -> LongCorpus:
    records = json.loads((DATA_DIR / "locomo" / "locomo10.json").read_text())
    passages: list[Passage] = []
    turn_to_passage: dict[tuple[int, str], str] = {}
    for number, record in enumerate(records):
        turns = conversation_turns(record["conversation"])
        current: list[tuple[str, str]] = []
        size = 0
        chunks = []
        for dia_id, line in turns:
            cost = token_length(line)
            if current and size + cost > chunk_tokens:
                chunks.append(current)
                current, size = [], 0
            current.append((dia_id, line))
            size += cost
        if current:
            chunks.append(current)
        for position, chunk in enumerate(chunks):
            pid = f"c{number}-{position}"
            passages.append(Passage(pid, "\n".join(line for _, line in chunk), [line for _, line in chunk]))
            for dia_id, _ in chunk:
                turn_to_passage[(number, dia_id)] = pid
    pool = []
    for number, record in enumerate(records):
        for question in record["qa"]:
            if question.get("evidence") and question["category"] != 5:
                pool.append((number, question))
    random.Random(seed).shuffle(pool)
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for position, (number, question) in enumerate(pool):
        hits = {turn_to_passage.get((number, ev.strip())) for ev in question["evidence"]}
        hits.discard(None)
        if not hits:
            continue
        qid = f"q{position}"
        queries[qid] = question["question"]
        qrels[qid] = {pid: 1 for pid in hits}
        if len(queries) == sample:
            break
    return LongCorpus(passages, queries, qrels)
