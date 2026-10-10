import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile

from huggingface_hub import snapshot_download

from .datasets import load_beir, load_trec
from .settings import DATA_DIR, MODELS_DIR, load_config

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
MSMARCO_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
TREC_URL = "https://trec.nist.gov/data/deep/{year}qrels-pass.txt"
WORDNET_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def download(url: str, target) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "decision-rerank"})
    with urllib.request.urlopen(request) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out)
    partial.rename(target)


def digest(path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def fetch_models() -> dict:
    config = load_config()
    wanted = {name: (spec.repo, spec.allow) for name, spec in config.models.items()}
    wanted.update({name: (spec.repo, spec.allow) for name, spec in config.support_models.items()})
    path = MODELS_DIR / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for name, (repo, allow) in wanted.items():
        target = MODELS_DIR / name
        if not (target / "config.json").exists():
            print(f"model {name} <- {repo}", flush=True)
            snapshot_download(repo_id=repo, local_dir=target, allow_patterns=allow)
        if name in manifest:
            continue
        weights = sorted(target.glob("*.safetensors"))
        manifest[name] = {
            "repo": repo,
            "bytes": sum(f.stat().st_size for f in weights),
            "sha256": {f.name: digest(f) for f in weights},
        }
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def fetch_beir(name: str) -> None:
    target = DATA_DIR / name
    if (target / "corpus.jsonl").exists():
        return
    archive = DATA_DIR / f"{name}.zip"
    download(BEIR_URL.format(name=name), archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(DATA_DIR)
    archive.unlink()


def fetch_trec(year: int) -> None:
    target = DATA_DIR / f"trec-dl-{year}"
    download(MSMARCO_URL + f"msmarco-passagetest{year}-top1000.tsv.gz", target / "top1000.tsv.gz")
    download(TREC_URL.format(year=year), target / "qrels.txt")


def fetch_wordnet() -> None:
    target = DATA_DIR / "nltk" / "corpora"
    if (target / "wordnet").exists():
        return
    archive = target / "wordnet.zip"
    download(WORDNET_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    archive.unlink()


def fetch_locomo() -> None:
    download(LOCOMO_URL, DATA_DIR / "locomo" / "locomo10.json")


def count_lines(path, opener=open) -> int:
    with opener(path, "rt") as handle:
        return sum(1 for _ in handle)


def fetch_data() -> dict:
    config = load_config()
    manifest = {}
    for name, spec in config.datasets.items():
        if spec.source == "beir":
            fetch_beir(name)
            manifest[name] = {
                "test_queries": len(load_beir(name).queries),
                "documents": count_lines(DATA_DIR / name / "corpus.jsonl"),
            }
        else:
            fetch_trec(spec.year)
            loaded = load_trec(name)
            manifest[name] = {"judged_queries": len(loaded.queries), "pooled_passages": len(loaded.corpus)}
    fetch_locomo()
    fetch_wordnet()
    manifest["locomo"] = {"conversations": len(json.loads((DATA_DIR / "locomo" / "locomo10.json").read_text()))}
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["models", "data", "all"], nargs="?", default="all")
    args = parser.parse_args()
    if args.target in ("data", "all"):
        print(json.dumps(fetch_data(), indent=2), flush=True)
    if args.target in ("models", "all"):
        fetch_models()


if __name__ == "__main__":
    main()
