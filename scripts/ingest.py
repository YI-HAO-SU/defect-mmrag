"""Build the Qdrant case database from MVTec AD (or any folder of labeled defects).

Expected layout (matches MVTec AD):

  data_dir/
    bottle/
      test/
        broken_large/
          000.png
          001.png
          ...
        broken_small/
          ...
    cable/
      test/
        ...

Each "defect type" folder becomes the label. We also embed a few "good" samples
so the retriever can return "this looks normal" cases.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.embedding import embed_image  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


# Hand-written SOPs for the demo. In production these come from MES / SOP DB.
DEFAULT_SOPS = {
    "broken_large": "Quarantine the lot; root-cause via stress test on handling robots.",
    "broken_small": "Sample 30 units from the same lot; check loader chuck pressure.",
    "contamination": "Trigger cleanroom particle audit; flush the affected tool.",
    "crack": "Stop the tool; inspect deposition chamber and inspect part carrier.",
    "cut": "Inspect blade/wire condition on the upstream tool; replace if necessary.",
    "hole": "Check etch recipe and gas flow stability; trigger DOE if recurring.",
    "scratch": "Inspect handling robot end-effector; clean and recalibrate.",
    "good": "No action required.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest MVTec-style defect images into Qdrant")
    p.add_argument("--data_dir", type=Path, required=True,
                   help="Root folder containing per-category subfolders.")
    p.add_argument("--collection", type=str, default=settings.qdrant_collection)
    p.add_argument("--per_class_limit", type=int, default=10,
                   help="Cap images per defect class to keep ingestion fast.")
    p.add_argument("--split", type=str, default="test",
                   help="Which split folder to walk inside each category.")
    return p.parse_args()


def discover_images(
    data_dir: Path, split: str, per_class_limit: int
) -> list[tuple[Path, str, str]]:
    """Return list of (image_path, category, defect_type) triples."""
    triples: list[tuple[Path, str, str]] = []
    for category_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        split_dir = category_dir / split
        if not split_dir.is_dir():
            continue
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            imgs = sorted(defect_dir.glob("*.png")) + sorted(defect_dir.glob("*.jpg"))
            for img in imgs[:per_class_limit]:
                triples.append((img, category_dir.name, defect_dir.name))
    return triples


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"data_dir {args.data_dir} does not exist")

    vs = VectorStore(collection=args.collection)
    print(f"[ingest] (re)creating collection '{args.collection}'")
    vs.reset_collection()

    triples = discover_images(args.data_dir, args.split, args.per_class_limit)
    if not triples:
        raise SystemExit(f"no images found under {args.data_dir}/*/{args.split}/*")
    print(f"[ingest] found {len(triples)} images across "
          f"{len({t[1] for t in triples})} categories")

    vectors: list[list[float]] = []
    payloads: list[dict] = []
    for img_path, category, defect in tqdm(triples, desc="embedding"):
        vectors.append(embed_image(img_path))
        payloads.append({
            "image_path": str(img_path.resolve()),
            "category": category,
            "defect_type": defect,
            "description": f"{category} with defect '{defect}'",
            "sop": DEFAULT_SOPS.get(defect, "Escalate to process engineer for review."),
        })

    vs.upsert_batch(vectors, payloads)
    print(f"[ingest] upserted {len(vectors)} cases into '{args.collection}'")


if __name__ == "__main__":
    main()
