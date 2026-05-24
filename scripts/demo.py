"""CLI demo: single defect diagnosis call.

Example:
    python scripts/demo.py \
        --image ./data/mvtec/bottle/test/broken_large/000.png \
        --question "What defect is this and what's the SOP?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import DefectDiagnosisRAG  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", type=str,
                        default="Identify the defect and recommend next steps.")
    parser.add_argument("--top_k", type=int, default=2)
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"image not found: {args.image}")

    rag = DefectDiagnosisRAG()
    result = rag.diagnose(args.image, args.question, top_k=args.top_k)

    print("=" * 70)
    print("RETRIEVED CASES")
    print("=" * 70)
    for i, case in enumerate(result.retrieved_cases, 1):
        print(f"[{i}] score={case['score']:.3f}  "
              f"category={case.get('category')}  "
              f"defect={case.get('defect_type')}")
        print(f"    SOP: {case.get('sop')}")
    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(result.answer)


if __name__ == "__main__":
    main()
