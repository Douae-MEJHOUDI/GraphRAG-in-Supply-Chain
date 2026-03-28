from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate extracted triples against edge-case gold triples."
    )
    parser.add_argument(
        "--gold",
        default="data/eval/edge_case_gold_triples.jsonl",
        help="Gold triples file (.jsonl).",
    )
    parser.add_argument(
        "--pred",
        default="data/processed/edge_case_triples_llama3.json",
        help="Predicted triples file (.json or .jsonl).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/edge_case_eval_metrics.json",
        help="Where to save metrics JSON.",
    )
    return parser.parse_args()


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;:.")


def _norm_relation(value: Any) -> str:
    rel = _norm_text(value)
    return rel.replace("-", "_").replace(" ", "_")


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _triple_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _norm_text(item.get("source")),
        _norm_text(item.get("head")),
        _norm_relation(item.get("relation")),
        _norm_text(item.get("tail")),
    )


def _pair_key(source: str, relation: str, head: str, tail: str) -> tuple[str, str, str, str]:
    a, b = sorted([head, tail])
    return source, relation, a, b


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)
    pred_path = Path(args.pred)
    out_path = Path(args.output)

    gold_items = _load_json_or_jsonl(gold_path)
    pred_items = _load_json_or_jsonl(pred_path)

    gold_keys = {_triple_key(x) for x in gold_items}
    pred_keys = {_triple_key(x) for x in pred_items}

    tp = gold_keys & pred_keys
    fp = pred_keys - gold_keys
    fn = gold_keys - pred_keys

    precision = _safe_div(len(tp), len(pred_keys))
    recall = _safe_div(len(tp), len(gold_keys))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    gold_by_pair: dict[tuple[str, str, str, str], set[tuple[str, str, str, str]]] = {}
    for k in gold_keys:
        src, head, rel, tail = k
        pk = _pair_key(src, rel, head, tail)
        gold_by_pair.setdefault(pk, set()).add(k)

    reverse_errors = 0
    for k in fp:
        src, head, rel, tail = k
        pk = _pair_key(src, rel, head, tail)
        if pk in gold_by_pair:
            reverse_errors += 1

    direction_accuracy = _safe_div(len(tp), len(tp) + reverse_errors)

    gold_loc = {k for k in gold_keys if k[2] == "located_in"}
    pred_loc = {k for k in pred_keys if k[2] == "located_in"}
    loc_recall = _safe_div(len(gold_loc & pred_loc), len(gold_loc))

    gold_aff = {k for k in gold_keys if k[2] == "affected_by"}
    pred_aff = {k for k in pred_keys if k[2] == "affected_by"}
    affected_by_recall = _safe_div(len(gold_aff & pred_aff), len(gold_aff))

    gold_types = {
        _triple_key(x): (_norm_text(x.get("head_type")), _norm_text(x.get("tail_type")))
        for x in gold_items
    }
    pred_types = {
        _triple_key(x): (_norm_text(x.get("head_type")), _norm_text(x.get("tail_type")))
        for x in pred_items
    }
    type_matches = 0
    for k in tp:
        g_types = gold_types.get(k, ("", ""))
        p_types = pred_types.get(k, ("", ""))
        if g_types == p_types:
            type_matches += 1
    type_accuracy_on_tp = _safe_div(type_matches, len(tp))

    evidence_non_empty = 0
    confidence_valid = 0
    for item in pred_items:
        if _norm_text(item.get("evidence")):
            evidence_non_empty += 1
        try:
            conf = float(item.get("confidence"))
            if 0.0 <= conf <= 1.0:
                confidence_valid += 1
        except Exception:
            pass

    evidence_coverage = _safe_div(evidence_non_empty, len(pred_items))
    confidence_valid_ratio = _safe_div(confidence_valid, len(pred_items))

    sources = sorted({_norm_text(x.get("source")) for x in gold_items})
    per_case: list[dict[str, Any]] = []
    for source in sources:
        g = {k for k in gold_keys if k[0] == source}
        p = {k for k in pred_keys if k[0] == source}
        tp_case = g & p
        rec = _safe_div(len(tp_case), len(g))
        prec = _safe_div(len(tp_case), len(p))
        per_case.append(
            {
                "source": source,
                "gold": len(g),
                "pred": len(p),
                "tp": len(tp_case),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }
        )

    metrics = {
        "counts": {
            "gold": len(gold_keys),
            "pred": len(pred_keys),
            "tp": len(tp),
            "fp": len(fp),
            "fn": len(fn),
        },
        "core_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "critical_checks": {
            "direction_accuracy": round(direction_accuracy, 4),
            "reverse_direction_errors": reverse_errors,
            "located_in_recall": round(loc_recall, 4),
            "affected_by_recall": round(affected_by_recall, 4),
            "invented_edges": len(fp),
        },
        "quality_checks": {
            "type_accuracy_on_true_positives": round(type_accuracy_on_tp, 4),
            "evidence_non_empty_ratio": round(evidence_coverage, 4),
            "confidence_valid_ratio": round(confidence_valid_ratio, 4),
        },
        "per_case": per_case,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[eval] saved ->", out_path)
    print(json.dumps(metrics["counts"], indent=2))
    print(json.dumps(metrics["core_metrics"], indent=2))
    print(json.dumps(metrics["critical_checks"], indent=2))


if __name__ == "__main__":
    main()
