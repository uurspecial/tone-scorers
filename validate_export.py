"""Compare exported adapters against original saved holdout predictions."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from artifacts import ROOT, read_manifest, sha256
from runtime import ScorerRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    manifest = read_manifest()
    rows = [json.loads(line) for line in (args.experiment / "data/test.jsonl").read_text().splitlines() if line.strip()]
    registry = ScorerRegistry()
    report = {"complete": False, "manifest_sha256": sha256(ROOT / "models.json"), "device": torch.cuda.get_device_name(),
              "evaluated_rows_per_model": len(rows), "models": []}
    try:
        for descriptor in manifest["models"]:
            if torch.cuda.mem_get_info()[0] < (23 if descriptor["id"] == "llama-8b-lora" else 12 if descriptor["load_in_4bit"] else 4) * 1024**3:
                raise RuntimeError("Insufficient free GPU memory for validation; preserve existing processes")
            start = time.monotonic()
            registry._load(descriptor["id"])
            predicted = []
            for offset in range(0, len(rows), args.batch_size):
                batch = [row["text"] for row in rows[offset:offset + args.batch_size]]
                inputs = registry._tokenizer(batch, truncation=True, max_length=512, padding=True,
                                             pad_to_multiple_of=8, return_tensors="pt")
                inputs = {name: value.to(registry._model.device) for name, value in inputs.items()}
                predicted.append(registry._forward_logits(inputs).numpy())
            actual = np.concatenate(predicted)
            original = np.load(args.experiment / "models" / descriptor["source_key"] / "holdout/logits.npy")
            gold = np.array([row["label"] - 1 for row in rows])
            matches = int(np.sum(actual.argmax(1) == original.argmax(1)))
            accuracy = float(np.mean(actual.argmax(1) == gold))
            if matches != len(rows) or accuracy != descriptor["holdout_accuracy"]:
                raise AssertionError(f"Predictions differ after export: {descriptor['id']} ({matches}/{len(rows)})")
            # Also exercise the actual public runtime's one-text path.
            single = registry.predict(descriptor["id"], rows[0]["text"])
            if single["label_id"] != int(original[0].argmax()) + 1:
                raise AssertionError("Single-text runtime prediction differs")
            exported_weight = registry.adapter_root / descriptor["id"] / "adapter_model.safetensors"
            if sha256(exported_weight) != descriptor["source_weights_sha256"]:
                raise AssertionError("Export changed weight bytes")
            item = {"model": descriptor["id"], "matching_predictions": matches, "evaluated_rows": len(rows),
                    "accuracy": accuracy, "max_absolute_logit_difference": float(np.max(np.abs(actual - original))),
                    "single_text_prediction_verified": True, "weights_identical": True,
                    "runtime_seconds": round(time.monotonic() - start, 2)}
            report["models"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
            registry.close()
        report["complete"] = True
    finally:
        registry.close()
        (ROOT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
