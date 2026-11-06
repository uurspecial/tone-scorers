"""Export four seed-3407 adapters without changing experimental artifacts."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
from pathlib import Path

from artifacts import ROOT, sha256

NAMES = {
    "hfl_roberta_base": "roberta-base",
    "hfl_roberta_large": "roberta-large",
    "meta_llama_8b_lora": "llama-8b-lora",
    "meta_llama_8b_4bit_lora": "llama-8b-4bit-lora",
}


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(experiment: Path, output: Path = ROOT):
    config = json.loads((experiment / "config.json").read_text())
    manifest = {
        "schema_version": 1,
        "release_tag": "v1.0.0",
        "source_experiment": "06_workbook3_model_comparison",
        "selection": "Full 496-train holdout model, seed 3407; identical selection for all four scorers",
        "labels": ["溫和", "中性", "不滿", "酸"],
        "max_length": 512,
        "training_packages": json.loads((experiment / "environment.json").read_text())["packages"],
        "models": [],
    }
    dist = output / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    for model in config["models"]:
        name = NAMES[model["key"]]
        source = experiment / "models" / model["key"] / "holdout"
        metrics = json.loads((source / "metrics.json").read_text())
        if metrics["seed"] != 3407 or metrics["train_rows"] != 496 or metrics["eval_rows"] != 496:
            raise ValueError("Unexpected source training protocol")
        directory = output / "adapters" / name
        directory.mkdir(parents=True, exist_ok=True)
        source_adapter = source / "adapter"
        for filename in ["adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
            shutil.copy2(source_adapter / filename, directory / filename)
        portable = json.loads((source_adapter / "adapter_config.json").read_text())
        portable["base_model_name_or_path"] = model["model_id"]
        portable["revision"] = model["revision"]
        portable["modules_to_save"] = list(dict.fromkeys(model["modules_to_save"]))
        write_json(directory / "adapter_config.json", portable)
        record = {
            "id": name,
            "source_key": model["key"],
            "base_model": model["model_id"],
            "base_revision": model["revision"],
            "family": model["family"],
            "load_in_4bit": model["load_in_4bit"],
            "seed": metrics["seed"],
            "train_rows": metrics["train_rows"],
            "test_rows": metrics["eval_rows"],
            "global_step": metrics["global_step"],
            "experiment_config_hash": metrics["config_hash"],
            "source_weights_sha256": sha256(source_adapter / "adapter_model.safetensors"),
            "holdout_accuracy": metrics["accuracy"],
            "holdout_macro_f1": metrics["macro_f1"],
            "precision": "NF4 double quantization + BF16 compute + LoRA" if model["load_in_4bit"] else "BF16 + LoRA",
            "base_license": "Llama 3.1 Community License" if model["family"] == "llama" else "Apache-2.0",
            "base_model_url": "https://huggingface.co/" + model["model_id"],
        }
        write_json(directory / "artifact.json", record)
        (directory / "README.md").write_text(
            f"# {name}\n\nChinese tone classification: 溫和 / 中性 / 不滿 / 酸.\n\n"
            f"Base: [{model['model_id']}]({record['base_model_url']}); revision `{model['revision']}`.\n\n"
            "LoRA + trained four-class linear head. Base weights are not included.\n\n"
            "Full 496-train model, seed 3407, 3 epochs, 372 updates. This is one trained model; "
            "it is not a fold ensemble or an average of five trained models.\n\n"
            f"Base license: {record['base_license']}. Refer to the base-model repository for terms.\n\n"
            "The adapter's base path, revision and saved-head module list were normalized for portability. "
            "Weight bytes are identical to the experimental artifact.\n", encoding="utf-8"
        )
        files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(directory.iterdir()) if p.is_file()}
        archive = dist / f"{name}.tar.gz"
        with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as handle:
                for filename in files:
                    info = handle.gettarinfo(directory / filename, arcname=filename)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with (directory / filename).open("rb") as source_file:
                        handle.addfile(info, source_file)
        record.update(files=files, archive=archive.name, archive_bytes=archive.stat().st_size, archive_sha256=sha256(archive))
        manifest["models"].append(record)
        print(f"Prepared {name}: {archive.stat().st_size / 1024**2:.1f} MiB", flush=True)
    write_json(output / "models.json", manifest)
    (dist / "SHA256SUMS").write_text("".join(f"{m['archive_sha256']}  {m['archive']}\n" for m in manifest["models"]))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT.parents[1] / "experiments/06_workbook3_model_comparison")
    args = parser.parse_args()
    prepare(args.experiment.resolve())
