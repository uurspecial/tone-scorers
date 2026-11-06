"""Load one sequence classifier at a time, preserving training tokenization."""
from __future__ import annotations

import gc
import os
import threading
from pathlib import Path

from artifacts import ROOT, read_manifest, verify_adapter


class ScorerRegistry:
    def __init__(self, manifest_path: Path | None = None, adapter_root: Path | None = None):
        self.manifest = read_manifest(manifest_path)
        self.descriptors = {item["id"]: item for item in self.manifest["models"]}
        self.adapter_root = adapter_root or Path(os.environ.get("SCORER_ADAPTER_DIR", ROOT / "adapters"))
        self.loaded_model = None
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _unload(self):
        self._model = None
        self._tokenizer = None
        self.loaded_model = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self, model_id: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

        descriptor = self.descriptors[model_id]
        directory = self.adapter_root / model_id
        verify_adapter(directory, descriptor)
        requested = os.environ.get("SCORER_DEVICE", "auto")
        if requested not in {"auto", "cuda", "cpu"}:
            raise ValueError("SCORER_DEVICE must be auto, cuda or cpu")
        use_cuda = requested != "cpu" and torch.cuda.is_available()
        if requested == "cuda" and not use_cuda:
            raise RuntimeError("CUDA was requested but is unavailable")
        if descriptor["family"] == "llama" and not use_cuda:
            raise RuntimeError("Llama scorers require a CUDA GPU")
        if use_cuda and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The experiment requires a BF16-capable CUDA GPU")
        dtype = torch.bfloat16 if use_cuda else torch.float32
        labels = self.manifest["labels"]
        kwargs = {
            "revision": descriptor["base_revision"],
            "num_labels": len(labels),
            "id2label": {i: label for i, label in enumerate(labels)},
            "label2id": {label: i for i, label in enumerate(labels)},
            "dtype": dtype,
            "trust_remote_code": False,
        }
        if descriptor["load_in_4bit"]:
            kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                device_map={"": 0},
            )
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForSequenceClassification.from_pretrained(descriptor["base_model"], **kwargs)
        base.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, directory, is_trainable=False)
        if not descriptor["load_in_4bit"]:
            model.to("cuda" if use_cuda else "cpu")
        model.config.use_cache = False
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self.loaded_model = model_id

    def _forward_logits(self, inputs):
        import torch

        # Trainer/Accelerate used BF16 autocast while LoRA weights stayed FP32.
        # Preserve that mixed-precision operation order for inference as well.
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self._model.device.type == "cuda"
        ):
            return self._model(**inputs).logits.float().cpu()

    def predict(self, model_id: str, text: str) -> dict:
        if model_id not in self.descriptors:
            raise KeyError(model_id)
        # Serialize loading and inference, including switching between model families.
        with self._lock:
            try:
                if self.loaded_model != model_id:
                    self._unload()
                    self._load(model_id)
                import torch

                inputs = self._tokenizer(
                    text,
                    truncation=True,
                    max_length=self.manifest["max_length"],
                    padding=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                tokens = int(inputs["attention_mask"].sum().item())
                inputs = {name: value.to(self._model.device) for name, value in inputs.items()}
                logits = self._forward_logits(inputs)[0]
                probabilities = logits.softmax(dim=-1).tolist()
                index = int(logits.argmax().item())
                return {
                    "model": model_id,
                    "label_id": index + 1,
                    "label": self.manifest["labels"][index],
                    "probabilities": dict(zip(self.manifest["labels"], probabilities)),
                    "logits": logits.tolist(),
                    "input_tokens": tokens,
                }
            except Exception:
                self._unload()
                raise

    def close(self):
        with self._lock:
            self._unload()
