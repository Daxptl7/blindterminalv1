"""Lazy, optional AI4Bharat IndicTrans2 adapter.

This module deliberately has no import-time dependency on torch, transformers,
or IndicTransToolkit.  BlindAssist must still boot and explain what is missing
when the offline translation models have not been provisioned yet.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


LANGUAGE_TAGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "gu": "guj_Gujr",
}

_MODEL_KEYS = {
    "en-indic": "indictrans2_en_indic_model",
    "indic-en": "indictrans2_indic_en_model",
    "indic-indic": "indictrans2_indic_indic_model",
}

_DEFAULT_DIRS = {
    "en-indic": "models_local/indictrans2/en-indic",
    "indic-en": "models_local/indictrans2/indic-en",
    "indic-indic": "models_local/indictrans2/indic-indic",
}


class IndicTrans2Engine:
    """Load only the model required by a requested language pair.

    Model values are local directories by default.  Hugging Face model IDs are
    accepted only when ``indictrans2_allow_download`` is explicitly enabled;
    this prevents an offline/private request from unexpectedly using the
    network or filling the Pi's SD card.
    """

    def __init__(self, settings: Optional[dict] = None,
                 base_dir: Optional[Path] = None) -> None:
        self.settings = settings or {}
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)
        self.device = str(self.settings.get("indictrans2_device", "cpu") or "cpu")
        self.allow_download = bool(
            self.settings.get("indictrans2_allow_download", False)
        )
        self.num_beams = max(1, int(self.settings.get("indictrans2_num_beams", 1)))
        self.max_new_tokens = max(
            32, int(self.settings.get("indictrans2_max_new_tokens", 512))
        )
        self._models: Dict[str, Tuple[Any, Any, Any, str]] = {}
        self._errors: Dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _bundle_for(src: str, dest: str) -> Optional[str]:
        if src == "en" and dest in ("hi", "gu"):
            return "en-indic"
        if src in ("hi", "gu") and dest == "en":
            return "indic-en"
        if src in ("hi", "gu") and dest in ("hi", "gu") and src != dest:
            return "indic-indic"
        return None

    def _configured_model(self, bundle: str) -> str:
        configured = str(self.settings.get(_MODEL_KEYS[bundle], "") or "").strip()
        return configured or _DEFAULT_DIRS[bundle]

    def _resolve_model(self, bundle: str) -> Tuple[Optional[str], Optional[str]]:
        configured = self._configured_model(bundle)
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        if path.is_dir():
            return str(path), None
        if self.allow_download and "/" in configured:
            return configured, None
        return None, (
            f"IndicTrans2 {bundle} model is not installed. Expected a local "
            f"model directory at {path}."
        )

    def _load(self, bundle: str) -> Optional[Tuple[Any, Any, Any, str]]:
        with self._lock:
            if bundle in self._models:
                return self._models[bundle]

            model_ref, error = self._resolve_model(bundle)
            if error:
                self._errors[bundle] = error
                return None

            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                try:
                    # Current toolkit release.
                    from IndicTransToolkit import IndicProcessor
                except ImportError:
                    try:
                        # Documented fallback for some binary builds.
                        from IndicTransToolkit.IndicTransToolkit import IndicProcessor
                    except ImportError:
                        # Compatibility with the older HF model-card layout.
                        from IndicTransToolkit.processor import IndicProcessor
            except Exception as exc:
                self._errors[bundle] = (
                    "IndicTrans2 runtime is unavailable. Install compatible "
                    "torch, transformers, sentencepiece and IndicTransToolkit "
                    f"packages ({exc.__class__.__name__}: {exc})."
                )
                return None

            try:
                local_only = not self.allow_download
                tokenizer = AutoTokenizer.from_pretrained(
                    model_ref,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_ref,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                model.to(self.device)
                model.eval()
                processor = IndicProcessor(inference=True)
                loaded = (model, tokenizer, processor, str(model_ref))
                self._models[bundle] = loaded
                self._errors.pop(bundle, None)
                return loaded
            except Exception as exc:
                self._errors[bundle] = (
                    f"Could not load IndicTrans2 {bundle} model from {model_ref} "
                    f"({exc.__class__.__name__}: {exc})."
                )
                return None

    def translate(self, text: str, src: str, dest: str) -> Optional[str]:
        bundle = self._bundle_for(src, dest)
        if bundle is None:
            self._errors["pair"] = f"Unsupported IndicTrans2 pair: {src}->{dest}"
            return None

        loaded = self._load(bundle)
        if loaded is None:
            return None

        model, tokenizer, processor, _model_ref = loaded
        try:
            import torch

            src_tag = LANGUAGE_TAGS[src]
            dest_tag = LANGUAGE_TAGS[dest]
            prepared = processor.preprocess_batch(
                [text], src_lang=src_tag, tgt_lang=dest_tag, visualize=False
            )
            inputs = tokenizer(
                prepared,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    num_beams=self.num_beams,
                    num_return_sequences=1,
                    max_new_tokens=self.max_new_tokens,
                )
            decoded = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            translated = processor.postprocess_batch(decoded, lang=dest_tag)
            result = translated[0].strip() if translated else ""
            return result or None
        except Exception as exc:
            self._errors[bundle] = (
                f"IndicTrans2 {bundle} inference failed "
                f"({exc.__class__.__name__}: {exc})."
            )
            raise RuntimeError(self._errors[bundle]) from exc

    def model_id(self, src: str, dest: str) -> Optional[str]:
        bundle = self._bundle_for(src, dest)
        if bundle is None:
            return None
        loaded = self._models.get(bundle)
        if loaded:
            return loaded[3]
        model_ref, _error = self._resolve_model(bundle)
        return model_ref

    def diagnostics(self) -> dict:
        bundles = {}
        for bundle in _MODEL_KEYS:
            model_ref, error = self._resolve_model(bundle)
            bundles[bundle] = {
                "configured": self._configured_model(bundle),
                "available": model_ref is not None,
                "loaded": bundle in self._models,
                "error": self._errors.get(bundle) or error,
            }
        return {
            "backend": "indictrans2",
            "device": self.device,
            "downloads_allowed": self.allow_download,
            "bundles": bundles,
        }
