"""Guard model loading: KoalaAI/Text-Moderation (default) or Llama-Guard-3-8B (gated).

KoalaAI/Text-Moderation is a DeBERTa-v3 SequenceClassification model (~180 MB,
ungated). It returns one of: OK, S, H, V, HR, SH, S3 per prompt.  We extract the
[CLS] token from the final encoder hidden state (dim=768) as the prompt embedding.

LlamaGuard is kept for environments with access to meta-llama/Llama-Guard-3-8B.
Both expose the same interface: classify_batch(texts) -> (decisions, embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer


@dataclass
class GuardDecision:
    is_unsafe: bool
    categories: list[str]   # e.g. ["S", "H"]; empty when safe/OK
    raw: str


# ---------------------------------------------------------------------------
# KoalaAI/Text-Moderation  (default — ungated, ~180 MB, DeBERTa encoder)
# ---------------------------------------------------------------------------

_KOALA_SAFE_LABELS = {"ok"}


class TextModerationGuard:
    """Wraps KoalaAI/Text-Moderation for batched decisions + CLS embedding capture."""

    def __init__(
        self,
        name: str = "KoalaAI/Text-Moderation",
        dtype: str = "float16",
        device_map: str = "auto",
        hidden_layer: int = -1,
        **_ignored,
    ) -> None:
        from transformers import AutoModelForSequenceClassification

        torch_dtype = torch.float16 if dtype in {"float16", "int8", "int4"} else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            output_hidden_states=True,
        )
        self.model.eval()
        self.hidden_layer = hidden_layer
        self.device = next(self.model.parameters()).device
        self.id2label: dict[int, str] = self.model.config.id2label

    @torch.no_grad()
    def classify_batch(
        self, texts: list[str]
    ) -> tuple[list[GuardDecision], torch.Tensor]:
        """Return per-text decisions and a (B, d) CLS embedding tensor."""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        out = self.model(**enc, output_hidden_states=True)

        # CLS token = position 0 of the last hidden state.
        embeddings = out.hidden_states[self.hidden_layer][:, 0, :].float().cpu()

        predicted_ids = out.logits.argmax(dim=-1).tolist()
        decisions: list[GuardDecision] = []
        for pid in predicted_ids:
            label = self.id2label[pid]
            is_unsafe = label.lower() not in _KOALA_SAFE_LABELS
            cats = [label] if is_unsafe else []
            decisions.append(GuardDecision(is_unsafe=is_unsafe, categories=cats, raw=label))

        return decisions, embeddings


# ---------------------------------------------------------------------------
# meta-llama/Llama-Guard-3-8B  (gated — requires HF token + license approval)
# ---------------------------------------------------------------------------


def _dtype_and_quant(dtype: str) -> tuple[torch.dtype | None, dict]:
    dtype = dtype.lower()
    if dtype in {"int8", "int4"}:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("Install extras: pip install '.[quant]'") from exc
        quant = BitsAndBytesConfig(
            load_in_8bit=(dtype == "int8"),
            load_in_4bit=(dtype == "int4"),
            bnb_4bit_compute_dtype=torch.float16,
        )
        return None, {"quantization_config": quant}
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        dtype, torch.float16
    )
    return torch_dtype, {}


class LlamaGuard:
    """Wraps Llama-Guard-3 for batched decisions + terminal hidden-state capture."""

    def __init__(
        self,
        name: str,
        dtype: str = "float16",
        device_map: str = "auto",
        max_new_tokens: int = 20,
        hidden_layer: int = -1,
    ) -> None:
        from transformers import AutoModelForCausalLM

        torch_dtype, extra = _dtype_and_quant(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            output_hidden_states=True,
            **extra,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.hidden_layer = hidden_layer
        self.device = next(self.model.parameters()).device

    @staticmethod
    def _parse(text: str) -> GuardDecision:
        lowered = text.strip().lower()
        is_unsafe = lowered.startswith("unsafe")
        categories: list[str] = []
        if is_unsafe:
            for line in text.strip().splitlines()[1:]:
                categories.extend(
                    tok.strip().upper()
                    for tok in line.split(",")
                    if tok.strip().upper().startswith("S")
                )
        return GuardDecision(is_unsafe=is_unsafe, categories=categories, raw=text.strip())

    @torch.no_grad()
    def classify_batch(
        self, texts: list[str]
    ) -> tuple[list[GuardDecision], torch.Tensor]:
        chats = [[{"role": "user", "content": t}] for t in texts]
        prompt_ids = self.tokenizer.apply_chat_template(
            chats, return_tensors="pt", padding=True, add_generation_prompt=True,
        ).to(self.device)
        attention_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
        forward = self.model(
            input_ids=prompt_ids, attention_mask=attention_mask, output_hidden_states=True,
        )
        embeddings = forward.hidden_states[self.hidden_layer][:, -1, :].float().cpu()
        generated = self.model.generate(
            input_ids=prompt_ids, attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = generated[:, prompt_ids.shape[1]:]
        texts_out = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [self._parse(t) for t in texts_out], embeddings


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LLAMA_GUARD_PREFIXES = {"meta-llama", "llama-guard"}


def load_guard(model_cfg) -> TextModerationGuard | LlamaGuard:
    """Instantiate the right guard class based on the model name in config."""
    name: str = model_cfg.name
    if any(name.lower().startswith(p) or p in name.lower() for p in _LLAMA_GUARD_PREFIXES):
        return LlamaGuard(
            name=name,
            dtype=model_cfg.dtype,
            device_map=model_cfg.device_map,
            max_new_tokens=getattr(model_cfg, "max_new_tokens", 20),
            hidden_layer=getattr(model_cfg, "hidden_layer", -1),
        )
    return TextModerationGuard(
        name=name,
        dtype=model_cfg.dtype,
        device_map=model_cfg.device_map,
        hidden_layer=getattr(model_cfg, "hidden_layer", -1),
    )
