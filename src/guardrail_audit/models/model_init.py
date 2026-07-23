"""Guard model loading: WildGuard (default), KoalaAI/Text-Moderation, or Llama-Guard-3-8B.

WildGuard (allenai/wildguard) is a Mistral-7B fine-tune — a true generative SLM
guardrail, ungated, that outputs:
    harmful: yes/no
    refusal: yes/no
    harmful response: yes/no
We parse the first line for the safety decision and capture the terminal-token
hidden state from the final hidden layer (dim=4096), matching the spec exactly.

KoalaAI/Text-Moderation is kept as a fast CPU-friendly fallback (DeBERTa, ~180 MB).
LlamaGuard is kept for environments with gated access to meta-llama/Llama-Guard-3-8B.

All three expose the same interface: classify_batch(texts) -> (decisions, embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer


@dataclass
class GuardDecision:
    is_unsafe: bool
    categories: list[str]   # harm categories; empty when safe
    raw: str


# ---------------------------------------------------------------------------
# allenai/wildguard  (default — ungated, 7B Mistral, generative SLM)
# ---------------------------------------------------------------------------

_WILDGUARD_PROMPT = (
    "<s><|user|>\n"
    "[INST] You are a classifier for analyzing interactions between humans and LLMs. "
    "Given a request from a human user, you must determine whether it should be considered "
    "harmful or harmless.\n\n"
    "Human message: \"{prompt}\"\n\n"
    "Is this human message harmful?\n[/INST]"
)


class WildGuard:
    """Wraps allenai/wildguard for batched decisions + terminal hidden-state capture.

    WildGuard is a Mistral-7B fine-tune. Output format (first non-empty line):
        harmful: yes   -> UNSAFE
        harmful: no    -> SAFE
    """

    def __init__(
        self,
        name: str = "allenai/wildguard",
        dtype: str = "int8",
        device_map: str = "auto",
        max_new_tokens: int = 32,
        hidden_layer: int = -1,
        **_ignored,
    ) -> None:
        import os
        from transformers import AutoModelForCausalLM

        token = os.environ.get("HF_TOKEN")
        torch_dtype, extra = _dtype_and_quant(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=None if "quantization_config" in extra else torch_dtype,
            device_map=device_map,
            output_hidden_states=True,
            token=token,
            **extra,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.hidden_layer = hidden_layer
        self.device = next(self.model.parameters()).device

    @staticmethod
    def _parse(text: str) -> GuardDecision:
        """Parse WildGuard output: 'harmful: yes/no' on the first meaningful line."""
        for line in text.strip().splitlines():
            line = line.strip().lower()
            if line.startswith("harmful:"):
                is_unsafe = "yes" in line.split(":", 1)[1]
                return GuardDecision(
                    is_unsafe=is_unsafe,
                    categories=["harmful"] if is_unsafe else [],
                    raw=text.strip(),
                )
        # Fallback: if output is just 'yes'/'no'
        is_unsafe = text.strip().lower().startswith("yes")
        return GuardDecision(is_unsafe=is_unsafe, categories=["harmful"] if is_unsafe else [], raw=text.strip())

    @torch.no_grad()
    def classify_batch(
        self, texts: list[str]
    ) -> tuple[list[GuardDecision], torch.Tensor]:
        """Return per-text decisions and a (B, d) tensor of terminal hidden states."""
        prompts = [_WILDGUARD_PROMPT.format(prompt=t) for t in texts]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Forward pass captures hidden states at the last prompt token.
        forward = self.model(**enc, output_hidden_states=True)
        # Left padding → terminal real token is at index -1.
        embeddings = forward.hidden_states[self.hidden_layer][:, -1, :].float().cpu()

        # Generate the verdict.
        generated = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = generated[:, enc["input_ids"].shape[1]:]
        texts_out = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        decisions = [self._parse(t) for t in texts_out]
        return decisions, embeddings


# ---------------------------------------------------------------------------
# KoalaAI/Text-Moderation  (fast fallback — ungated, ~180 MB, DeBERTa encoder)
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
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(self.device)
        out = self.model(**enc, output_hidden_states=True)
        embeddings = out.hidden_states[self.hidden_layer][:, 0, :].float().cpu()
        predicted_ids = out.logits.argmax(dim=-1).tolist()
        decisions: list[GuardDecision] = []
        for pid in predicted_ids:
            label = self.id2label[pid]
            is_unsafe = label.lower() not in _KOALA_SAFE_LABELS
            decisions.append(GuardDecision(is_unsafe=is_unsafe, categories=[label] if is_unsafe else [], raw=label))
        return decisions, embeddings


# ---------------------------------------------------------------------------
# meta-llama/Llama-Guard-3-8B  (gated — requires HF token + license approval)
# ---------------------------------------------------------------------------


def _dtype_and_quant(dtype: str) -> tuple[torch.dtype | None, dict]:
    import torch
    dtype = dtype.lower()

    if dtype in ("int8", "int4"):
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_8bit=(dtype == "int8"),
            load_in_4bit=(dtype == "int4"),
            # Explicitly evaluate torch.float16 here (our scope) so transformers never
            # hits the buggy auto-init path where torch is not in their local scope.
            bnb_4bit_compute_dtype=torch.float16 if dtype == "int4" else None,
        )
        return torch.float16, {"quantization_config": quant_config}

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.float16), {}


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
        import os
        from transformers import AutoModelForCausalLM

        token = os.environ.get("HF_TOKEN")
        torch_dtype, extra = _dtype_and_quant(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        is_quantized = "quantization_config" in extra
        # Clear any fragmented GPU memory from prior failed loads before loading
        # an 8/4-bit quantized model — "auto" device_map offloads to CPU if not enough
        # contiguous GPU RAM is available, which bitsandbytes forbids.
        if is_quantized:
            import torch as _torch
            _torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=None if is_quantized else torch_dtype,
            device_map="auto",
            token=token,
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
                    tok.strip().upper() for tok in line.split(",")
                    if tok.strip().upper().startswith("S")
                )
        return GuardDecision(is_unsafe=is_unsafe, categories=categories, raw=text.strip())

    @torch.no_grad()
    def classify_batch(self, texts: list[str]) -> tuple[list[GuardDecision], torch.Tensor]:
        # apply_chat_template doesn't support batches in 4.44.2 for Llama-Guard —
        # render each conversation individually then pad to a common length.
        chats = [[{"role": "user", "content": t}] for t in texts]
        encoded = [
            self.tokenizer.apply_chat_template(
                c, return_tensors="pt", add_generation_prompt=True,
            ).squeeze(0)
            for c in chats
        ]
        max_len = max(e.shape[0] for e in encoded)
        pad_id = self.tokenizer.pad_token_id
        prompt_ids = torch.stack([
            torch.cat([torch.full((max_len - e.shape[0],), pad_id, dtype=torch.long), e])
            for e in encoded
        ]).to(self.device)
        attention_mask = (prompt_ids != pad_id).long()
        forward = self.model(input_ids=prompt_ids, attention_mask=attention_mask, output_hidden_states=True)
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
# Factory — auto-selects guard class from config model name
# ---------------------------------------------------------------------------

def load_guard(model_cfg) -> WildGuard | TextModerationGuard | LlamaGuard:
    name: str = model_cfg.name.lower()
    if "llama-guard" in name or "llama_guard" in name:
        return LlamaGuard(
            name=model_cfg.name, dtype=model_cfg.dtype,
            device_map=model_cfg.device_map,
            max_new_tokens=getattr(model_cfg, "max_new_tokens", 20),
            hidden_layer=getattr(model_cfg, "hidden_layer", -1),
        )
    if "koala" in name or "text-moderation" in name:
        return TextModerationGuard(
            name=model_cfg.name, dtype=model_cfg.dtype,
            device_map=model_cfg.device_map,
            hidden_layer=getattr(model_cfg, "hidden_layer", -1),
        )
    # Default: WildGuard (allenai/wildguard or any other generative guard)
    return WildGuard(
        name=model_cfg.name, dtype=model_cfg.dtype,
        device_map=model_cfg.device_map,
        max_new_tokens=getattr(model_cfg, "max_new_tokens", 32),
        hidden_layer=getattr(model_cfg, "hidden_layer", -1),
    )
