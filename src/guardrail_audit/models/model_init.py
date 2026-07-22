"""Llama-Guard-3 loading, quantization, and batched decision + embedding (Days 2-4).

Llama-Guard-3 is a generative moderation model: given a chat, it generates ``safe`` or
``unsafe`` on the first line, and (when unsafe) a comma-separated list of violated
category codes (S1..S13) on the second line. We parse that generation for the decision
and simultaneously capture the terminal-token hidden state for clustering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _dtype_and_quant(dtype: str) -> tuple[torch.dtype | None, dict]:
    """Map a config dtype string to torch_dtype + quantization kwargs."""
    dtype = dtype.lower()
    if dtype in {"int8", "int4"}:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - env dependent
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


@dataclass
class GuardDecision:
    is_unsafe: bool
    categories: list[str]      # e.g. ["S1", "S9"]; empty when safe
    raw: str                   # raw generated text


class LlamaGuard:
    """Wraps Llama-Guard-3 for batched moderation decisions + hidden-state capture."""

    def __init__(
        self,
        name: str,
        dtype: str = "float16",
        device_map: str = "auto",
        max_new_tokens: int = 20,
        hidden_layer: int = -1,
    ) -> None:
        torch_dtype, extra = _dtype_and_quant(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding so the terminal (last) token is at index -1 for every row.
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
        self, chats: list[list[dict[str, str]]]
    ) -> tuple[list[GuardDecision], torch.Tensor]:
        """Return per-chat decisions and an (B, d) tensor of terminal hidden states.

        The hidden state is taken from the last *prompt* token (before generation),
        matching the spec's terminal-token extraction at the final layer.
        """
        prompt_ids = self.tokenizer.apply_chat_template(
            chats,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(self.device)
        attention_mask = (prompt_ids != self.tokenizer.pad_token_id).long()

        # One forward pass gives us hidden states for the prompt (terminal token = -1).
        forward = self.model(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Left padding => terminal real token is column -1 for all rows.
        embeddings = forward.hidden_states[self.hidden_layer][:, -1, :].float().cpu()

        # Generate the decision text.
        generated = self.model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = generated[:, prompt_ids.shape[1] :]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        decisions = [self._parse(t) for t in texts]
        return decisions, embeddings


def load_guard(model_cfg) -> LlamaGuard:
    """Construct a LlamaGuard from a config.model section."""
    return LlamaGuard(
        name=model_cfg.name,
        dtype=model_cfg.dtype,
        device_map=model_cfg.device_map,
        max_new_tokens=model_cfg.max_new_tokens,
        hidden_layer=model_cfg.hidden_layer,
    )
