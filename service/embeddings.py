from dataclasses import dataclass
from typing import Any, List, Tuple

import torch

from .config import settings


@dataclass(frozen=True)
class InputItem:
    text: str | None = None
    tokens: List[int] | None = None


def normalize_input(raw_input: Any) -> List[InputItem]:
    if isinstance(raw_input, str):
        if raw_input == "":
            raise ValueError("input must not be empty")
        return [InputItem(text=raw_input)]
    if isinstance(raw_input, list):
        if not raw_input:
            return []
        if all(isinstance(item, str) for item in raw_input):
            if any(item == "" for item in raw_input):
                raise ValueError("input must not be empty")
            return [InputItem(text=item) for item in raw_input]
        if all(isinstance(item, int) for item in raw_input):
            if not raw_input:
                raise ValueError("input token array must not be empty")
            return [InputItem(tokens=raw_input)]
        if all(isinstance(item, list) for item in raw_input):
            items = []
            for entry in raw_input:
                if not all(isinstance(token, int) for token in entry):
                    raise ValueError("input token arrays must contain integers only")
                if not entry:
                    raise ValueError("input token array must not be empty")
                items.append(InputItem(tokens=entry))
            return items
    raise ValueError(
        "input must be a string, list of strings, list of integers, or list of token arrays"
    )


@torch.inference_mode()
def embed_texts(
    inputs: List[InputItem],
    tokenizer,
    model,
    device,
) -> Tuple[List[List[float]], List[int]]:
    encoded_items = []
    token_counts = []

    max_length = settings.max_input_tokens
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_length, int) and 0 < model_max_length < 1_000_000:
        max_length = min(max_length, model_max_length)
    truncate_inputs = settings.truncate_long_inputs

    for item in inputs:
        if item.text is not None:
            encoded = tokenizer(
                item.text,
                add_special_tokens=True,
                truncation=truncate_inputs,
                max_length=max_length,
                return_attention_mask=True,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            if not truncate_inputs and len(input_ids) > max_length:
                raise ValueError(
                    f"input exceeds max token limit ({max_length}). "
                    "Set EMBEDDINGS_TRUNCATE_INPUTS=true to truncate."
                )
        elif item.tokens is not None:
            input_ids = item.tokens
            if not input_ids:
                raise ValueError("input token array must not be empty")
            if len(input_ids) > max_length:
                if truncate_inputs:
                    input_ids = input_ids[:max_length]
                else:
                    raise ValueError(
                        f"input exceeds max token limit ({max_length}). "
                        "Set EMBEDDINGS_TRUNCATE_INPUTS=true to truncate."
                    )
            attention_mask = [1] * len(input_ids)
        else:
            raise ValueError("input item must include text or tokens")

        token_counts.append(len(attention_mask))
        encoded_items.append((input_ids, attention_mask))

    if sum(token_counts) > settings.max_total_tokens:
        raise ValueError(
            f"total input tokens exceed limit ({settings.max_total_tokens})"
        )

    if not encoded_items:
        return [], []

    max_seq_len = max(len(item[0]) for item in encoded_items)
    if max_seq_len == 0:
        raise ValueError("input token arrays must not be empty")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    input_ids = []
    attention_masks = []
    for ids, mask in encoded_items:
        pad_len = max_seq_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_masks.append(mask + [0] * pad_len)

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
    hidden = outputs.last_hidden_state

    mask_expanded = attention_mask_tensor.unsqueeze(-1).expand(hidden.size()).float()
    summed = torch.sum(hidden * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    embeddings = summed / counts

    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().tolist(), token_counts
