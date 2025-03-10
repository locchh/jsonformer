"""
This module provides custom logits processors and stopping criteria for controlling
the text generation process in language models, specifically for JSON generation.
It includes specialized handlers for string and number generation to ensure valid JSON output.
"""

from typing import List
from transformers import PreTrainedTokenizer, LogitsWarper, StoppingCriteria
import torch


class StringStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria for string generation in JSON.
    Stops generation when a closing quote (") is encountered to ensure valid string values.
    
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for decoding tokens
        prompt_length (int): Length of the input prompt to properly track generated content
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        _,
    ) -> bool:
        # Don't stop if we're still in the prompt
        if len(input_ids[0]) <= self.prompt_length:
            return False

        # Check if the last generated token contains a quote
        last_token_id = input_ids[0][-1]
        last_token = self.tokenizer.decode(last_token_id, skip_special_tokens=True)

        result = '"' in last_token

        return result


class NumberStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria for number generation in JSON.
    Controls the precision of decimal numbers and ensures valid number formatting.
    
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for decoding tokens
        prompt_length (int): Length of the input prompt to properly track generated content
        precision (int): Maximum number of decimal places allowed (default: 3)
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_length: int,
        precision: int = 3,
    ):
        self.tokenizer = tokenizer
        self.precision = precision
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> bool:
        # Decode only the generated part (excluding prompt)
        decoded = self.tokenizer.decode(
            input_ids[0][self.prompt_length :], skip_special_tokens=True
        )

        # Stop if there are multiple decimal points (invalid number)
        if decoded.count(".") > 1:
            return True

        # Stop if decimal places exceed specified precision
        if (
            decoded.count(".") == 1
            and len(decoded.strip().split(".")[1]) > self.precision
        ):
            return True

        # Stop if we have at least one digit and hit a space or newline
        if (
            len(decoded) > 1
            and any(c.isdigit() for c in decoded)
            and decoded[-1] in [" ", "\n"]
        ):
            return True

        return False


class OutputNumbersTokens(LogitsWarper):
    """
    Logits processor that restricts token generation to only valid number characters.
    This ensures that only digits and decimal points can be generated for number values.
    
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding/decoding tokens
        prompt (str): The input prompt to initialize the processor
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt: str):
        self.tokenizer = tokenizer
        self.tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        vocab_size = len(tokenizer)
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)

        # Create a mask of allowed tokens (empty string, digits, and single decimal point)
        for _, token_id in tokenizer.get_vocab().items():
            token_str = tokenizer.decode(token_id).strip()

            if token_str == "" or (
                all(c.isdigit() or c == "." for c in token_str)
                and token_str.count(".") <= 1
            ):
                self.allowed_mask[token_id] = True

    def __call__(self, _, scores):
        # Apply mask to scores, setting disallowed tokens to negative infinity
        mask = self.allowed_mask.expand_as(scores)
        scores[~mask] = -float("inf")

        return scores
