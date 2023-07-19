from typing import Union
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
NUM_SENTINEL_TOKENS: int = 100

def adapt_tokenizer_for_denoising(tokenizer: Tokenizer):
    """Adds sentinel tokens and padding token (if missing).

    Expands the tokenizer vocabulary to include sentinel tokens
    used in mixture-of-denoiser tasks as well as a padding token.

    All added tokens are added as special tokens. No tokens are
    added if sentinel tokens and padding token already exist.
    """
    sentinels_to_add = [f'<extra_id_{i}>' for i in range(NUM_SENTINEL_TOKENS)]
    tokenizer.add_tokens(sentinels_to_add, special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.add_tokens('<pad>', special_tokens=True)
        tokenizer.pad_token = '<pad>'
        assert tokenizer.pad_token_id is not None
    sentinels = ''.join([f'<extra_id_{i}>' for i in range(NUM_SENTINEL_TOKENS)])
    _sentinel_token_ids = tokenizer(sentinels, add_special_tokens=False).input_ids
    tokenizer.sentinel_token_ids = _sentinel_token_ids

class AutoTokenizerForMOD(AutoTokenizer):
    """AutoTokenizer + Adaptation for MOD.

    A simple wrapper around AutoTokenizer to make instantiating
    an MOD-adapted tokenizer a bit easier.

    MOD-adapted tokenizers have sentinel tokens (e.g., <extra_id_0>),
    a padding token, and a property to get the token ids of the
    sentinel tokens.
    """

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """See `AutoTokenizer.from_pretrained` docstring."""
        tokenizer = super().from_pretrained(*args, **kwargs)
        adapt_tokenizer_for_denoising(tokenizer)
        return tokenizer