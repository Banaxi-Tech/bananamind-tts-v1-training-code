from __future__ import annotations

import re
import string
from dataclasses import dataclass

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"


_ALLOWED = set(string.ascii_lowercase + " '.,!?;:-")
_VOCAB_SYMBOLS = [PAD, UNK, BOS, EOS] + list("abcdefghijklmnopqrstuvwxyz") + [
    " ",
    "'",
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
    "-",
]


def normalize_text(text: str, keep_punctuation: bool = True) -> str:
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("&", " and ")
    allowed = _ALLOWED if keep_punctuation else set(string.ascii_lowercase + " '")
    text = "".join(ch if ch in allowed else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class TextTokenizer:
    symbols: list[str]
    use_phonemes: bool = False
    keep_punctuation: bool = True

    @classmethod
    def default(cls, use_phonemes: bool = False, keep_punctuation: bool = True) -> "TextTokenizer":
        return cls(symbols=list(_VOCAB_SYMBOLS), use_phonemes=use_phonemes, keep_punctuation=keep_punctuation)

    @property
    def pad_id(self) -> int:
        return self.symbol_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.symbol_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.symbol_to_id[EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    @property
    def symbol_to_id(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.symbols)}

    def normalize(self, text: str) -> str:
        if self.use_phonemes:
            return self._phonemize(text)
        return normalize_text(text, keep_punctuation=self.keep_punctuation)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        normalized = self.normalize(text)
        stoi = self.symbol_to_id
        ids = [stoi.get(ch, stoi[UNK]) for ch in normalized]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        specials = {PAD, BOS, EOS}
        return "".join(self.symbols[i] for i in ids if 0 <= i < len(self.symbols) and self.symbols[i] not in specials)

    def to_dict(self) -> dict[str, object]:
        return {
            "symbols": self.symbols,
            "use_phonemes": self.use_phonemes,
            "keep_punctuation": self.keep_punctuation,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "TextTokenizer":
        use_phonemes = bool(config.get("use_phonemes", False))
        if use_phonemes:
            # Placeholder structure: phonemizer can be added without changing the model interface.
            # For now, this validates availability and falls back to normalized characters.
            try:
                import phonemizer  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("text.use_phonemes=true requires installing phonemizer") from exc
        return cls.default(
            use_phonemes=use_phonemes,
            keep_punctuation=bool(config.get("keep_punctuation", True)),
        )

    def _phonemize(self, text: str) -> str:
        raise NotImplementedError(
            "Phoneme mode is intentionally optional and not implemented in v0.1. "
            "Set text.use_phonemes=false for the character tokenizer."
        )
