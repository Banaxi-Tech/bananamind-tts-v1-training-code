from __future__ import annotations

import re
import string
from dataclasses import dataclass

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"


_PUNCTUATION_SYMBOLS = set(".,!?;:-")
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


def normalize_text(
    text: str,
    keep_punctuation: bool = True,
    symbols: list[str] | None = None,
    ampersand_replacement: str = " and ",
) -> str:
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u201e", '"').replace("\u201f", '"')
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("&", ampersand_replacement)
    if symbols is None:
        allowed = set(_ALLOWED)
    else:
        special = {PAD, UNK, BOS, EOS}
        allowed = {symbol for symbol in symbols if len(symbol) == 1 and symbol not in special}
    if not keep_punctuation:
        allowed -= _PUNCTUATION_SYMBOLS
    text = "".join(ch if ch in allowed else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class TextTokenizer:
    symbols: list[str]
    use_phonemes: bool = False
    keep_punctuation: bool = True
    ampersand_replacement: str = " and "

    @classmethod
    def default(
        cls,
        use_phonemes: bool = False,
        keep_punctuation: bool = True,
        extra_symbols: list[str] | None = None,
        ampersand_replacement: str = " and ",
    ) -> "TextTokenizer":
        symbols = list(_VOCAB_SYMBOLS)
        for symbol in extra_symbols or []:
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        return cls(
            symbols=symbols,
            use_phonemes=use_phonemes,
            keep_punctuation=keep_punctuation,
            ampersand_replacement=ampersand_replacement,
        )

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
        return normalize_text(
            text,
            keep_punctuation=self.keep_punctuation,
            symbols=self.symbols,
            ampersand_replacement=self.ampersand_replacement,
        )

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
            "ampersand_replacement": self.ampersand_replacement,
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
        symbols = config.get("symbols")
        ampersand_replacement = str(config.get("ampersand_replacement", " and "))
        if isinstance(symbols, list) and symbols:
            return cls(
                symbols=[str(symbol) for symbol in symbols],
                use_phonemes=use_phonemes,
                keep_punctuation=bool(config.get("keep_punctuation", True)),
                ampersand_replacement=ampersand_replacement,
            )
        extra_symbols = config.get("extra_symbols", [])
        if not isinstance(extra_symbols, list):
            extra_symbols = []
        return cls.default(
            use_phonemes=use_phonemes,
            keep_punctuation=bool(config.get("keep_punctuation", True)),
            extra_symbols=[str(symbol) for symbol in extra_symbols],
            ampersand_replacement=ampersand_replacement,
        )

    def _phonemize(self, text: str) -> str:
        raise NotImplementedError(
            "Phoneme mode is intentionally optional and not implemented in v0.1. "
            "Set text.use_phonemes=false for the character tokenizer."
        )
