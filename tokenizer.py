from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple
import json

Pair = Tuple[int, int]


class BPEUtils:
    """Small helper methods used by the tokenizer."""

    @staticmethod
    def get_pair_stats(ids: List[int]) -> Dict[Pair, int]:
        counts: Dict[Pair, int] = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(ids: List[int], pair: Pair, new_idx: int) -> List[int]:
        merged: List[int] = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(new_idx)
                i += 2
            else:
                merged.append(ids[i])
                i += 1

        return merged


@dataclass
class BPETokenizer:
    vocab_size: int = 256
    merges: Dict[Pair, int] = field(default_factory=dict)
    vocab: Dict[int, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE.")
        self._reset_vocab()
        if self.merges:
            self._rebuild_vocab()

    def _reset_vocab(self) -> None:
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def _rebuild_vocab(self) -> None:
        self._reset_vocab()
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def train(self, text: str) -> List[int]:
        """
        Learn merges from text and return the final compressed token ids.
        """
        ids = list(text.encode("utf-8"))
        self.merges.clear()
        self._reset_vocab()

        num_merges = self.vocab_size - 256
        for i in range(num_merges):
            pair_stats = BPEUtils.get_pair_stats(ids)
            if not pair_stats:
                break

            pair = max(pair_stats, key=pair_stats.get) # type: ignore
            new_idx = 256 + i

            ids = BPEUtils.merge(ids, pair, new_idx)
            self.merges[pair] = new_idx
            self.vocab[new_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        return ids

    def encode(self, text: str) -> List[int]:
        """
        Encode text using already learned merges.
        """
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            pair_stats = BPEUtils.get_pair_stats(ids)
            if not pair_stats:
                break

            pair = min(pair_stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            ids = BPEUtils.merge(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """
        Decode token ids back into a Python string.
        """
        raw = b"".join(self.vocab[idx] for idx in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """
        Save merges and config to disk as JSON.
        """
        payload = {
            "vocab_size": self.vocab_size,
            "merges": [
                {"pair": [p0, p1], "idx": idx}
                for (p0, p1), idx in self.merges.items()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load a tokenizer from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        merges = {
            tuple(item["pair"]): item["idx"]
            for item in payload["merges"]
        }
        return cls(vocab_size=payload["vocab_size"], merges=merges)


if __name__ == "__main__":
    text = "Hello, my name is Leo! My name is like the Pope! Pope Leo!"

    tokenizer = BPETokenizer(vocab_size=276)

    trained_ids = tokenizer.train(text)
    print("Trained ids:", trained_ids)
    print("Decoded trained ids:", tokenizer.decode(trained_ids))

    encoded = tokenizer.encode("Hello, my name is Leo!")
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))
