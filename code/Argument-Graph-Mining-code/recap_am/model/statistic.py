import json
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from recap_am.model.query import Query


@dataclass
class Statistic:
    """One statistic per input text with filename `id`."""

    query: Query
    inodes_agreement_end2end: float = 0.0
    inodes_agreement_preset: float = 0.0
    mc_agreement_end2end: float = 0.0
    mc_agreement_preset: float = 0.0
    snodes_agreement_end2end: float = 0.0
    snodes_agreement_preset: float = 0.0
    edges_agreement_end2end: float = 0.0
    edges_agreement_preset: float = 0.0
    duration: float = 0.0

    def to_dict(self) -> t.Dict[str, str]:
        return {
            "Duration": _round(self.duration),
            "I-Nodes Agreement End2End": _round(self.inodes_agreement_end2end),
            "I-Nodes Agreement Preset": _round(self.inodes_agreement_preset),
            "Major Claim Agreement End2End": _round(self.mc_agreement_end2end),
            "Major Claim Agreement Preset": _round(self.mc_agreement_preset),
            "S-Nodes Agreement End2End": _round(self.snodes_agreement_end2end),
            "S-Nodes Agreement Preset": _round(self.snodes_agreement_preset),
            "Edges Agreement End2End": _round(self.edges_agreement_end2end),
            "Edges Agreement Preset": _round(self.edges_agreement_preset),
        }

    def save(self, folder: Path):
        _save(self.to_dict(), folder / f"{self.query.name}-stats.json")


class Statistics(t.MutableSequence[Statistic]):
    """Statistics per run and aggregation of individual statistic values.

    Each score in the Statistic class should be mirrored here.
    """

    _store: t.List[Statistic]
    name: str
    duration: float = 0.0

    def __init__(self, name: str):
        self._store = []
        self.name = name

    def new(self, query: Query) -> Statistic:
        statistic = Statistic(query)
        self.append(statistic)

        return statistic

    def _mean(self, attr: str) -> float:
        score = 0

        for item in self:
            score += getattr(item, attr)

        return score / len(self) if len(self) > 0 else 0.0

    def _sum(self, attr: str) -> float:
        return sum([getattr(item, attr) for item in self])

    def to_dict(self) -> t.Dict[str, str]:
        return {
            "Total Processing Duration": _round(self.duration),
            "Total Mining Duration": _round(self._sum("duration")),
            "Average Mining Duration": _round(self._mean("duration")),
            "I-Nodes Agreement End2End": _round(self._mean("inodes_agreement_end2end")),
            "I-Nodes Agreement Preset": _round(self._mean("inodes_agreement_preset")),
            "Major Claim Agreement End2End": _round(self._mean("mc_agreement_end2end")),
            "Major Claim Agreement Preset": _round(self._mean("mc_agreement_preset")),
            "S-Nodes Agreement End2End": _round(self._mean("snodes_agreement_end2end")),
            "S-Nodes Agreement Preset": _round(self._mean("snodes_agreement_preset")),
            "Edges Agreement End2End": _round(self._mean("edges_agreement_end2end")),
            "Edges Agreement Preset": _round(self._mean("edges_agreement_preset")),
        }

    def save(self, folder: Path):
        import os
        _save(self.to_dict(), Path(os.path.join(folder,  f"{self.name}-stats.json")))

    # Methods to comply with interface
    def __getitem__(self, item):
        return self._store[item]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __len__(self):
        return len(self._store)

    def insert(self, key, value):
        self._store.insert(key, value)


def _save(data: t.Mapping[str, str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def _round(number: float) -> str:
    if number < 1:
        return ("%.3f" % number).lstrip("0")

    return "%.2f" % number
