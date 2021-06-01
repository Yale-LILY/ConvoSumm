from __future__ import absolute_import, annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import typing as t
import recap_argument_graph as ag
import werkzeug
from nltk.tokenize import sent_tokenize

from recap_am.model.config import Config
from recap_am.controller import nlp

logger = logging.getLogger(__name__)
config = Config.get_instance()


extensions = [".json", ".txt", ".text", ".label", ".ann"]


@dataclass
class Query:
    name: str  # File name without suffix
    text: t.Optional[
        str
    ] = None  # Content of .txt files, fallbacks: .text file, benchmark.plain_text
    benchmark: t.Optional[ag.Graph] = None  # Content of .json and .ann files
    _text: t.Optional[str] = None  # Content of .text files
    _labels: t.Optional[t.List[str]] = None  # Labels corresponding to .text files

    @classmethod
    def from_folder(cls, path: Path) -> t.List[Query]:
        queries: t.Dict[str, Query] = {}

        for ext in extensions:
            for file in sorted(path.rglob(f"*{ext}")):
                name = file.stem
                suffix = file.suffix

                if name not in queries:
                    queries[name] = Query(name)

                with file.open(encoding="utf-8") as f:
                    _parse_file(name, suffix, f, queries[name])

        _postprocess(queries)

        return list(queries.values())

    @classmethod
    def from_flask(
        cls, files: t.Optional[t.Iterable[werkzeug.FileStorage]]
    ) -> t.List[Query]:
        queries: t.Dict[str, Query] = {}

        if files:
            for file in files:
                base = os.path.basename(file.filename or ".")
                name, suffix = os.path.splitext(base)[0:2]

                if suffix in extensions:
                    if name not in queries:
                        queries[name] = Query(name)

                    _parse_file(name, suffix, file.stream, queries[name])

        _postprocess(queries)

        return list(queries.values())


def _postprocess(queries: t.Mapping[str, Query]) -> None:
    for query in queries.values():
        if query._text and not query.text:
            query.text = query._text

        elif query.benchmark and not query.text:
            query.text = query.benchmark.plain_text

        if query._text and query._labels and not query.benchmark:
            query.benchmark = _create_graph(query.name, query._text, query._labels)


def _parse_file(name: str, suffix: str, file: t.IO, query: Query) -> None:
    if suffix == ".json":
        query.benchmark = _parse_json(name, file)
    elif suffix == ".txt":
        query.text = _parse_txt(file)
    elif suffix == ".text":
        query._text = _parse_txt(file)
    elif suffix == ".label":
        query._labels = _parse_label(file)
    elif suffix == ".ann":
        query.benchmark = _parse_ann(name, file)


def _parse_txt(file: t.IO) -> str:
    return file.read()


def _parse_json(name: str, file: t.IO) -> ag.Graph:
    return ag.Graph.from_json(file, name, nlp=nlp.parse)


def _parse_ann(name: str, file: t.IO) -> ag.Graph:
    return ag.Graph.from_brat(file, name, nlp=nlp.parse)


def _parse_label(file: t.IO) -> t.List[str]:
    return file.read().splitlines()


def _create_graph(name: str, text: str, labels: t.Iterable[str]) -> ag.Graph:
    graph = ag.Graph(name)
    doc = nlp.parse(text)
    sents = list(doc.sents)

    for sent, line_label in zip(sents, labels):
        if line_label != "None":
            node = ag.Node(graph.keygen(), sent, ag.NodeCategory.I)

            if line_label == "MajorClaim":
                node.major_claim = True

            graph.add_node(node)

    return graph
