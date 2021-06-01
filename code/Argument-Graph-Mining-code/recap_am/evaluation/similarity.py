import logging
from typing import Any, Dict, List, Optional, Set

import nltk
import edlib

from recap_am.model.config import Config

import recap_argument_graph as ag

logger = logging.getLogger(__name__)
config = Config.get_instance()


# def _jaccard(node1: ag.Node, node2: ag.Node) -> float:
#     return 1 - nltk.jaccard_distance(
#         {t.text for t in node1.text}, {t.text for t in node2.text}
#     )


def _edit(node1: ag.Node, node2: ag.Node) -> float:
    # distance = nltk.edit_distance(node1.raw_text, node2.raw_text)
    distance = edlib.align(node1.raw_text, node2.raw_text)["editDistance"]

    return 1 - (distance / max(len(node1.raw_text), len(node2.raw_text)))


def _exact(node1: ag.Node, node2: ag.Node) -> float:
    return 1.0 if node1.raw_text == node2.raw_text else 0.0


def nodes(node1: ag.Node, node2: ag.Node) -> float:
    switch = {
        # "jaccard": _jaccard,
        "edit": _edit,
        "exact": _exact,
    }

    if node1.category == node2.category:
        if node1.raw_text == node2.raw_text:
            return 1.0
        elif node1.category == ag.NodeCategory.I:
            return switch[config["evaluation"]["similarity"]](node1, node2)

    return 0.0


def edges(edge1: ag.Edge, edge2: ag.Edge) -> float:
    return 0.5 * (nodes(edge1.start, edge2.start) + nodes(edge1.end, edge2.end))
