import typing as t

from spacy.tokens.doc import Doc

from recap_am.model.config import config
from recap_am.relation.controller import pairwise_comparison, flat_tree, adu_position


def main(doc: Doc, relations: t.Any, preset_mc: bool = False, index: int = -1, combine: bool = False, claim2node: t.Any = None, claim_sents: t.Any = None, use_premises: bool = False, only_claims: bool = False):
    method = config["relation"]["method"]

    if combine:
        return flat_tree.run_combine(claim2node, claim_sents)
    if method == "ours":
        return flat_tree.run_ours(doc, preset_mc)
    elif method == "ours_new":
        return flat_tree.run_ours_new(doc, preset_mc, index, use_premises, only_claims)
    #if method == "adu_position":
    #    return adu_position.run(doc, relations, preset_mc)

    #elif method == "pairwise_comparison":
    #    return pairwise_comparison.run(doc, preset_mc)

    #elif method == "flat_tree":
    #return flat_tree.run(doc, preset_mc)

    raise ValueError("Wrong config value for 'relation.method'.")
