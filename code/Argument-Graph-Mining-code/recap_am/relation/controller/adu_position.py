from typing import List, Dict, Optional

import recap_argument_graph as ag
from spacy.tokens import Span, Doc

from recap_am.model.config import config
from recap_am.relation.controller import mc_from_relations
from recap_am.relation.controller.attack_support import (
    Relation,
    RelationClass,
    classify,
)


def run(
    doc: Doc, relations: Optional[Dict[str, List[Relation]]], preset_mc: bool
) -> ag.Graph:
    adus = doc._.ADU_Sents
    claims = doc._.Claim_Sents
    premises = doc._.Premise_Sents
    mc = doc._.MajorClaim

    if config["adu"]["MC"]["method"] == "relations" and not preset_mc:
        mc = mc_from_relations.run_str(adus, relations)

    if not relations:
        relations = classify(adus)

    graph = ag.Graph(name=doc._.key.split("/")[-1])
    mc_node = ag.Node(graph.keygen(), mc, ag.NodeCategory.I, major_claim=True)
    cnodes = []
    graph.add_node(mc_node)

    for claim in claims:
        if claim != mc:
            cnode = ag.Node(graph.keygen(), claim, ag.NodeCategory.I)
            snode = _gen_snode(graph, relations[claim.text], mc)

            if snode:
                cnodes.append(cnode)

                graph.add_edge(ag.Edge(graph.keygen(), start=cnode, end=snode))
                graph.add_edge(ag.Edge(graph.keygen(), start=snode, end=mc_node))

    for premise in premises:
        if premise != mc:
            pnode = ag.Node(graph.keygen(), premise, ag.NodeCategory.I)
            match = (mc_node, 0)

            if cnodes:
                scores = {
                    cnode: min(
                        abs(cnode.text.start - pnode.text.end),
                        abs(cnode.text.end - pnode.text.start),
                    )
                    for cnode in cnodes
                }

                min_score = min(scores.values())
                candidates = [
                    node for node, score in scores.items() if score == min_score
                ]

                for candidate in candidates:
                    sim = pnode.text.similarity(candidate.text)
                    if sim > match[1]:
                        match = (candidate, sim)

            snode = _gen_snode(graph, relations[premise.text], match[0].text)

            if snode:
                graph.add_edge(ag.Edge(graph.keygen(), start=pnode, end=snode))
                graph.add_edge(ag.Edge(graph.keygen(), start=snode, end=match[0]))

    return graph


def _gen_snode(
    graph: ag.Graph, relations: List[Relation], adu: Span
) -> Optional[ag.Node]:
    candidates = list(filter(lambda rel: rel.adu == adu.text, relations))

    if candidates:
        relation = candidates[0]

        if relation and relation.classification == RelationClass.ATTACK:
            return ag.Node(graph.keygen(), "Default Conflict", ag.NodeCategory.CA)

        elif relation and relation.classification == RelationClass.SUPPORT:
            return ag.Node(graph.keygen(), "Default Inference", ag.NodeCategory.RA)

    return None
