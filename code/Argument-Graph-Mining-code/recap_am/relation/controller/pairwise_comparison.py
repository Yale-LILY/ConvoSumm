import logging
from collections import defaultdict

import recap_argument_graph as ag
import copy
from recap_am.model.config import config
from recap_am.relation.controller import mc_from_relations
from recap_am.relation.controller.attack_support import (
    _load_model,
    _transform,
    _predict,
)
from recap_am.relation.model.relation import Relation, RelationClass
import numpy as np


def compare_all(adus, mc):
    """Compute classification for all ADU Pairs."""
    model = _load_model()
    relations = defaultdict(dict)
    for adu in adus:
        sample = _transform(adu, mc)
        pred_type, pred_prob = _predict(sample, model)
        relations[adu][mc] = Relation(mc, pred_prob, pred_type)
        relations[adu]["main"] = relations[adu][mc]
        if adu == mc:
            relations[adu]["main"] = Relation(mc, 0.0, pred_type)
        for adu2 in adus:
            if adu == adu2:
                pass
            else:
                sample = _transform(adu, adu2)
                pred_type, pred_prob = _predict(sample, model)
                relations[adu][adu2] = Relation(adu2, pred_prob, pred_type)
                if (
                    relations[adu][adu2].probability
                    > relations[adu]["main"].probability
                ):
                    relations[adu]["main"] = relations[adu][adu2]
    return relations


def run(doc, preset_mc: bool):
    """Create Graph through classfication values."""
    mc = doc._.MajorClaim
    adus = doc._.ADU_Sents

    if isinstance(mc, list):
        mc = mc[0]
        if mc == []:
            mc = adus.pop(0)
    elif not mc:
        mc = adus.pop(0)
    relations = compare_all(adus, mc)

    if config["adu"]["MC"]["method"] == "relations" and not preset_mc:
        mc = mc_from_relations.run_spacy(adus, relations)
        relations = compare_all(adus, mc)

    graph = ag.Graph(name=doc._.key.split("/")[-1])
    mc_node = ag.Node(
        key=graph.keygen(), text=mc, category=ag.NodeCategory.I, major_claim=True
    )
    graph.add_node(mc_node)
    outer_adus = [a for a in adus if not a == mc]
    inner_adus = []
    nodes = dict()
    connections = dict()
    connections[mc] = []
    for adu in outer_adus:
        main_rel = relations[adu]["main"]
        if relations[adu][mc].probability >= main_rel.probability * 0.90:
            logging.debug("MC Match")

            if relations[adu][mc].classification == RelationClass.ATTACK:
                snode = ag.Node(
                    key=graph.keygen(),
                    text="Default Conflict",
                    category=ag.NodeCategory.CA,
                )
            elif relations[adu][mc].classification == RelationClass.SUPPORT:
                snode = ag.Node(
                    key=graph.keygen(),
                    text="Default Inference",
                    category=ag.NodeCategory.RA,
                )
            else:
                snode = None

            if snode:
                cnode = ag.Node(
                    key=graph.keygen(), text=adu, category=ag.NodeCategory.I
                )
                nodes[adu] = cnode
                graph.add_edge(ag.Edge(key=graph.keygen(), start=cnode, end=snode))
                graph.add_edge(ag.Edge(key=graph.keygen(), start=snode, end=mc_node))
                outer_adus.remove(adu)
                inner_adus.append(adu)
    if len(graph.incoming_nodes[mc_node]) == 0:
        iterator = 0
        snode = None
        designated_adu = None

        while snode == None and iterator < len(outer_adus):
            designated_adu = outer_adus[iterator]
            if relations[designated_adu][mc].classification == RelationClass.ATTACK:
                snode = ag.Node(
                    key=graph.keygen(),
                    text="Default Conflict",
                    category=ag.NodeCategory.CA,
                )
            elif relations[designated_adu][mc].classification == RelationClass.SUPPORT:
                snode = ag.Node(
                    key=graph.keygen(),
                    text="Default Inference",
                    category=ag.NodeCategory.RA,
                )
            else:
                iterator += 1
                snode = None
        if not snode or not designated_adu:
            if outer_adus == []:
                logging.info("No ADUs classified, aborting")
                return graph
            else:
                designated_adu = outer_adus[0]
                snode = snode = ag.Node(
                    key=graph.keygen(),
                    text="Default Inference",
                    category=ag.NodeCategory.RA,
                )
        cnode = ag.Node(
            key=graph.keygen(), text=designated_adu, category=ag.NodeCategory.I
        )
        nodes[designated_adu] = cnode
        graph.add_edge(ag.Edge(key=graph.keygen(), start=cnode, end=snode))
        graph.add_edge(ag.Edge(key=graph.keygen(), start=snode, end=mc_node))
        outer_adus.remove(designated_adu)
        inner_adus.append(designated_adu)

    max_iter = 0

    while len(outer_adus) > 0 and max_iter < 40000:
        max_iter += 1
        for adu in outer_adus:
            inner_found = False
            for adu2 in inner_adus:
                if adu2 == adu:
                    pass
                elif (
                    relations[adu][adu2].probability
                    >= relations[adu]["main"].probability * 0.98
                ):
                    logging.debug("Match")
                    if relations[adu][adu2].classification == RelationClass.ATTACK:
                        snode = ag.Node(
                            key=graph.keygen(),
                            text="Default Conflict",
                            category=ag.NodeCategory.CA,
                        )
                    elif relations[adu][adu2].classification == RelationClass.SUPPORT:
                        snode = ag.Node(
                            key=graph.keygen(),
                            text="Default Inference",
                            category=ag.NodeCategory.RA,
                        )
                    else:
                        snode = None

                    if snode:
                        if adu in nodes:
                            cnode1 = nodes[adu]
                        else:
                            cnode1 = ag.Node(
                                key=graph.keygen(),
                                text=adu,
                                category=ag.NodeCategory.I,
                            )
                            nodes[adu] = cnode1

                        if adu2 in nodes:
                            cnode2 = nodes[adu2]
                        else:
                            cnode2 = ag.Node(
                                key=graph.keygen(),
                                text=adu2,
                                category=ag.NodeCategory.I,
                            )
                            nodes[adu2] = cnode2
                        graph.add_edge(
                            ag.Edge(key=graph.keygen(), start=cnode1, end=snode)
                        )
                        graph.add_edge(
                            ag.Edge(key=graph.keygen(), start=snode, end=cnode2)
                        )
                        inner_found = True
            if inner_found:
                outer_adus.remove(adu)
                inner_adus.append(adu)
    if len(outer_adus) > 0:
        for adu in outer_adus:
            snode = ag.Node(
                key=graph.keygen(),
                text="Default Inference",
                category=ag.NodeCategory.RA,
            )
            cnode = ag.Node(key=graph.keygen(), text=adu, category=ag.NodeCategory.I,)
            graph.add_edge(ag.Edge(key=graph.keygen(), start=cnode, end=snode))
            graph.add_edge(ag.Edge(key=graph.keygen(), start=snode, end=mc_node))

    return graph
