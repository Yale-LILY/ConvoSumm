import typing as t
from dataclasses import dataclass

import recap_argument_graph as ag
from spacy.tokens.doc import Doc

from recap_am.adu.classify import predict_clpr, predict_mc
from recap_am.adu.feature_select import filter_feats, add_embeddings
from recap_am.controller import nlp
from recap_am.controller.extract_features import set_features
from recap_am.controller.nlp import parse
from recap_am.evaluation import similarity
from recap_am.model.config import Config
from recap_am.model.statistic import Statistic
from recap_am.relation import construct_graph
from recap_am.relation.controller import attack_support
from recap_am.relation.model.relation import Relation, RelationClass

config = Config.get_instance()


@dataclass(frozen=True)
class MappingEntry:
    obj: t.Any
    sim: float


Mappings = t.Dict[t.Any, t.Optional[MappingEntry]]

"""Compare the automatically and manually generated graphs.

When comparing edges, the type (i.e., the S-nodes) are ignored.
preset assessment of the parts is possible by providing the ADUs from the reference/benchmark graph.
"""


def run(
    statistic: Statistic,
    generated_doc_end2end: Doc,
    generated_relations_end2end: t.Dict[str, t.List[Relation]],
    generated_graph_end2end: ag.Graph,
    benchmark_graph: ag.Graph,
) -> ag.Graph:
    (
        generated_doc_preset,
        generated_relations_preset,
        generated_graph_preset,
    ) = _preset_adus(benchmark_graph)

    inode_mappings_end2end = _map_inodes(benchmark_graph, generated_graph_end2end)
    inode_mappings_preset = _map_inodes(benchmark_graph, generated_graph_preset)

    statistic.inodes_agreement_end2end = eval_inodes(
        benchmark_graph, generated_graph_end2end, inode_mappings_end2end
    )
    statistic.inodes_agreement_preset = eval_inodes(
        benchmark_graph, generated_graph_preset, inode_mappings_preset
    )

    statistic.mc_agreement_end2end = eval_major_claim(inode_mappings_end2end)
    statistic.mc_agreement_preset = eval_major_claim(inode_mappings_preset)

    statistic.snodes_agreement_end2end = eval_snodes(
        benchmark_graph, inode_mappings_end2end, generated_relations_end2end,
    )
    statistic.snodes_agreement_preset = eval_snodes(
        benchmark_graph, inode_mappings_preset, generated_relations_preset,
    )

    benchmark_igraph = _strip_snodes(benchmark_graph)
    generated_igraph_end2end = _strip_snodes(generated_graph_end2end)
    generated_igraph_preset = _strip_snodes(generated_graph_preset)

    edge_mappings_end2end = _map_edges(benchmark_igraph, generated_igraph_end2end)
    edge_mappings_preset = _map_edges(benchmark_igraph, generated_igraph_preset)

    statistic.edges_agreement_end2end = eval_edges(
        benchmark_igraph, generated_igraph_end2end, edge_mappings_end2end
    )
    statistic.edges_agreement_preset = eval_edges(
        benchmark_igraph, generated_igraph_preset, edge_mappings_preset
    )

    return generated_graph_preset


def _strip_snodes(base_graph: ag.Graph) -> ag.Graph:
    g = base_graph.copy(nlp=nlp.parse)
    g.strip_snodes()

    return g


def eval_major_claim(inode_mappings: Mappings) -> float:
    for benchmark, entry in inode_mappings.items():
        if benchmark.major_claim:
            if entry and entry.obj.major_claim:
                return 1.0
            else:
                return 0.0

    return 1.0


def eval_inodes(
    benchmark_graph: ag.Graph, generated_graph: ag.Graph, inode_mappings: Mappings
) -> float:
    return sum((entry.sim for entry in inode_mappings.values() if entry)) / max(
        len(generated_graph.inodes), len(benchmark_graph.inodes), 1
    )


def eval_snodes(
    benchmark_graph: ag.Graph,
    inode_mappings: Mappings,
    relations: t.Mapping[str, t.Iterable[Relation]],
) -> float:
    """Count the matching incoming/outgoing nodes based on the mapping."""

    total_relations = 0
    matching_relations = 0

    for snode_benchmark in benchmark_graph.snodes:
        incoming_benchmarks = benchmark_graph.incoming_nodes[snode_benchmark]
        outgoing_benchmarks = benchmark_graph.outgoing_nodes[snode_benchmark]

        for incoming_benchmark in incoming_benchmarks:
            for outgoing_benchmark in outgoing_benchmarks:
                incoming_generated = inode_mappings.get(incoming_benchmark)
                outgoing_generated = inode_mappings.get(outgoing_benchmark)

                if incoming_generated and outgoing_generated:
                    total_step, matching_step = _eval_snodes_category(
                        snode_benchmark.category.value,
                        incoming_generated.obj,
                        outgoing_generated.obj,
                        relations,
                    )

                    total_relations += total_step
                    matching_relations += matching_step

    if total_relations > 0:
        return matching_relations / total_relations

    return 0.0


def _eval_snodes_category(
    benchmark_category: str,
    incoming_generated: ag.Node,
    outgoing_generated: ag.Node,
    relations: t.Mapping[str, t.Iterable[Relation]],
) -> t.Tuple[int, int]:
    total_relations = 0
    matching_relations = 0

    # relations_generated = relations.get(incoming_generated.raw_text, list())
    relations_generated = relations[incoming_generated.raw_text]

    for relation_generated in relations_generated:
        if outgoing_generated and outgoing_generated.raw_text == relation_generated.adu:
            total_relations += 1

            generated_category = relation_generated.classification

            if generated_category == RelationClass.NONE:
                generated_category = RelationClass.SUPPORT

            if benchmark_category == generated_category.value:
                matching_relations += 1

    return total_relations, matching_relations


def eval_edges(
    benchmark_graph: ag.Graph, generated_graph: ag.Graph, edge_mappings: Mappings
) -> float:
    return sum((1 for entry in edge_mappings.values() if entry)) / max(
        len(generated_graph.edges), len(benchmark_graph.edges), 1
    )


def _preset_segment(doc):
    for token in doc[:-1]:
        if "\n\n" in token.text_with_ws:
            doc[token.i + 1].is_sent_start = True
        else:
            doc[token.i + 1].is_sent_start = False

    return doc


def _preset_adus(
    graph: ag.Graph, preset_mc: bool = True
) -> t.Tuple[Doc, t.Dict[str, t.List[Relation]], ag.Graph]:
    sents = []
    labels = []
    mc_list = []

    for node in graph.inodes:
        sent_text = node.raw_text

        # if not sent_text.endswith("."):
        #    sent_text += "."

        sents.append(sent_text)
        labels.append(1)

        if node.major_claim:
            mc_list.append(1)
        else:
            mc_list.append(0)

    doc_text = "\n\n".join(sents)

    parse.add_pipe(_preset_segment, name="preset_segment", before="parser")
    doc = parse(doc_text)
    parse.remove_pipe("preset_segment")

    total_inodes = len(graph.inodes)
    total_sents = len(list(doc.sents))

    # assert total_inodes == total_sents

    if total_sents > total_inodes:
        labels += [1] * (total_sents - total_inodes)
        mc_list += [0] * (total_sents - total_inodes)

    elif total_sents < total_inodes:
        labels = labels[:total_sents]
        mc_list = mc_list[:total_sents]

    if not 1 in mc_list:
        mc_list[0] = 1

    if preset_mc:
        doc._.MC_List = mc_list
    else:
        doc._.MC_List = predict_mc(doc)

    doc._.Labels = labels
    doc._.key = graph.name
    doc = set_features(doc)
    doc = filter_feats(doc, load=True)
    doc = add_embeddings(doc)
    doc._.CLPR_Features = doc._.Features
    doc = predict_clpr(doc)

    rel_types = attack_support.classify(doc._.ADU_Sents)

    # Create graph with relationships
    graph = construct_graph.main(doc, rel_types, preset_mc)

    return doc, rel_types, graph


def _map_inodes(benchmark_graph: ag.Graph, generated_graph: ag.Graph) -> Mappings:
    """Map nodes.

    One node of the generated graph might be mapped to multiple nodes in the benchmark graph.
    It can also happen that nodes from both graphs do not have a mapping.
    """

    benchmark_mappings = _map(
        benchmark_graph.inodes, generated_graph.inodes, similarity.nodes,
    )

    generated_mappings = _map(
        generated_graph.inodes, benchmark_graph.inodes, similarity.nodes,
    )

    _verify_mappings(benchmark_mappings, generated_mappings)

    return benchmark_mappings


def _map_edges(benchmark_graph: ag.Graph, generated_graph: ag.Graph) -> Mappings:
    # benchmark_mappings = _map(
    #     benchmark_graph.edges, generated_graph.edges, similarity.edges,
    # )
    #
    # generated_mappings = _map(
    #     generated_graph.edges, benchmark_graph.edges, similarity.edges,
    # )
    #
    # _verify_mappings(benchmark_mappings, generated_mappings)
    #
    # return benchmark_mappings

    inode_mappings = _map_inodes(benchmark_graph, generated_graph)

    edge_mappings = {}

    for bench_edge in benchmark_graph.edges:
        edge_mappings[bench_edge] = None

        for gen_edge in generated_graph.edges:
            gen_nodes = (gen_edge.start, gen_edge.end)
            bench_start = inode_mappings[bench_edge.start]
            bench_end = inode_mappings[bench_edge.end]

            if (
                bench_start
                and bench_end
                and bench_start.obj in gen_nodes
                and bench_end.obj in gen_nodes
            ):
                edge_mappings[bench_edge] = MappingEntry(
                    gen_edge, similarity.edges(bench_edge, gen_edge)
                )

    return edge_mappings


def _map(queries, cases, sim_func, threshold=0.0):
    result = {}

    for query in queries:
        result[query] = None
        best_match = (None, 0.0)

        for case in cases:
            sim = sim_func(query, case)

            if sim > best_match[1]:
                best_match = (case, sim)

        best_case = best_match[0]
        best_sim = best_match[1]

        if best_case and best_sim >= threshold:
            result[query] = MappingEntry(best_case, best_sim)

    return result


def _verify_mappings(query_mapping, case_mapping):
    for query, query_entry in query_mapping.items():
        if query_entry:
            case_entry = case_mapping[query_entry.obj]

            if not case_entry or case_entry.obj != query:
                query_mapping[query] = None
