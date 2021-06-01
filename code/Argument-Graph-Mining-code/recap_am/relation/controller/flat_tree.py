import logging
import uuid
from collections import defaultdict
from tqdm import tqdm

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

from collections import defaultdict

from fairseq.data.data_utils import collate_tokens
from scipy.special import softmax
from ast import literal_eval
import numpy as np
import torch
import scipy
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()
roberta.cuda()


from sentence_transformers import SentenceTransformer
import scipy.cluster.hierarchy as hcluster
embedder = SentenceTransformer('distilroberta-base-paraphrase-v1')
#embedder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
cosine_threshold = 0.8

bsz = 128

def compare_all(adus, mc):
    """Compute classification for all ADU Pairs."""
    model = _load_model()
    relations = defaultdict(dict)
    for adu in adus:
        sample = _transform(adu, mc)
        pred_type, pred_prob = _predict(sample, model)
        relations[adu][mc] = Relation(mc, pred_prob, pred_type)
        relations[adu]["main"] = relations[adu][mc]
        for adu2 in adus:
            if adu == adu2 or adu2 == mc:
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


connections = defaultdict(dict)


def run(doc, preset_mc: bool):
    """Create Graph through classfication values."""
    mc = doc._.MajorClaim
    adus = doc._.ADU_Sents
    relations = compare_all(adus, mc)

    if config["adu"]["MC"]["method"] == "relations" and not preset_mc:
        mc = mc_from_relations.run_spacy(adus, relations)
        relations = compare_all(adus, mc)

    graph = ag.Graph(name=doc._.key.split("/")[-1])
    mc_node = ag.Node(
        key=graph.keygen(), text=str(mc), category=ag.NodeCategory.I, major_claim=True
    )
    graph.add_node(mc_node)
    outer_adus = [a for a in adus if not a == mc]
    for adu in outer_adus:
        cnode = ag.Node(key=graph.keygen(), text=str(adu), category=ag.NodeCategory.I)
        if relations[adu][mc].classification == RelationClass.ATTACK:
            snode = ag.Node(
                key=graph.keygen(), text="Default Conflict", category=ag.NodeCategory.CA
            )
        else:
            snode = ag.Node(
                key=graph.keygen(),
                text="Default Inference",
                category=ag.NodeCategory.RA,
            )

        logging.debug("Match")
        graph.add_edge(ag.Edge(key=graph.keygen(), start=cnode, end=snode))
        graph.add_edge(ag.Edge(key=graph.keygen(), start=snode, end=mc_node))
    return graph


def get_probs(pairs):
    actual_len = []
    batch = []
    slines = []
    ex_count = 0
    logprobs_return = []
    for cur_pair in pairs:
        if ex_count % bsz == 0 and ex_count != 0:
            with torch.no_grad():
                batch = []
                for batch_pair in slines:
                    enc_0 = roberta.encode(batch_pair[0]).tolist()
                    enc_1 = roberta.encode(batch_pair[1]).tolist()
                    actual_len.append(len(enc_0))
                    # not necessary for this data, only when applied on article level
                    enc_0_len = 512 - len(enc_1)
                    if len(enc_1) >= 512:
                        enc_1 = enc_1[:256]
                        enc_0_len = 512 - len(enc_1)
                    if enc_0_len < len(enc_0):
                        enc_0_len -= 1
                    if enc_0_len < len(enc_0):
                        batch.append(torch.tensor(enc_0[:enc_0_len] + [2, 2] + enc_1[1:], dtype=torch.int64))
                    else:
                        batch.append(torch.tensor(enc_0[:enc_0_len] + [2] + enc_1[1:], dtype=torch.int64))
                batch = collate_tokens(batch, pad_idx=1)
                logprobs = roberta.predict('mnli', batch)
                for prob in logprobs:
                    out = list(softmax(prob.cpu().numpy()))
                    logprobs_return.append(out)
                slines = []
        slines.append(cur_pair)
        ex_count += 1
    if slines != []:
        with torch.no_grad():
            batch = []
            for batch_pair in slines:
                enc_0 = roberta.encode(batch_pair[0]).tolist()
                enc_1 = roberta.encode(batch_pair[1]).tolist()
                enc_0_len = 512 - len(enc_1)
                if len(enc_1) >= 512:
                    enc_1 = enc_1[:256]
                    enc_0_len = 512 - len(enc_1)
                if enc_0_len < len(enc_0):
                    enc_0_len -= 1
                if enc_0_len < len(enc_0):
                    batch.append(torch.tensor(enc_0[:enc_0_len] + [2, 2] + enc_1[1:], dtype=torch.int64))
                else:
                    batch.append(torch.tensor(enc_0[:enc_0_len] + [2] + enc_1[1:], dtype=torch.int64))
            batch = collate_tokens(batch, pad_idx=1)
            logprobs = roberta.predict('mnli', batch)
            for prob in logprobs:
                out = list(softmax(prob.cpu().numpy()))
                logprobs_return.append(out)
    return logprobs_return


def run_ours(doc, preset_mc: bool):
    """Create Graph through classfication values."""
    adus = doc._.ADU_Sents
    labels = doc._.Labels
    clpr = doc._.CLPR_Labels
    graph = ag.Graph(name=doc._.key.split("/")[-1])

    claim_sents = [x for count, x in enumerate(adus) if clpr[count] == 1]
    premise_sents = [x for count, x in enumerate(adus) if clpr[count] == 0]
    claim2node = {str(adu): ag.Node(key=graph.keygen(), text=str(adu), category=ag.NodeCategory.I) for adu in claim_sents}
    premise2node = {str(adu): ag.Node(key=graph.keygen(), text=str(adu), category=ag.NodeCategory.I) for adu in premise_sents}
    todo = []
    claim2premises = defaultdict(list)
    for premise in premise_sents:
        for claim in claim_sents:
            premise_node = premise2node[str(premise)]
            claim_node = claim2node[str(claim)]
            claim2premises[claim_node] = []
            todo.append((premise_node.text, claim_node.text))
    logprobs = get_probs(todo)
    curindex = 0
    min_threshold = 0.33
    isolated_premises = []
    for premise in premise_sents:
        premise_node = premise2node[str(premise)]
        cur_logs = logprobs[curindex: curindex + len(claim_sents)]
        curindex += len(claim_sents)
        max_support_index = np.argmax([x[2] for x in cur_logs])
        max_attack_index = np.argmax([x[0] for x in cur_logs])
        max_attack = cur_logs[max_attack_index][0]
        max_support = cur_logs[max_support_index][2]
        #max_overall = max(max_attack, max_support)
        max_overall = max_support
        max_attack = -1
        if max_overall >= min_threshold:
            if max_attack > max_support:
                cur_claim = claim_sents[max_attack_index]
                snode = ag.Node(
                    key=graph.keygen(), text="Default Conflict", category=ag.NodeCategory.CA
                )
            else:
                cur_claim = claim_sents[max_support_index]
                snode = ag.Node(
                    key=graph.keygen(),
                    text="Default Inference",
                    category=ag.NodeCategory.RA,
                )
            cur_claim_node = claim2node[str(cur_claim)]
            #print("premise-node", premise_node)
            #print("claim-node", cur_claim_node)
            graph.add_edge(ag.Edge(key=graph.keygen(), start=premise_node, end=snode))
            graph.add_edge(ag.Edge(key=graph.keygen(), start=snode, end=cur_claim_node))
            claim2premises[cur_claim_node].append(str(premise))
        else:
            #snode = ag.Node(
            #    key=graph.keygen(),
            #    text="Isolated node",
            #    category=ag.NodeCategory.I,
            #)
            #graph.add_edge(ag.Edge(key=graph.keygen(), start=premise_node, end=snode))
            isolated_premises.append(premise)

    for premise in isolated_premises:
        for claim in claim_sents:
            premise_node = premise2node[str(premise)]
            claim_node = claim2node[str(claim)]
            todo.append((premise_node.text, claim_node.text))
    logprobs = get_probs(todo)
    curindex = 0
    for premise in isolated_premises:
        premise_node = premise2node[str(premise)]
        cur_logs = logprobs[curindex: curindex + len(claim_sents)]
        curindex += len(claim_sents)
        max_attack_index = np.argmax([x[0] for x in cur_logs])
        max_attack = cur_logs[max_attack_index][0]
        max_overall = max_attack
        max_support = -1
        if max_overall >= min_threshold:
            if max_attack > max_support:
                cur_claim = claim_sents[max_attack_index]
                snode = ag.Node(
                    key=graph.keygen(), text="Default Conflict", category=ag.NodeCategory.CA
                )
            cur_claim_node = claim2node[str(cur_claim)]
            graph.add_edge(ag.Edge(key=graph.keygen(), start=premise_node, end=snode))
            graph.add_edge(ag.Edge(key=graph.keygen(), start=snode, end=cur_claim_node))
            claim2premises[cur_claim_node].append(premise)
        else:
            snode = ag.Node(
                key=graph.keygen(),
                text="Isolated node",
                category=ag.NodeCategory.I,
            )
            graph.add_edge(ag.Edge(key=graph.keygen(), start=premise_node, end=snode))
            #isolated_premises.append(premise)

    todo = []
    #cluster_sents = [claim.text for claim in claim_sents]
    #corpus_embeddings = embedder.encode(cluster_sents)
    #cluster_assignment = hcluster.fclusterdata(
    #    corpus_embeddings, 
    #    1-cosine_threshold, 
    #    criterion="distance", 
    #    metric="cosine", 
    #    method="average"
    #    )
    #
    #num_clusters = len(set(cluster_assignment))
    #clustered_sentences = [[] for i in range(num_clusters)]
    #for sentence_id, cluster_id in enumerate(cluster_assignment):
    #    clustered_sentences[cluster_id-1].append(cluster_sents[sentence_id])
    #
    #
    #for i, cluster in enumerate(clustered_sentences):
    #    print("Cluster ", i+1)
    #    print(cluster)
    #    print("")
    #exit()
    claim2parent = {}
    todo = []
    for claim_1 in claim_sents:
        claim2parent[str(claim_1)] = str(claim_1)
        for claim_2 in claim_sents:
            if claim_1 == claim_2:
                continue
            todo.append((claim_1.text, claim_2.text))

    if len(todo) > 0:
        logprobs = get_probs(todo)
        curindex = 0
        claim2conflict = {}
        for claim_count, claim_1 in enumerate(claim_sents):
            cur_logs = logprobs[curindex: curindex + len(claim_sents) - 1]
            curindex += len(claim_sents) - 1
            max_support_index = np.argmax([x[2] for x in cur_logs])
            max_attack_index = np.argmax([x[0] for x in cur_logs])
            max_attack = cur_logs[max_attack_index][0]
            max_support = cur_logs[max_support_index][2]
            #print(max_attack, max_support)
            max_overall = max_support
            max_attack = -1
            if max_overall >= min_threshold:
                if max_attack > max_support:
                    print("error")
                    if max_attack_index >= claim_count:
                        max_attack_index += 1
                    cur_claim = claim_sents[max_attack_index]
                else:
                    if max_support_index >= claim_count:
                        max_support_index += 1
                    cur_claim = claim_sents[max_support_index]
                #cur_claim_node = claim2node[str(cur_claim)]
                #source_claim = claim2node[str(claim_1)]
                source_claim = claim2node[str(claim_1)]
                cur_claim_node = claim2node[str(cur_claim)]
                if claim2parent[str(cur_claim)] == str(claim_1):
                    continue
                claim2parent[str(claim_1)] = str(cur_claim)
                #if str(cur_claim_node) in claim2conflict:
                #    tmp_node = claim2conflict[str(cur_claim_node)]
                #else:
                #    tmp_node = ag.Node(key=graph.keygen(), text="Issue", category=ag.NodeCategory.I)
                #    claim2conflict[str(cur_claim_node)] = tmp_node
                #    claim2conflict[str(source_claim)] = tmp_node
                graph.add_edge(ag.Edge(key=graph.keygen(), start=source_claim, end=cur_claim_node))
                #graph.add_edge(ag.Edge(key=graph.keygen(), start=source_claim, end=tmp_node))
    parent_nodes = []
    for claim in claim_sents:
        orig_claim = str(claim)
        parent = claim2parent[orig_claim]
        while orig_claim != parent:
            orig_claim = parent
            parent = claim2parent[parent]
        parent_nodes.append(parent)
    #print("PARENTS!!!")
    parent_nodes = list(set(parent_nodes))
    #for ex in parent_nodes:
    #    print(ex)
    todo = []
    for claim_1 in parent_nodes:
        claim2parent[str(claim_1)] = str(claim_1)
        for claim_2 in parent_nodes:
            if claim_1 == claim_2:
                continue
            todo.append((claim_1, claim_2))
    potential_edges = []
    if len(todo) > 0:
        logprobs = get_probs(todo)
        curindex = 0
        claim2conflict = {}
        for claim_count, claim_1 in enumerate(parent_nodes):
            cur_logs = logprobs[curindex: curindex + len(parent_nodes) - 1]
            curindex += len(parent_nodes) - 1
            max_support_index = np.argmax([x[2] for x in cur_logs])
            max_attack_index = np.argmax([x[0] for x in cur_logs])
            max_attack = cur_logs[max_attack_index][0]
            max_support = cur_logs[max_support_index][2]
            max_overall = max_attack
            max_support = -1
            if max_overall >= min_threshold:
                if max_attack > max_support:
                    if max_attack_index >= claim_count:
                        max_attack_index += 1
                    cur_claim = parent_nodes[max_attack_index]
                else:
                    if max_support_index >= claim_count:
                        max_support_index += 1
                    cur_claim = parent_nodes[max_support_index]
                source_claim = claim2node[str(claim_1)]
                cur_claim_node = claim2node[str(cur_claim)]
                tmp_node = ag.Node(key=graph.keygen(), text="Issue", category=ag.NodeCategory.I)
                node_1 = ag.Edge(key=graph.keygen(), start=source_claim, end=tmp_node)
                node_2 = ag.Edge(key=graph.keygen(), start=cur_claim_node, end=tmp_node)
                potential_edges.append((node_1, node_2, max_attack))
    used_nodes = []
    sorted_potential_edges = sorted(potential_edges, key = lambda x: x[2], reverse=True)
    #print(potential_edges)
    for edge in sorted_potential_edges:
        node_1, node_2, score = edge
        if node_1.start.text in used_nodes or node_2.start.text in used_nodes:
            continue
        used_nodes.append(node_1.start.text)
        used_nodes.append(node_2.start.text)
        graph.add_edge(node_1)
        graph.add_edge(node_2)
        #print(score)
    return graph


def run_ours_new(doc, preset_mc: bool, query_count: int, use_premises: bool, only_claims: bool):
    """Create Graph through classfication values."""
    #adus = doc._.ADU_Sents
    adus = list(doc.sents)
    labels = doc._.Labels
    clpr = doc._.CLPR_Labels
    query_count = str(query_count)
    graph = ag.Graph(name=doc._.key.split("/")[-1])
    claim_sents = [x for count, x in enumerate(adus) if clpr[count] == 1]
    # TODO
    #print("Hi")
    if use_premises:
        premise_sents = [x for count, x in enumerate(adus) if clpr[count] != 1]
    else:
        premise_sents = [x for count, x in enumerate(adus) if clpr[count] == 0]
    claim2node = {str(adu): ag.Node(key=query_count + "-" + str(graph.keygen()) + str(uuid.uuid4().hex), text=str(adu), category=ag.NodeCategory.I) for adu in claim_sents}
    premise2node = {str(adu): ag.Node(key=query_count + "-" + str(graph.keygen()) + str(uuid.uuid4().hex), text=str(adu), category=ag.NodeCategory.I) for adu in premise_sents}
    todo = []
    claim2premises = defaultdict(list)
    for premise in premise_sents:
        for claim in claim_sents:
            premise_node = premise2node[str(premise)]
            claim_node = claim2node[str(claim)]
            claim2premises[claim_node] = []
            todo.append((premise_node.text, claim_node.text))
    logprobs = get_probs(todo)
    curindex = 0
    min_threshold = 0.33
    isolated_premises = []
    for premise in premise_sents:
        premise_node = premise2node[str(premise)]
        cur_logs = logprobs[curindex: curindex + len(claim_sents)]
        curindex += len(claim_sents)
        max_support_index = np.argmax([x[2] for x in cur_logs])
        max_support = cur_logs[max_support_index][2]
        if max_support >= min_threshold:
            cur_claim = claim_sents[max_support_index]
            cur_claim_node = claim2node[str(cur_claim)]
            claim2premises[cur_claim_node].append(premise)
        else:
            isolated_premises.append(premise)
    claim_starts = [claim.start for claim in claim_sents]
    isolated_premises_start = [premise.start for premise in isolated_premises]
    for premise_count, premise_start in enumerate(isolated_premises_start):
        premise = isolated_premises[premise_count]
        i = 0
        chosen_claim = None
        for claim_count, claim_start in enumerate(claim_starts):
            if claim_start > premise_start:
                if claim_count > 0:
                    chosen_claim = claim_sents[claim_count - 1]
                    break
                else:
                    chosen_claim = claim_sents[0]
                    break
        if chosen_claim is None:
            chosen_claim = claim_sents[-1]

        cur_claim_node = claim2node[str(chosen_claim)]
        premise_node = premise2node[str(premise)]

        claim2premises[cur_claim_node].append(premise)
    claim_sents_ordered = sorted(claim_sents, key = lambda x: x.start)
    for claim_sent in claim_sents_ordered:
        cur_claim_node = claim2node[str(claim_sent)]
        premises = claim2premises[cur_claim_node]
        premises_sorted = sorted(premises, key = lambda x: x.start)
        for premise in premises_sorted:
            premise_node = premise2node[str(premise)]
            snode = ag.Node(
                key=query_count + "-" + str(graph.keygen()) + str(uuid.uuid4().hex),
                text="Default Inference",
                category=ag.NodeCategory.RA,
            )
            # TODO
            #graph.add_edge(ag.Edge(key=query_count + "-" + str(graph.keygen()), start=premise_node, end=snode))
            #graph.add_edge(ag.Edge(key=query_count + "-" + str(graph.keygen()), start=snode, end=cur_claim_node))
            graph.add_edge(ag.Edge(key=query_count + "-" + str(graph.keygen()) + str(uuid.uuid4().hex), start=snode, end=premise_node))
            graph.add_edge(ag.Edge(key=query_count + "-" + str(graph.keygen()) + str(uuid.uuid4().hex), start=cur_claim_node, end=snode))
    return claim2node, claim_sents_ordered, graph



def find_parent(parent,i):
    if parent[i] == -1:
        return i
    if parent[i]!= -1:
         return find_parent(parent,parent[i])

def run_combine(claim2node, claim_sents):
    graph = ag.Graph(name='final')
    min_threshold = 0.33
    claim2parent = {}
    parent = [-1 for claim in claim_sents]
    todo = []
    for claim_count, claim_1 in enumerate(claim_sents):
        claim2parent[str(claim_1)] = str(claim_1)
        for claim_2 in claim_sents:
            if claim_1 == claim_2:
                continue
            try:
                todo.append((claim_1.text, claim_2.text))
            except:
                todo.append((claim_1, claim_2))

    if len(todo) > 0:
        logprobs = get_probs(todo)
        curindex = 0
        claim2conflict = {}
        for claim_count, claim_premise in enumerate(claim_sents):
            cur_logs = logprobs[curindex: curindex + len(claim_sents) - 1]
            curindex += len(claim_sents) - 1
            max_support_index = np.argmax([x[2] for x in cur_logs])
            max_attack_index = np.argmax([x[0] for x in cur_logs])
            max_attack = cur_logs[max_attack_index][0]
            max_support = cur_logs[max_support_index][2]
            #print(max_attack, max_support)
            max_overall = max_support
            max_attack = -1
            if max_overall >= min_threshold:
                if max_attack > max_support:
                    print("error")
                    if max_attack_index >= claim_count:
                        max_attack_index += 1
                    claim_claim = claim_sents[max_attack_index]
                else:
                    if max_support_index >= claim_count:
                        max_support_index += 1
                    claim_claim = claim_sents[max_support_index]
                #cur_claim_node = claim2node[str(cur_claim)]
                #source_claim = claim2node[str(claim_1)]
                claim_premise_node = claim2node[str(claim_premise)]
                claim_claim_node = claim2node[str(claim_claim)]
                # TODO
                #if claim2parent[str(cur_claim)] == str(claim_1):
                claim_premise_set  = find_parent(parent, claim_count)
                claim_claim_set = find_parent(parent, max_support_index)
                #if claim2parent[str(claim_claim)] == str(claim_premise):
                if claim_claim_set == claim_premise_set:
                    continue
                parent[claim_premise_set] = claim_claim_set
                # TODO
                #claim2parent[str(claim_1)] = str(cur_claim)
                claim2parent[str(claim_premise)] = str(claim_claim)
                # TODO
                #graph.add_edge(ag.Edge(key=graph.keygen(), start=source_claim, end=cur_claim_node))
                graph.add_edge(ag.Edge(key=str(graph.keygen()) + "-" +  str(uuid.uuid4().hex), start=claim_claim_node, end=claim_premise_node))

    parent_nodes = []
    for claim_count, claim in enumerate(claim_sents):
        claim_set = find_parent(parent, claim_count)
        if claim_set in parent_nodes:
            continue
        parent_nodes.append(claim_set)
        #orig_claim = str(claim)
        #parent = claim2parent[orig_claim]
        #while orig_claim != parent:
        #    orig_claim = parent
        #    parent = claim2parent[parent]
        #parent_nodes.append(parent)
    parent_nodes = sorted(parent_nodes)
    try:
        parent_nodes = [claim_sents[x].text for x in parent_nodes]
    except:
        parent_nodes = [claim_sents[x] for x in parent_nodes]
    #print("PARENTS!!!")
    #parent_nodes = list(set(parent_nodes))
    #for ex in parent_nodes:
    #    print(ex)
    todo = []
    for claim_1 in parent_nodes:
        claim2parent[str(claim_1)] = str(claim_1)
        for claim_2 in parent_nodes:
            if claim_1 == claim_2:
                continue
            todo.append((claim_1, claim_2))
    potential_edges = []
    claims_done = []
    tmp_nodes = []
    if len(todo) > 0:
        logprobs = get_probs(todo)
        curindex = 0
        claim2conflict = {}
        for claim_count, claim_1 in enumerate(parent_nodes):
            cur_logs = logprobs[curindex: curindex + len(parent_nodes) - 1]
            curindex += len(parent_nodes) - 1
            max_support_index = np.argmax([x[2] for x in cur_logs])
            max_attack_index = np.argmax([x[0] for x in cur_logs])
            max_attack = cur_logs[max_attack_index][0]
            max_support = cur_logs[max_support_index][2]
            max_overall = max_attack
            max_support = -1
            if max_overall >= min_threshold:
                if max_attack > max_support:
                    if max_attack_index >= claim_count:
                        max_attack_index += 1
                    cur_claim = parent_nodes[max_attack_index]
                else:
                    if max_support_index >= claim_count:
                        max_support_index += 1
                    cur_claim = parent_nodes[max_support_index]
                source_claim = claim2node[str(claim_1)]
                cur_claim_node = claim2node[str(cur_claim)]
                tmp_node = ag.Node(key=str(graph.keygen()) + "-" + str(uuid.uuid4().hex), text="Issue", category=ag.NodeCategory.I)
                tmp_nodes.append(tmp_node)
                # TODO
                #node_1 = ag.Edge(key=graph.keygen(), start=source_claim, end=tmp_node)
                #node_2 = ag.Edge(key=graph.keygen(), start=cur_claim_node, end=tmp_node)
                node_1 = ag.Edge(key=str(graph.keygen()) + "-" + str(uuid.uuid4().hex), start=tmp_node, end=source_claim)
                node_2 = ag.Edge(key=str(graph.keygen()) + "-" + str(uuid.uuid4().hex), start=tmp_node, end=cur_claim_node)
                potential_edges.append((node_1, node_2, max_attack))
    used_nodes = []
    sorted_potential_edges = sorted(potential_edges, key = lambda x: x[2], reverse=True)
    #print(potential_edges)
    for edge in sorted_potential_edges:
        node_1, node_2, score = edge
        #TODO
        #if node_1.start.text in used_nodes or node_2.start.text in used_nodes:
        #used_nodes.append(node_1.start.text)
        #used_nodes.append(node_2.start.text)
        #tmp_nodes.append(node_1.end)
        #claims_done.append(str(node_1.start))
        #claims_done.append(str(node_2.start))
        if node_1.end.text in used_nodes or node_2.end.text in used_nodes:
            continue
        used_nodes.append(node_1.end.text)
        used_nodes.append(node_2.end.text)
        #tmp_nodes.append(node_1.start)
        claims_done.append(str(node_1.end.text))
        claims_done.append(str(node_2.end.text))
        graph.add_edge(node_1)
        graph.add_edge(node_2)
        #print(score)

    conversation_node = ag.Node(key=str(graph.keygen()) + "-"  + str(uuid.uuid4().hex), text="Conversation", category=ag.NodeCategory.I)
    for claim_count, claim_1 in enumerate(parent_nodes):
        if str(claim_1) in claims_done:
            continue
        cur_claim_node = claim2node[str(claim_1)]
        tmp_nodes.append(cur_claim_node)
    for node in tmp_nodes:
        # TODO
        #edge_1 = ag.Edge(key=graph.keygen(), start=node, end=conversation_node)
        edge_1 = ag.Edge(key=str(graph.keygen()) + "-" + str(str(uuid.uuid4().hex)), start=conversation_node, end=node)
        graph.add_edge(edge_1)
    return graph, conversation_node
