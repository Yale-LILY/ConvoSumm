import re
import os

from nltk import PunktSentenceTokenizer
from spacy.tokens import Doc, Span, Token
import multiprocessing
import itertools
import numpy as np

from recap_am.controller.extract_features import set_features
from recap_am.controller.nlp import parse
from recap_am.model.config import Config

config = Config.get_instance()

# nltk.download("punkt")

lang = config["nlp"]["language"]


def clean_text(text):
    text = re.sub(r"&nbsp;[a-zA-Z0-9]?", "", text)
    text = (
        text.replace("Art.", "Artikel")
        .replace("Abs.", "Absatz")
        .replace("u.a.", "unter anderem")
        .replace("U.a.", "Unter anderem")
        .replace("u.E.", "unseres Erachtens")
        .replace("U.E.", "Unseres Erachtens")
        .replace("vgl.", "vergleiche")
        .replace("Vgl.", "Vergleiche")
        .replace("bzw.", "beziehungsweise")
        .replace("i.V.m.", "im Vergleich mit")
        .replace("Buchst.", "Buchstabe")
        .replace("d.h.", "das heißt")
        .replace("'", "")
        .replace("-", " ")
        .replace(";", "")
    )
    text = re.sub(r"[^a-zA-Z0-9.,?!äÄöÖüÜ:;&ß%$'\"()[\]{} -]\n", "", text)
    text = text.replace("...", "")
    text = re.sub(r" +", " ", text)
    text = text.strip(" ")
    return text


def pre_segment(doc):
    """Set sentence boundaries with nltk instead of spacy."""
    if len(str(doc.text).split()) > 3:
        tokenizer = PunktSentenceTokenizer(doc.text)
        sentences = tokenizer.tokenize(doc.text)
        for nltk_sentence in sentences:
            words = re.findall(r"[\w]+|[^\s\w]", nltk_sentence)
            for i in range(len(doc) - len(words) + 1):
                token_list = [str(token) for token in doc[i : i + len(words)]]
                if token_list == words:
                    doc[i].is_sent_start = True
                    for token in doc[i + 1 : i + len(words)]:
                        token.is_sent_start = False
    return doc


parse.add_pipe(pre_segment, before="parser")


def get_sentences(doc):
    """Return list of sentences."""
    return list(doc.sents)


def add_labels(doc, labels):
    """Add labels from list to doc."""
    adu_labels_list = []
    clpr_label_list = []
    for idx, label in enumerate(labels):
        label = label.strip("\n").strip(" ")
        if label == "Claim":
            adu_labels_list.append(1)
            clpr_label_list.append(1)
        elif label == "Premise":
            adu_labels_list.append(1)
            clpr_label_list.append(0)
        elif label == "MajorClaim":
            adu_labels_list.append(1)
            clpr_label_list.append(1)
        elif label == "None":
            adu_labels_list.append(0)
        elif label == "ADU":
            adu_labels_list.append(1)
        elif label == "1":
            adu_labels_list.append(1)
        elif label == "0":
            adu_labels_list.append(0)
    if len(adu_labels_list) > len(doc._.Features):
        adu_labels_list = adu_labels_list[: len(doc._.Features)]
    elif len(adu_labels_list) < len(doc._.Features):
        add_on = np.random.randint(
            low=0, high=1, size=len(doc._.Features) - len(adu_labels_list)
        ).tolist()
        adu_labels_list.extend(add_on)
    nr_adus = sum([1 for l in adu_labels_list if l == 1])
    if len(clpr_label_list) > nr_adus:
        clpr_label_list = clpr_label_list[:nr_adus]
    elif len(clpr_label_list) < nr_adus:
        add_on = np.random.randint(
            low=0, high=1, size=nr_adus - len(clpr_label_list)
        ).tolist()
        clpr_label_list.extend(add_on)
    doc._.Labels = adu_labels_list
    doc._.CLPR_Labels = clpr_label_list
    return doc


def get_token_label(token):
    """Return token label for specified task."""
    label_list = token.doc._.Labels
    for idx, sent in enumerate(token.doc.sents):
        if idx + 1 < len(list(token.doc.sents)):
            if token.i >= sent.start and token.i < list(token.doc.sents)[idx + 1].start:
                return label_list[idx]
        else:
            return label_list[idx]


def get_sentence_label(span):
    """Return sentence label."""
    return span.doc._.Labels[span._.index]


def get_index(span):
    """Return index of sentence in doc."""
    for idx, s in enumerate(span.doc.sents):
        if span == s:
            return idx


def set_empty_labels(doc):
    """Set labels to zero for each sentence."""
    labels = [0] * len(list(doc.sents))
    doc._.Labels = labels
    doc._.CLPR_Labels = labels
    return doc


def get_ADU(doc, mc=False):
    """Return all sentences labeled as ADU."""
    adu = doc._.Labels
    result = []
    for idx, s in enumerate(doc._.sentences):
        if adu[idx] == 1:
            result.append(s)
    return result


def get_CL(doc, mc=False):
    """Return all sentences labeled as ADU but not as majorclaim."""
    adu = doc._.CLPR_Labels
    result = []
    for idx, s in enumerate(doc._.ADU_Sents):
        if adu[idx] == 1:
            result.append(s)
    return result


def get_PR(doc, mc=False):
    """Return all sentences labeled as ADU but not as majorclaim."""
    adu = doc._.CLPR_Labels
    result = []
    for idx, s in enumerate(doc._.ADU_Sents):
        if adu[idx] == 0:
            result.append(s)
    return result


def get_features(span):
    return span.doc._.Features[span._.index]


def get_mc(doc):
    for idx, val in enumerate(list(doc.sents)):
        if doc._.MC_List[idx] == 1:
            return val


Span.set_extension("Label", getter=get_sentence_label)
Span.set_extension("CLPR_Label", getter=get_sentence_label)
Span.set_extension("index", getter=get_index)
Span.set_extension("Feature", getter=get_features)
Span.set_extension("mc", default=0)

Token.set_extension("Label", getter=get_token_label)

Doc.set_extension("ADU_Sents", getter=get_ADU)
Doc.set_extension("Claim_Sents", getter=get_CL)
Doc.set_extension("Premise_Sents", getter=get_PR)
Doc.set_extension("MC_List", default=[])
Doc.set_extension("MajorClaim", getter=get_mc)
Doc.set_extension("sentences", getter=get_sentences)
Doc.set_extension("Labels", default=[0])
Doc.set_extension("CLPR_Labels", default=[0])


def prep_production(filename, input_text):
    """Prepare single document for classification."""
    input_text = clean_text(input_text)
    doc = parse(input_text)
    doc._.key = filename
    set_features(doc)
    set_empty_labels(doc)
    return doc


def merge_docs(doc_list):
    """Merge multiple parsed docs into one."""

    comb_feat = list(
        itertools.chain.from_iterable(list(map(lambda x: x._.Features, doc_list)))
    )
    comb_label = list(
        itertools.chain.from_iterable(list(map(lambda x: x._.Labels, doc_list)))
    )
    comb_clpr_label = list(
        itertools.chain.from_iterable(list(map(lambda x: x._.CLPR_Labels, doc_list)))
    )
    comb_embedding = list(
        itertools.chain.from_iterable(list(map(lambda x: x._.embeddings, doc_list)))
    )

    final_text = "FinalDocument"
    final_doc = parse(final_text)
    final_doc._.Features = comb_feat
    final_doc._.Labels = comb_label
    final_doc._.CLPR_Labels = comb_clpr_label
    final_doc._.embeddings = comb_embedding

    print("Merged Lists")
    return final_doc


def prep_training(filename, input_text, labels_list):
    """Prepare a single file for training or testing."""
    doc = parse(input_text)
    doc._.key = filename
    doc = set_features(doc)
    doc = add_labels(doc, labels_list)
    return doc


if config["debug"]:
    texts = list()
    label_list = list()
else:
    manager = multiprocessing.Manager()
    texts = manager.list()
    label_list = manager.list()


def read_in(file_name1, file_name2):
    if os.path.isfile(file_name1):
        with open(file_name1, "r+", encoding="utf8", errors="ignore",) as f:
            text = f.read()
        f.close()
        with open(file_name2, "r+", encoding="utf8", errors="ignore",) as f:
            labels = f.read().split("\n")
        f.close()
    else:
        with open(
            config["adu"]["path"]["input"] + "/" + file_name1,
            "r+",
            encoding="utf8",
            errors="ignore",
        ) as f:
            text = f.read()
        f.close()
        with open(
            config["adu"]["path"]["label"] + "/" + file_name2,
            "r+",
            encoding="utf8",
            errors="ignore",
        ) as f:
            labels = f.read().split("\n")
        f.close()
    text = clean_text(text)
    texts.append(text)
    label_list.append(labels)
    return


def read_files(input_list, label):
    """Read files from directory, merge and prepare for classification."""
    if isinstance(input_list, list) or isinstance(input_list, tuple):
        jobs = []
        for idx, infile in enumerate(input_list):
            print("Reading Document\t%s" % (infile))
            p = multiprocessing.Process(target=read_in, args=(infile, label[idx]))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        jobs = []
        doc_list = []
        for idx, doc in enumerate(
            parse.pipe(
                texts,
                disable=["ner"],
                batch_size=80,
                n_process=multiprocessing.cpu_count(),
            )
        ):
            print("Processing Document\t%i" % (idx))
            doc._.key = input_list[idx]
            doc = set_features(doc)
            doc = add_labels(doc, label_list[idx])
            doc_list.append(doc)
        final_doc = merge_docs(doc_list)
        texts[:] = []
        label_list[:] = []
    else:
        with open(input_list, "r+", encoding="utf8") as f:
            text = f.read()
        f.close()
        text = clean_text(text)
        with open(label, "r+", encoding="utf8") as f:
            labels = f.read().split("\n")
        f.close()
        final_doc = prep_training(input_list, text, labels)
    return final_doc
