import pickle
import string
import numpy as np
from nltk import Tree
from spacy.tokens import Doc, Span
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer, models
from recap_am.model.config import Config

config = Config.get_instance()

lang = config["nlp"]["language"]
auxdir = config["adu"]["auxiliary"]["dir"]
vocab_path = auxdir + "/" + lang + "/" + config["adu"]["auxiliary"]["vocab"]
freqdist_path = auxdir + "/" + lang + "/" + config["adu"]["auxiliary"]["freqdist"]

if lang == "en":
    stemmer = SnowballStemmer("english")
elif lang == "de":
    stemmer = SnowballStemmer("german")

embedder = None
if config["nlp"]["embeddings"] == "sentence_transformers":
    if lang == "de":
        embedder = SentenceTransformer("distiluse-base-multilingual-cased")
    elif lang == "en":
        embedder = SentenceTransformer("roberta-large-nli-stsb-mean-tokens")


def sentence_embedding(doc):
    """Returns list of sentence embeddings, each as a list."""
    return [embedding for embedding in doc._.embedding_list]


def set_embedding_list(doc, value):
    doc._.embedding_list = value


Doc.set_extension("embedding_list", default=list())


def embedding(span):
    """Returns custom or default sentence embedding as list."""
    if config["nlp"]["embeddings"] == "spacy":
        embed = span.vector.tolist()
    elif config["nlp"]["embeddings"] == "sentence_transformers":
        embed = list(np.mean(embedder.encode(str(span))), axis=1)
    return embed


def get_freqdist(doc):
    """Returns list of frequency vectors."""
    return [s._.freq for s in doc.sents]


def freq(span):
    """Returns ngram frequency vector."""
    sd = [0] * len(vocab)
    text = str(span).translate(str.maketrans("", "", string.punctuation))
    tok = [stemmer.stem(token) for token in text.split(" ")]
    for idx, word in enumerate(vocab):
        if word in tok:
            sd[idx] = freqdist[word]
    return tuple(sd)


claim_indicators = [
    "accordingly",
    "as a result",
    "consequently",
    "conclude that",
    "clearly",
    "demonstrates that",
    "entails",
    "follows that",
    "hence",
    "however",
    "implies",
    "in fact",
    "in my opinion",
    "in short",
    "in conclusion",
    "indicates that",
    "it follows that",
    "it is highly probable that",
    "it is my contention",
    "it should be clear that",
    "I believe",
    "I mean",
    "I think",
    "must be that",
    "on the contrary",
    "points to the conclusions",
    "proves that",
    "shows that",
    "so",
    "suggests that",
    "the most obvious explanation",
    "the point I’m trying to make",
    "therefore",
    "thus",
    "the truth of the matter",
    "to sum up",
    "we may deduce",
]

de_claim_indicators = [
    "dementsprechend",
    "als Ergebnis",
    "folglich",
    "schliessen, dass",
    "offensichtlich",
    "zeigt, dass",
    "hat zur Folge",
    "folgt, dass",
    "daher",
    "jedoch",
    "besagt",
    "tatsächlich",
    "meiner Meinung nach",
    "kurz gesagt",
    "schiesslich",
    "deutet darauf hin, dass",
    "es folgt, dass",
    "es ist sehr wahrscheinlich, dass",
    "es ist mein Argument",
    "es sollte klar sein, dass",
    "ich glaube",
    "ich meine",
    "ich denke",
    "muss",
    "hingegen",
    "deutet auf die schlussfolgerung",
    "beweist, dass",
    "zeigt, dass",
    "also",
    "meint, dass",
    "die offensichtlichste Erklärung",
    "was ich damit sagen will",
    "deshalb",
    "folglich",
    "in Wirklichkeit",
    "zusammenfassend",
    "können wir folgern",
]

premise_indicators = [
    "after all",
    "assuming that",
    "as",
    "as indicated by",
    "as shown",
    "besides",
    "because",
    "deduced",
    "derived from",
    "due to",
    "firstly",
    "follows from",
    "for",
    "for example",
    "for instance",
    "for one thing",
    "for the reason that",
    "furthermore",
    "given that",
    "in addition",
    "in light of",
    "in that",
    "in view of",
    "in view of the fact that",
    "indicated by",
    "is supported by",
    "may be inferred",
    "moreover",
    "owing to",
    "researchers found that",
    "secondly",
    "this can be seen from",
    "since",
    "since the evidence is",
    "what’s more",
    "whereas",
]

de_premise_indicators = [
    "im Grunde genommen",
    "angenommen, dass",
    "da",
    "wie angedeutet von",
    "wie gezeigt",
    "ausserdem",
    "weil",
    "abgeleitet",
    "hergeleitet von",
    "durch",
    "zunächst",
    "folgt aus",
    "zum",
    "zum Beispiel",
    "beispielsweise",
    "zum Einen",
    "aus dem Grund, dass",
    "weiterhin",
    "gegeben, dass",
    "zusätzlich",
    "angesichts dessen",
    "insofern, als",
    "im Hinblick auf",
    "angesichts der Tatsache",
    "angedeutet von",
    "ist unterstützt durch",
    "kann gefolgert werden",
    "ferner",
    "aufgrund",
    "Wissenschaftler fanden heraus, dass",
    "zum Anderen",
    "dies kann gesehen werden, durch",
    "denn",
    "da der Beweis",
    "darüber hinaus",
    "wohingegen",
    ",dass",
]


def get_cl_indicators(doc):
    """Returns of claim indicator values."""
    return [s._.cl_indicator for s in doc.sents]


def get_pr_indicators(doc):
    """Returns of premise indicator values."""
    return [s._.pr_indicator for s in doc.sents]


def has_cl_indicator(span):
    """Return if sentence contains a claim indicator."""
    ind_list = de_claim_indicators if lang == "de" else claim_indicators
    for token in span:
        if token.text in ind_list:
            return 1
    return 0


def has_pr_indicator(span):
    """Return if sentence contains a premise indicator."""
    ind_list = de_premise_indicators if lang == "de" else premise_indicators
    for token in span:
        if token.text in ind_list:
            return 1
    return 0


def get_sent_lengths(doc):
    """Return list of sentence lengths."""
    return [len(s) for s in doc.sents]


def get_paragraphs(doc):
    """Return list of paragraph values of sentences, assuming each line is one paragraph."""
    par = [0] * len(list(doc.sents))
    lines = doc.text.splitlines()
    for idx, line in enumerate(lines):
        for id2, s in enumerate(doc._.sentences):
            if str(s) in line:
                par[id2] = idx
    return par


def get_sent_paragraph(span):
    """Returns the paragraph of the sentence."""
    doc = span.doc
    pars = span.doc._.paragraphs
    for idx, s in enumerate(doc._.sentences):
        if s == span:
            return pars[idx]
    return 0


def is_head(span):
    """Returns if the sentence is in the first or second paragraph."""
    if span._.paragraph == 1 or span._.paragraph == 0:
        return 1
    else:
        return 0


def head(doc):
    """Returns list of head values of sentences."""
    return [s._.is_head for s in doc.sents]


def is_tail(span):
    """Returns if the sentence is in the last paragraph."""
    mpar = max(span.doc._.paragraphs)
    if span._.paragraph == mpar:
        return 1
    else:
        return 0


def tail(doc):
    """Returns list of tail values of sentences."""
    return [s._.is_tail for s in doc.sents]


def is_paragraph_head(span):
    """Returns if sentence is the first sentence of a paragraph."""
    sents = span.doc.sents
    spar = span._.paragraph
    res = 1
    for s in sents:
        if s._.paragraph == spar:
            if s.start < span.start:
                res = 0
    return res


def paragraph_heads(doc):
    """Returns list of paragraph head values of sentences"""
    return [s._.is_par_head for s in doc.sents]


def is_paragraph_tail(span):
    """Returns if sentence is the last sentence of a paragraph."""
    sents = span.doc.sents
    spar = span._.paragraph
    res = 1
    for s in sents:
        if s._.paragraph == spar:
            if s.start > span.start:
                res = 0
    return res


def paragraph_tails(doc):
    """Returns list of paragraph tail values of sentences."""
    return [s._.is_par_tail for s in doc.sents]


def get_sent_pos(doc):
    """Returns list of sentence positions equivalent to their index."""
    return [s._.index for s in doc.sents]


punctuation = set(string.punctuation)


def get_punctuation_count(doc):
    """Returns list of punctuation values of sentences."""
    return [s._.punct for s in doc.sents]


def get_punct(span):
    """Returns number of punctuation marks in sentence."""
    return sum([1 for token in span if token.text in punctuation])


def get_question_marker(doc):
    """Returns list of question values of sentences."""
    return [s._.is_question for s in doc.sents]


def is_question(span):
    """Returns if sentence is a question by determining if it contains a question mark."""
    if "?" in [token.text for token in span]:
        return 1
    else:
        return 0


personal = ["me", "myself", "I", "my"]
de_personal = ["mich", "mir", "ich", "meine", "meiner", "mir selbst"]


def has_personal(span):
    """Returns if the sentence contains personal pronomena."""
    p_list = de_personal if lang == "de" else personal
    for token in span:
        if token.text in p_list:
            return 1
    return 0


def get_personal(doc):
    """Returns list of personal values of sentences."""
    return [s._.has_personal for s in doc.sents]


def get_modal(doc):
    """Returns list of modal values of sentences."""
    return [s._.has_modal for s in doc.sents]


def has_modal(span):
    """Returns if a sentence contains a modal verb."""
    for token in span:
        if token.tag_ == "MD":
            return 1
    return 0


def tree_depth(span):
    """Returns the depth of the dependency tree of a sentence."""
    i = 1
    d = 1
    obj = span.root
    children = obj.children
    children_left = True
    while children_left == True:
        clist = []
        i += 1
        for c in children:
            if len(list(c.children)) > 0:
                for c2 in c.children:
                    clist.append(c2)
                d = i
        if len(clist) == 0:
            children_left = False
        else:
            children = clist
    return d


def get_tree_depths(doc):
    """Returns list of tree depths of sentences."""
    return [s._.tree_depth for s in doc.sents]


def to_nltk_tree(node):
    """Transforms spacy dependency structure to nltk dependency tree."""
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def get_production_count(span):
    """Returns number of productions in dependency tree of sentence."""
    root = span.root
    tree = to_nltk_tree(root)
    if isinstance(tree, str):
        return 1
    else:
        prod = tree.productions()
        return len(prod)


def get_prod_count(doc):
    """Returns list of production values of sentences."""
    return [s._.prod_count for s in doc.sents]


Span.set_extension("embedding", getter=embedding)
Span.set_extension("freq", getter=freq)
Span.set_extension("cl_indicator", getter=has_cl_indicator)
Span.set_extension("pr_indicator", getter=has_pr_indicator)
Span.set_extension("punct", getter=get_punct)
Span.set_extension("is_question", getter=is_question)
Span.set_extension("has_personal", getter=has_personal)
Span.set_extension("has_modal", getter=has_modal)
Span.set_extension("tree_depth", getter=tree_depth)
Span.set_extension("prod_count", getter=get_production_count)
Span.set_extension("paragraph", getter=get_sent_paragraph)
Span.set_extension("is_head", getter=is_head)
Span.set_extension("is_tail", getter=is_tail)
Span.set_extension("is_par_head", getter=is_paragraph_head)
Span.set_extension("is_par_tail", getter=is_paragraph_tail)

Doc.set_extension("embeddings", getter=sentence_embedding, setter=set_embedding_list)
Doc.set_extension("freqdist", getter=get_freqdist)
Doc.set_extension("cl_indicators", getter=get_cl_indicators)
Doc.set_extension("pr_indicators", getter=get_pr_indicators)
Doc.set_extension("punctcount", getter=get_punctuation_count)
Doc.set_extension("questions", getter=get_question_marker)
Doc.set_extension("personals", getter=get_personal)
Doc.set_extension("modals", getter=get_modal)
Doc.set_extension("tree_depths", getter=get_tree_depths)
Doc.set_extension("prodcount", getter=get_prod_count)
Doc.set_extension("paragraphs", getter=get_paragraphs)
# Doc.set_extension("tfidf", getter=get_tfidf)
Doc.set_extension("sentence_positions", getter=get_sent_pos)
Doc.set_extension("par_tails", getter=paragraph_tails)
Doc.set_extension("par_heads", getter=paragraph_heads)
Doc.set_extension("tail", getter=tail)
Doc.set_extension("head", getter=head)
Doc.set_extension("sentlengths", getter=get_sent_lengths)

Doc.set_extension("Features", default=[])
Doc.set_extension("CLPR_Features", default=[])


def flat2gen(alist):
    """Transforms nested list to flat list."""
    for item in alist:
        if isinstance(item, tuple):
            for subitem in item:
                yield subitem
        elif isinstance(item, list):
            for subitem in item:
                yield subitem
        else:
            yield item


def set_features(doc):
    doc._.embeddings = [s._.embedding for s in doc.sents]
    """Zips all extracted features to large vector and returns doc."""

    first_list = list(
        zip(
            doc._.cl_indicators,
            doc._.pr_indicators,
            doc._.punctcount,
            doc._.questions,
            doc._.personals,
            doc._.modals,
            doc._.tree_depths,
            doc._.prodcount,
            doc._.sentence_positions,
            doc._.sentlengths,
        )
    )
    new_list = []
    for feat in first_list:
        new_list.append(list(flat2gen(feat)))
    doc._.Features = new_list
    return doc
