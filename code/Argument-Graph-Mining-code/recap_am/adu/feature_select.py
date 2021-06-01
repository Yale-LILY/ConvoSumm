import logging
import pickle
import copy
import os
from sklearn.feature_selection import (
    mutual_info_classif,
    SelectKBest,
    VarianceThreshold,
)
from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler,
    QuantileTransformer,
    PowerTransformer,
    StandardScaler,
)
from sklearn.decomposition import PCA
import multiprocessing

from recap_am.model.config import Config

config = Config.get_instance()

lang = config["nlp"]["language"]

featdir = config["adu"]["feature_selection"]["dir"] + "/" + lang
if not os.path.isdir(featdir):
    os.makedirs(featdir)


def variance_threshold(input_doc, load=True):
    """Apply variance threshold selection to features."""
    vt_path = featdir + "/final/"
    # vt_path = featdir
    if load:
        with open(vt_path + "VT.pkl", "rb") as f:
            VT = pickle.load(f)
        f.close()
        input_doc._.Features = VT.transform(input_doc._.Features)
    else:
        VT = VarianceThreshold().fit(input_doc._.Features)
        input_doc._.Features = VT.transform(input_doc._.Features)
        with open(vt_path + "VT.pkl", "wb") as f:
            pickle.dump(VT, f)
        f.close()
    return input_doc


def load_scaler(use_scaler=None):
    """a"""
    if use_scaler:
        method = use_scaler
    else:
        method = config["adu"]["feature_selection"]["scaling_method"]
    if method == "Robust":
        scaler = RobustScaler()
    elif method == "Power":
        scaler = PowerTransformer()
    elif method == "MinMax":
        scaler = MinMaxScaler()
    elif method == "Standard":
        scaler = StandardScaler()
    elif method == "QuantileUniform":
        scaler = QuantileTransformer(output_distribution="uniform")
    elif method == "QuantileGaussian":
        scaler = QuantileTransformer(output_distribution="normal")
    else:
        exit(1)
    return scaler


def scale(input_doc, load=True, use_scaler=None):
    """Apply scaling to features."""
    if use_scaler:
        method = use_scaler
    else:
        method = config["adu"]["feature_selection"]["scaling_method"]
    scale_path = featdir + "/final/" + method + ".pkl"
    if not os.path.isdir(featdir + "/scaler/"):
        os.makedirs(featdir + "/scaler/")
    if load:
        with open(scale_path, "rb") as f:
            scaler = pickle.load(f)
        f.close()
        input_doc._.Features = scaler.transform(input_doc._.Features)
    else:
        scaler = load_scaler(use_scaler=use_scaler)
        print("Loaded scaler:\t%s" % (use_scaler))
        scaler.fit(input_doc._.Features)
        print("fitted scaler:\t%s" % (use_scaler))
        input_doc._.Features = scaler.transform(input_doc._.Features)
        print("transformed input w \t%s" % (use_scaler))
        with open(scale_path, "wb") as f:
            pickle.dump(scaler, f)
        f.close()
        print("dumped\t%s" % (use_scaler))
    return input_doc


def select(input_doc, scaler):
    nfeats = round(len(input_doc._.Features[0]) * 0.1)
    Selector = SelectKBest(score_func=mutual_info_classif, k=nfeats).fit(
        input_doc._.Features, input_doc._.Labels
    )
    Features = Selector.transform(input_doc._.Features)
    a_path = featdir + "/Selector" + "_" + scaler + ".pkl"
    if not os.path.isdir(featdir + "/Selector/"):
        os.makedirs(featdir + "/Selector/")
    with open(a_path, "wb") as f:
        pickle.dump(Selector, f)
    f.close()
    return Features


def select_kbest(input_doc, load=True, scaler="None"):
    """Apply SelectKBest feature selection to features."""
    if scaler == "None":
        scaler = config["adu"]["feature_selection"]["scaling_method"]
    if load:
        a_path = featdir + "/Selector" + "_" + scaler + ".pkl"
        with open(a_path, "rb") as f:
            Selector = pickle.load(f)
        input_doc._.Features = Selector.transform(input_doc._.Features)
        return input_doc
    else:
        input_doc._.Features = select(input_doc, scaler)
        return input_doc


def pca(input_doc, load=True, scaler="None"):
    """Apply PCA decomposition to features."""
    if scaler == "None":
        scaler = config["adu"]["feature_selection"]["scaling_method"]
    if load:
        path = featdir + "/PCA" + "_" + scaler + ".pkl"
        with open(path, "rb") as f:
            pca_model = pickle.load(f)
        f.close()
        input_doc._.Features = pca_model.transform(input_doc._.Features)
    else:
        pca_model = PCA(n_components=0.95).fit(input_doc._.Features)
        input_doc._.Features = pca_model.transform(input_doc._.Features)
        path = featdir + "/PCA" + "_" + scaler + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(pca_model, f)
        f.close()
    return input_doc


def concating(lists):
    l0 = []
    l = lists[0]
    ll = lists[1]
    for idx, a in enumerate(l):
        if isinstance(a, list):
            l0.append(a.extend(ll[idx]))
        else:
            l0.append(a.tolist().extend(ll[idx]))
    return l0


def concat_feats(input_doc, doc1, doc2):
    """Concatenate features from two docs after they were selected by different methods."""
    a1 = doc1._.Features
    a2 = doc2._.Features
    listings = [a1, a2]

    result = concating(listings)
    input_doc._.Features = result

    return input_doc


def add_embeds(listing, input_doc):
    logging.debug("adding embeds")
    for idx, feat in enumerate(listing):
        if isinstance(feat, list):
            feat.extend(input_doc._.embeddings[idx])
        else:
            feat.tolist().extend(input_doc._.embeddings[idx])
    return listing


def add_embeddings(input_doc):
    """Add sentence embeddings after feature selection."""
    nlist = input_doc._.Features
    nlist = add_embeds(nlist, input_doc)
    input_doc._.Features = nlist
    logging.debug("Added embeddings to doc")
    return input_doc


def pca_selectk_union(input_doc, load=True, scaler="None"):
    """Perform PCA and select_kbest independently and concatenate features for classification."""
    pca_doc = copy.deepcopy(input_doc)
    select_doc = copy.deepcopy(input_doc)

    pca_doc = pca(pca_doc, load=load, scaler=scaler)
    select_doc = select_kbest(select_doc, load=load, scaler=scaler)

    input_doc = concat_feats(input_doc, pca_doc, select_doc)

    return input_doc


methods_dict = dict()
methods_dict["VT"] = variance_threshold
methods_dict["scale"] = scale
methods_dict["PCA"] = pca
methods_dict["select_kbest"] = select_kbest
methods_dict["PCA_selectk_union"] = pca_selectk_union


def filter_feats(input_doc, load):
    """Apply feature selection of doc."""
    methods = config["adu"]["feature_selection"]["methods"]
    for method in methods:
        input_doc = methods_dict[method](input_doc, load=load)
    return input_doc


def call_pca(doc, scaler):
    _ = pca(doc, load=False, scaler=scaler)
    return


def call_select(doc, scaler):
    _ = select_kbest(doc, load=False, scaler=scaler)
    return


def call_pca_select(doc, scaler):
    _ = pca_selectk_union(doc, load=False, scaler=scaler)
    return


def loop(input_doc, scaler):
    print("Scaler:\t%s" % (scaler))
    sub_doc = scale(copy.deepcopy(input_doc), load=False, use_scaler=scaler)
    print("Scaled")
    p1 = multiprocessing.Process(target=call_pca, args=(copy.deepcopy(sub_doc), scaler))
    p1.start()
    print("PCA")
    p2 = multiprocessing.Process(
        target=call_select, args=(copy.deepcopy(sub_doc), scaler)
    )
    p2.start()
    print("SelectKbest")
    p3 = multiprocessing.Process(
        target=call_pca_select, args=(copy.deepcopy(sub_doc), scaler)
    )
    p3.start()
    jobs2 = [p1, p2, p3]
    print("PCASelectKbest")
    for proc in jobs2:
        proc.join()
    return


def full_run(input_doc):
    print("Started full run")
    print(len(input_doc._.Features[0]))
    input_doc = variance_threshold(input_doc, load=True)
    print(len(input_doc._.Features[0]))
    print("Variance Threshold Done")
    scalers = ["QuantileGaussian"]
    jobs = []
    for scaler in scalers:
        print("Beginning:\t%s" % (scaler))
        p = multiprocessing.Process(target=loop, args=(input_doc, scaler))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    return
