import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    RandomizedSearchCV,
)
from sklearn.metrics.pairwise import cosine_similarity
from recap_am.adu import utilities
from recap_am.model.config import Config



config = Config.get_instance()


def fit_model(input_doc):
    """Load and train model."""
    model = utilities.get_model()
    model = train_model(model, input_doc)
    return model


def fit_clpr_model(input_doc):
    """Load and train model."""
    model = utilities.get_model()
    model = train_clpr_model(model, input_doc)
    return model


def train_model(model, input_doc):
    """Apply GridSearch or normal fitting to model and save it."""
    feature = input_doc._.Features
    label = input_doc._.Labels
    feature = np.asarray(feature)
    label = np.asarray(label)
    train_method = config["adu"]["train_method"]
    model_type = config["adu"]["model"]
    if (
        train_method in ["GridSearch", "RandomSearch"]
        and not model_type == "AutoML"
        and not model_type == "Stacking"
    ):
        param_grid = utilities.get_param_grid()
        cv_split = StratifiedShuffleSplit(
            n_splits=config["adu"]["n_splits"], test_size=0.33
        )
        if train_method == "GridSearch":
            model = GridSearchCV(model, param_grid=param_grid, cv=cv_split, refit=True)
        elif train_method == "RandomSearch":
            model = RandomizedSearchCV(
                model, cv=cv_split, refit=True, param_distributions=param_grid
            )
        model.fit(feature, label)
        utilities.save_model(model, model.best_params_)
    else:
        model.fit(feature, label)
        utilities.save_model(model, params=None)
    return model


def train_clpr_model(model, input_doc):
    """Apply GridSearch or normal fitting to model and save it."""
    feature = input_doc._.CLPR_Features
    label = input_doc._.CLPR_Labels
    feature = np.asarray(feature)
    label = np.asarray(label)
    train_method = config["adu"]["train_method"]
    model_type = config["adu"]["model"]
    if (
        train_method in ["GridSearch", "RandomSearch"]
        and not model_type == "AutoML"
        and not model_type == "Stacking"
    ):
        param_grid = utilities.get_param_grid()
        cv_split = StratifiedShuffleSplit(
            n_splits=config["adu"]["n_splits"], test_size=0.33
        )
        if train_method == "GridSearch":
            model = GridSearchCV(model, param_grid=param_grid, cv=cv_split, refit=True)
        elif train_method == "RandomSearch":
            model = RandomizedSearchCV(
                model, cv=cv_split, refit=True, param_distributions=param_grid
            )
        model.fit(feature, label)
        utilities.save_clpr_model(model, model.best_params_)
    else:
        model.fit(feature, label)
        utilities.save_clpr_model(model, params=None)
    return model


def test_model(model, input_doc):
    """Test model and return metrics."""
    feature = input_doc._.Features
    label = input_doc._.Labels
    feature = np.asarray(feature)
    predictions = model.predict(feature)
    label = np.asarray(label)
    acc, prec, rec, f1 = utilities.print_metrics(label, predictions)
    return acc, prec, rec, f1


def test_clpr_model(model, input_doc):
    """Test model and return metrics."""
    feature = input_doc._.CLPR_Features
    label = input_doc._.CLPR_Labels
    feature = np.asarray(feature)
    predictions = model.predict(feature)
    acc, prec, rec, f1 = utilities.print_metrics(label, predictions)
    return acc, prec, rec, f1


def predict(input_doc):
    """Apply model on doc."""
    feature = input_doc._.Features
    feature = np.asarray(feature)
    model = utilities.load_model()
    predictions = model.predict(feature)
    input_doc._.Labels = predictions
    return input_doc


def predict_clpr(input_doc):
    """Apply model on doc."""
    feature = input_doc._.CLPR_Features
    feature = np.asarray(feature)
    model = utilities.load_clpr_model()
    predictions = model.predict(feature)
    input_doc._.CLPR_Labels = predictions.tolist()
    return input_doc


def predict_mc(input_doc):
    """Run majorclaim classification."""
    method = config["adu"]["MC"]["method"]
    embeddings = [
        np.asarray(e)
        for idx, e in enumerate(input_doc._.embeddings)
        if input_doc._.Labels[idx] == 1
    ]

    if method == "centroid":
        result = compute_centroid_sim(embeddings)
        return_list = [0] * len(input_doc._.sentences)
        mc_iter = 0
        for idx, l in enumerate(input_doc._.Labels):
            if l == 1:
                return_list[idx] = result[mc_iter]
                mc_iter += 1
            else:
                return_list[idx] = 0
        input_doc._.MC_List = return_list
    elif method == "pairwise":
        result = compute_pairwise_sim(embeddings)
        return_list = [0] * len(input_doc._.sentences)
        mc_iter = 0
        for idx, l in enumerate(input_doc._.Labels):
            if l == 1:
                return_list[idx] = result[mc_iter]
                mc_iter += 1
            else:
                return_list[idx] = 0
        input_doc._.MC_List = return_list
    elif method == "first":
        input_doc._.MC_List = get_first_claim(input_doc)
    else:
        input_doc._.MC_List = get_first_claim(input_doc)  # TODO: Maybe wrong fallback

    return input_doc


def get_first_claim(input_doc):
    sentences = input_doc._.sentences
    mc_list = [0] * len(sentences)

    for i, sent in enumerate(sentences):
        if sent._.Label == 1:
            mc_list[i] = 1
            break

    if 1 not in mc_list:
        mc_list[0] = 1

    return mc_list


def get_centroid(embeddings):
    """Compute embedding centroid."""
    centroid = sum_vectors(embeddings) / len(embeddings)
    return centroid


def sum_vectors(vec_list):
    """Sum up vectors element wise."""
    np_list = []
    for vec in vec_list:
        np_list.append(np.array(vec))
    s = sum(np_list)
    return s


def compute_centroid_sim(embeddings):
    """Compute similarity to centroid and return list with nearest vector marked as 1."""
    center = get_centroid(embeddings).reshape(1, -1)

    def sim(x):
        return cosine_similarity(x, center)

    mc_list = [0] * len(embeddings)
    center_sim = []
    for embed in embeddings:
        center_sim.append(sim(embed.reshape(1, -1)))
    max_id = max(range(len(center_sim)), key=center_sim.__getitem__)
    mc_list[max_id] = 1
    return mc_list


def compute_pairwise_sim(embeddings):
    """Compute pairwise similarity and mark vector with highest average to all with 1."""
    mc_list = [0] * len(embeddings)
    pair_sim_sum = []
    for embed1 in embeddings:
        pair_sim = 0
        for embed2 in embeddings:
            pair_sim += cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1))
        pair_sim_sum.append(pair_sim / len(embeddings))
    index_max = max(range(len(pair_sim_sum)), key=pair_sim_sum.__getitem__)
    mc_list[index_max] = 1
    return mc_list
