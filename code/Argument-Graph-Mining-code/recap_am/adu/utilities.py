import json
import os
from pathlib import Path

import joblib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    RandomizedSearchCV,
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC
from sklearn.utils import shuffle
from spacy.tokens import Doc, Span

try:
    import xgboost as xgb
except:
    pass
try:
    import autosklearn.classification
except ImportError:
    pass
try:
    from sklearn.ensemble import StackingClassifier
except ImportError:
    pass


from recap_am.model.config import Config
from recap_am.adu.grids import Grid

config = Config.get_instance()

lang = config["nlp"]["language"]


def get_model():
    """Return model of specified type."""
    model_type = config["adu"]["model"]
    if model_type == "SVC":
        model = SVC()
    elif model_type == "LogisticRegression":
        model = LogisticRegression()
    elif model_type == "RandomForest":
        model = RandomForestClassifier()
    elif model_type == "AdaBoost":
        model = AdaBoostClassifier()
    elif model_type == "XGBoost":
        model = xgb.XGBClassifier()
    elif model_type == "XGBRF":
        model = xgb.XGBRFClassifier()
    elif model_type == "AutoML":
        model = autosklearn.classification.AutoSklearnClassifier(
            resampling_strategy="cv",
            resampling_strategy_arguments={"folds": 10},
            n_jobs=80,
            ensemble_memory_limit=10240,
            ml_memory_limit=30720,
        )
    elif model_type == "Stacking":
        cv_split = StratifiedShuffleSplit(
            n_splits=config["adu"]["n_splits"], test_size=0.33
        )
        rf_param_grid = Grid["RandomForest"]
        log_param_grid = Grid["LogisticRegression"]
        svc_param_grid = Grid["SVC"]
        ada_param_grid = Grid["AdaBoost"]
        xgb_param_grid = Grid["XGBoost"]
        xgbrf_param_grid = Grid["XGBRF"]
        train_method = config["adu"]["train_method"]
        estimator_dict = dict()
        if train_method == "GridSearch":
            estimator_dict["rf"] = GridSearchCV(
                RandomForestClassifier(),
                param_grid=rf_param_grid,
                cv=cv_split,
                refit=True,
            )
            estimator_dict["log"] = GridSearchCV(
                LogisticRegression(), param_grid=log_param_grid, cv=cv_split, refit=True
            )
            estimator_dict["svc"] = GridSearchCV(
                SVC(), param_grid=svc_param_grid, cv=cv_split, refit=True
            )
            estimator_dict["ada"] = GridSearchCV(
                AdaBoostClassifier(), param_grid=ada_param_grid, cv=cv_split, refit=True
            )
            estimator_dict["xgbrf"] = GridSearchCV(
                xgb.XGBRFClassifier(),
                param_grid=xgbrf_param_grid,
                cv=cv_split,
                refit=True,
            )
            estimator_dict["xgb"] = GridSearchCV(
                xgb.XGBClassifier(), param_grid=xgb_param_grid, cv=cv_split, refit=True
            )
        elif train_method == "RandomSearch":
            estimator_dict["rf"] = RandomizedSearchCV(
                RandomForestClassifier(),
                param_distributions=rf_param_grid,
                cv=cv_split,
                refit=True,
            )
            estimator_dict["log"] = RandomizedSearchCV(
                LogisticRegression(),
                param_distributions=log_param_grid,
                cv=cv_split,
                refit=True,
            )
            estimator_dict["svc"] = RandomizedSearchCV(
                SVC(), param_distributions=svc_param_grid, cv=cv_split, refit=True
            )
            estimator_dict["ada"] = RandomizedSearchCV(
                AdaBoostClassifier(),
                param_distributions=ada_param_grid,
                cv=cv_split,
                refit=True,
            )
            estimator_dict["xgbrf"] = RandomizedSearchCV(
                xgb.XGBRFClassifier(),
                param_distributions=xgbrf_param_grid,
                cv=cv_split,
                refit=True,
            )
            estimator_dict["xgb"] = RandomizedSearchCV(
                xgb.XGBClassifier(),
                param_distributions=xgb_param_grid,
                cv=cv_split,
                refit=True,
            )

        stacks = config["adu"]["stacking"]["estimator_stack"]
        final_est = estimator_dict[config["adu"]["stacking"]["final_estimator"]]
        passth = config["adu"]["stacking"]["passthrough"]
        single_layer = []
        for i, m in enumerate(stacks):
            if isinstance(m, list):
                sublayer = [
                    (mo + str(i + j), estimator_dict[mo]) for j, mo in enumerate(m)
                ]
                layer = StackingClassifier(
                    estimators=sublayer,
                    final_estimator=final_est,
                    n_jobs=-1,
                    passthrough=passth,
                    verbose=0,
                )
                final_est = layer
            else:
                single_layer.append((m + str(i), estimator_dict[m]))

        if len(single_layer) > 0:
            model = StackingClassifier(
                estimators=single_layer,
                final_estimator=final_est,
                n_jobs=-1,
                passthrough=passth,
                verbose=0,
            )
        else:
            model = layer
    else:
        print("Invalid model option")
        exit(1)
    return model


def load_model():
    """Load model from file."""
    load_path = Path("data/ADU/models") / lang / "final"
    load_name = "ADU_Stacking.pkl"
    path = load_path / load_name
    model = joblib.load(path)
    return model


def load_clpr_model():
    """Load model from file."""
    load_path = Path("data/ADU/models") / lang / "final"
    load_name = "CLPR_Stacking.pkl"
    path = load_path / load_name
    model = joblib.load(path)
    return model


def get_param_grid():
    """Load parameter grid from file for GridSearch."""
    param_grid = Grid[config["adu"]["model"]]
    return param_grid


def print_metrics(targets, predictions):
    """Print testing metrics."""
    print("Accuracy\t%8.8f" % accuracy_score(targets, predictions))
    print("Precision:\t%8.8f" % precision_score(targets, predictions))
    print("Recall:\t%8.8f" % recall_score(targets, predictions))
    print("F1-Score:\t%8.8f" % f1_score(targets, predictions))
    return (
        accuracy_score(targets, predictions),
        precision_score(targets, predictions),
        recall_score(targets, predictions),
        f1_score(targets, predictions),
    )


def save_model(model, params):
    """Save model and parameters to file."""
    if params is not None:
        param_dir = (
            Path("data") / "ADU" / "models" / lang / config["adu"]["model"] / "params"
        )
        param_dir.mkdir(parents=True, exist_ok=True)

        param_index = len(os.listdir(param_dir)) + 1
        param_name = config["adu"]["model"] + str(param_index) + ".params"
        param_path = param_dir / param_name
        if not os.path.isdir(param_dir):
            os.makedirs(param_dir)
        with open(param_path, "w+") as f:
            json.dump(params, f)

    save_dir = (
        Path("data") / "ADU" / "models" / lang / config["adu"]["model"] / "joblib"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    save_index = len(os.listdir(save_dir)) + 1
    save_name = config["adu"]["model"] + str(save_index) + ".pkl"
    model_path = save_dir / save_name
    joblib.dump(model, model_path)


def save_clpr_model(model, params):
    """Save model and parameters to file."""
    if params is not None:
        param_dir = (
            Path("data")
            / "ADU"
            / "models"
            / lang
            / "clpr"
            / config["adu"]["model"]
            / "params"
        )
        param_dir.mkdir(parents=True, exist_ok=True)

        param_index = len(os.listdir(param_dir)) + 1
        param_name = config["adu"]["model"] + str(param_index) + ".params"
        param_path = param_dir / param_name
        if not os.path.isdir(param_dir):
            os.makedirs(param_dir)
        with open(param_path, "w+") as f:
            json.dump(params, f)

    save_dir = (
        Path("data")
        / "ADU"
        / "models"
        / lang
        / "clpr"
        / config["adu"]["model"]
        / "joblib"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    save_index = len(os.listdir(save_dir)) + 1
    save_name = config["adu"]["model"] + str(save_index) + ".pkl"
    model_path = save_dir / save_name
    joblib.dump(model, model_path)
