import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from spacy.tokens import Doc, Span
from dataclasses import dataclass
from enum import Enum


from recap_am.model.config import Config
from recap_am.relation.model.relation import Relation, RelationClass



config = Config.get_instance()


def classify(adus: Iterable[Union[Doc, Span]]) -> Dict[str, List[Relation]]:
    model = _load_model()
    classification = {}

    for adu1 in adus:
        classification[adu1.text] = list()

        for adu2 in adus:
            if adu1 != adu2:
                our_approach = True
                if our_approach:
                    tokens = roberta.encode(adu1.text, adu2.text)
                    output = roberta.predict('mnli', tokens)
                    output = output.detach().cpu().numpy()
                    softs = scipy.special.softmax(output, axis=-1)[0]
                    if softs[0] > softs[2]:
                        pred_type = RelationClass.ATTACK
                        pred_prob = softs[0]
                    else:
                        pred_type = RelationClass.SUPPORT
                        pred_prob = softs[2]
                else:
                    sample = _transform(adu1, adu2)
                    pred_type, pred_prob = _predict(sample, model)
                #print(adu1.text, "|||", adu2.text, pred_prob, pred_type)
                classification[adu1.text].append(
                    Relation(adu2.text, pred_prob, pred_type)
                )

    return classification


def _load_model():
    if config["nlp"]["language"] == "en":
        model_path = Path(config["relation"]["en_model"])
    elif config["nlp"]["language"] == "de":
        model_path = Path(config["relation"]["de_model"])
    else:
        raise ValueError("Wrong language given.")

    with model_path.open("rb") as f:
        return pickle.load(f)


def _transform(adu1: Union[Doc, Span], adu2: Union[Doc, Span]):
    data1 = [adu1.vector]
    data2 = [adu2.vector]

    data = np.array(list(zip(data1, data2)))
    data = np.reshape(data, (len(data), 600))

    return data


def _predict(
    sample: np.ndarray, model: LogisticRegression
) -> Tuple[RelationClass, float]:
    prediction = model.predict_proba(sample)
    threshold = config["relation"]["threshold"]
    fallback = config["relation"]["fallback"]

    prob_support = prediction[0, 1]
    prob_attack = prediction[0, 0]
    max_prob = max(prob_support, prob_attack)

    if max_prob >= threshold:
        if prob_attack > prob_support:
            return RelationClass.ATTACK, max_prob
        else:
            return RelationClass.SUPPORT, max_prob

    elif fallback == "none":
        return RelationClass.NONE, max_prob
    elif fallback == "attack":
        return RelationClass.ATTACK, max_prob
    elif fallback == "support":
        return RelationClass.SUPPORT, max_prob

    raise ValueError("Wrong value given for 'relation.fallback'.")
