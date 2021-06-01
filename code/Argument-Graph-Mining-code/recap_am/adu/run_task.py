from recap_am.adu import utilities
from recap_am.adu.classify import (
    fit_model,
    test_model,
    predict,
    predict_mc,
    predict_clpr,
    fit_clpr_model,
    test_clpr_model,
)
import spacy
nlp = spacy.load("en_core_web_sm")
from recap_am.model.config import Config
from recap_am.adu.feature_select import filter_feats, add_embeddings

#
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import scipy
import numpy as np
import torch

#model_dir = "/home/ubuntu/convosumm/transformers/CLPR/"
model_dir = "/project/fas/radev/af726/convosumm/transformers/CLPR"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

config = Config.get_instance()


def run_train(input_doc):
    """Train ADU and CLPR models."""
    input_doc = filter_feats(input_doc, load=True)
    print("Finished Feature selection")
    input_doc = add_embeddings(input_doc)
    print("Added Embeddings")
    adu_model = fit_model(input_doc)
    print("Fit ADU Model")
    return adu_model


def run_clpr_train(input_doc):
    """Train ADU and CLPR models."""
    input_doc = filter_feats(input_doc, load=True)
    print("Finished Feature selection")
    input_doc = add_embeddings(input_doc)
    clpr_feats = []
    for idx, l in enumerate(input_doc._.Labels):
        if l == 1:
            clpr_feats.append(input_doc._.Features[idx])
    input_doc._.CLPR_Features = clpr_feats
    print("Added Embeddings")
    adu_model = fit_clpr_model(input_doc)
    print("Fit CLPR Model")
    return adu_model


def run_test(input_doc, model):
    """Test ADU and CLPR models."""
    input_doc = filter_feats(input_doc, load=True)
    print("Filtered feats")
    input_doc = add_embeddings(input_doc)
    print("Added embeds")
    acc, prec, rec, f1 = test_model(model, input_doc)
    return acc, prec, rec, f1


def run_clpr_test(input_doc, model):
    """Test ADU and CLPR models."""
    input_doc = filter_feats(input_doc, load=True)
    print("Filtered feats")
    input_doc = add_embeddings(input_doc)
    clpr_feats = []
    for idx, l in enumerate(input_doc._.Labels):
        if l == 1:
            clpr_feats.append(input_doc._.Features[idx])
    input_doc._.CLPR_Features = clpr_feats
    print("Added embeds")
    acc, prec, rec, f1 = test_clpr_model(model, input_doc)
    return acc, prec, rec, f1


def run_production(input_doc, default_claim=False):
    """Apply classification on doc."""
    try:
        input_doc = filter_feats(input_doc, load=True)
        input_doc = add_embeddings(input_doc)
        spacy_var = True
    except:
        spacy_var = False
    our_approach = True
    if our_approach:
        doc = input_doc
        doc_sents = list(doc.sents)
        if len(doc_sents) == 1:
            assert default_claim == True
            doc._.Labels = [1]
            doc._.CLPR_Labels = [1]
            return doc
        batch_sentences = [x.text for x in doc_sents]
        encoded_inputs = tokenizer(batch_sentences, padding=True, return_tensors="pt")
        classification_logits = model(**encoded_inputs)[0]
        softmax_output = torch.nn.functional.softmax(classification_logits)
        numps = classification_logits.detach().numpy()
        softs = scipy.special.softmax(numps, axis=-1)
        preds = np.argmax(softs, axis=-1)
        doc._.Labels = [1 if x != 2 else 0 for x in preds]
        labs = []
        for count, x in enumerate(preds):
            if x == 2:
                labs.append(2)
            elif x == 1:
                labs.append(0)
            else:
                # TODO
                if spacy_var:
                    if len(doc_sents[count]) <= 3:
                        labs.append(0)
                    else:
                        labs.append(1)
                else:
                    cur_len = len(nlp(doc_sents[count].text))
                    if cur_len <= 3:
                        labs.append(0)
                    else:
                        labs.append(1)
        if default_claim:
            if 1 not in labs:
                doc._.Labels[0] = 1
                labs[0] = 1
        doc._.CLPR_Labels = labs
        #print(doc._.Labels, doc._.CLPR_Labels)
    else:
        doc = predict(input_doc)
        clpr_feats = []
        for idx, l in enumerate(input_doc._.Labels):
            if l == 1:
                clpr_feats.append(input_doc._.Features[idx])
        if len(clpr_feats) < 2:
            input_doc._.Labels = [1 for s in input_doc._.Labels]
            clpr_feats = input_doc._.Features
        input_doc._.CLPR_Features = clpr_feats
        doc = predict_clpr(input_doc)
        #print(doc._.Labels, doc._.CLPR_Labels)
    return doc
