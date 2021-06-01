
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import scipy
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained("/private/home/alexfabbri/convosumm/transformers/CLPR/")
model = AutoModelForSequenceClassification.from_pretrained("/private/home/alexfabbri/convosumm/transformers/CLPR/")


import sys
import spacy

nlp = spacy.load("en_core_web_sm")

filename = sys.argv[1]

import pdb;pdb.set_trace()
with open(filename) as inputf:
    for line in inputf:
        doc = nlp(line.strip())
        doc_sents = list(doc.sents)
        batch_sentences = [x.text for x in doc_sents]
        encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
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
                if len(doc_sents[count]) <= 3:
                    labs.append(0)
                else:
                    labs.append(1)
