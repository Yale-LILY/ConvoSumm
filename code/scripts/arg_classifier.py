import json
import re
import numpy as np
import pickle as pkl
import glob
import torch
from tqdm import tqdm
import scipy
import os

import spacy
nlp = spacy.load("en_core_web_sm")

from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
# TODO download the checkpoint from S3
model_dir = "./CLPR"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

def get_source_target(inputfs, inputft, outputfs):
    with open(inputfs) as inputf1, open(inputft) as inputf2, open(outputfs, "w") as outputf:
        for count, (l1, l2) in enumerate(zip(inputf1, inputf2)):
            print(count)
            emails = l1.strip().split("</s>")
            texts = [x.split("</S>")[1] for x in emails]
            roles = [x.split("</S>")[0] for x in emails]

            texts_filtered = []
            roles_filtered = []
            for role, text in zip(roles, texts):
                
                new_texts = list(nlp(text).sents)
                new_texts = [x.text for x in new_texts]
                new_roles = [role] * len(new_texts)
                
                texts_filtered.extend(new_texts)
                roles_filtered.extend(new_roles)
        
                assert len(roles_filtered) == len(texts_filtered)
        
            ranges = list(range(len(roles_filtered)))
            num_parts = len(roles_filtered)//32
            num_parts = max(num_parts, 1)
            splits = np.array_split(ranges, num_parts)
            predictions = []
            for split in splits:
                batch_sentences = [texts_filtered[x] for x in split]
                encoded_inputs = tokenizer(batch_sentences, padding=True, return_tensors="pt")
                classification_logits = model(**encoded_inputs)[0]
                softmax_output = torch.nn.functional.softmax(classification_logits)
                numps = classification_logits.detach().numpy()
                softs = scipy.special.softmax(numps, axis=-1)
                cur_preds = np.argmax(softs, axis=-1).tolist()
                predictions.extend(cur_preds)
    
            summary = l2.strip()
            cur_data = {"roles": roles_filtered, "texts": texts_filtered, "predictions": predictions, "summary": summary, "id": count}
    
            json.dump(cur_data, outputf)
            outputf.write("\n")
    
if not os.path.isdir("./out"):
    os.mkdir("./out")

get_source_target("train.source", "train.target", "out/train.jsonl")
get_source_target("val.source", "val.target", "out/val.jsonl")
get_source_target("test.source", "test.target", "out/test.jsonl")
