import sys
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import scipy
import numpy as np
import torch

torch.manual_seed(1)
np.random.seed(0)


clpr_folder = sys.argv[1]
sentence = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(clpr_folder)
model = AutoModelForSequenceClassification.from_pretrained(clpr_folder)


batch_sentences = [sentence]
encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
classification_logits = model(**encoded_inputs)[0]
softmax_output = torch.nn.functional.softmax(classification_logits)
numps = classification_logits.detach().numpy()
softs = scipy.special.softmax(numps, axis=-1)
preds = np.argmax(softs, axis=-1)
print(preds)
