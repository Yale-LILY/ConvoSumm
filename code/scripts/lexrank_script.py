from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path

import sys
import spacy
nlp = spacy.load("en_core_web_sm")
domain = sys.argv[1]
summary_size = int(sys.argv[2])
filename = f"{domain}.test.source.remove_markers_simple_separator"

documents = []
with open(filename) as inputf:
    for count, line in enumerate(inputf):
        if count % 50 == 0:
            print(count)
        line = line.strip()
        sentences = [x.text for x in nlp(line.strip()).sents]
        sentences = [x for x in sentences if (not x.isspace() and len(x) > 0)]
        documents.append(sentences)

lxr = LexRank(documents, stopwords=STOPWORDS['en'])
print("done loading lexrank")
with open(filename) as inputf, open(filename + ".lexrank_more", "w") as outputf:
    for count, line in enumerate(inputf):
        if count % 500 == 0:
            print(count)
        line = line.strip()
        sentences = [x.text for x in nlp(line.strip()).sents]
        # get summary with classical LexRank algorithm
        summary = lxr.get_summary(sentences, summary_size=summary_size, threshold=None)
        output_summary = " ".join(summary)
        outputf.write(output_summary + "\n")
