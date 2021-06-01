import sys
from summ_eval.rouge_metric import RougeMetric

hyp_file = sys.argv[1]
ref_file = sys.argv[2]

rouge = RougeMetric()


summaries = []
with open(hyp_file) as inputf:
    for line in inputf:
        summaries.append(line.strip())

multi_references = []
with open(ref_file) as inputf:
    for line in inputf:
        multi_references.append(line.strip().split("||||"))

rouge_dict = rouge.evaluate_batch(summaries, multi_references)
for key in rouge_dict.keys():
    print(key, rouge_dict[key])
