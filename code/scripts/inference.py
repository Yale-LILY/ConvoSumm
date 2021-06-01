import sys
import time
import torch
from fairseq.models.bart import BARTModel

model_dir = sys.argv[1]
model_file = sys.argv[2]
bin_folder = sys.argv[3]
inputf = sys.argv[4]
outputf = sys.argv[5]
beam = int(sys.argv[6])
lenpen = float(sys.argv[7])
min_len = int(sys.argv[8])
max_len = int(sys.argv[9])
bsz = int(sys.argv[10])
max_source_positions = int(sys.argv[11])
encoder_file = sys.argv[12]
vocab_file = sys.argv[13]

bart = BARTModel.from_pretrained(
    model_dir,
    checkpoint_file=model_file,
    data_name_or_path=bin_folder,
    gpt2_encoder_json=encoder_file,
    gpt2_vocab_bpe=vocab_file,
    max_source_positions=max_source_positions
)

bart.cuda()
bart.eval()
# bart.half()
count = 1
with open(inputf) as source, open(outputf, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                print(count)
                hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, min_len=min_len, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        with torch.no_grad():
            hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, min_len=min_len, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
