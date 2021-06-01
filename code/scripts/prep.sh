wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'


INPUT_DIR=$1
OUTPUT_DIR=$2

mkdir -p $OUTFOLDER
SPLIT=train
for LANG in source target
do
  python -m examples.roberta.multiprocessing_bpe_encoder \
  --encoder-json ../misc/encoder.json \
  --vocab-bpe ../misc/vocab.bpe \
  --inputs "$INPUT_DIR/$SPLIT.$LANG" \
  --outputs "$OUTFOLDER/$SPLIT.bpe.$LANG" \
  --workers 60 \
  --keep-empty;
done

SPLIT=val
for LANG in source target
do
  python -m examples.roberta.multiprocessing_bpe_encoder \
  --encoder-json ../misc/encoder.json \
  --vocab-bpe ../misc/vocab.bpe \
  --inputs "$INPUT_DIR/$SPLIT.$LANG" \
  --outputs "$OUTFOLDER/$SPLIT.bpe.$LANG" \
  --workers 60 \
  --keep-empty;
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "$INPUT_DIR/train.bpe" \
  --validpref "$INPUT_DIR/val.bpe" \
  --destdir "$OUTPUT_DIR" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
