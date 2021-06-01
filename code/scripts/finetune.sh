TOTAL_NUM_UPDATES=$8
WARMUP_UPDATES=$7
LR=$6
MAX_TOKENS=2048
UPDATE_FREQ=$5
BART_PATH=$1
MIN=$9

data_dir=$2
CUDA_VISIBLE_DEVICES=$4 fairseq-train $data_dir --ddp-backend=no_c10d \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.01 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ --max-update $TOTAL_NUM_UPDATES --no-last-checkpoints \
    --skip-invalid-size-inputs-valid-test --save-dir ./checkpoints/$3  \
    --tensorboard-logdir ./tensorboards/$3 \
    --reset-optimizer \
    --find-unused-parameters \
    --save-interval-updates 1 \
    --keep-interval-updates 1 \
    --min-valid-check $MIN \
    --max-source-positions 2048 \
    --reset-dataloader --reset-meters ;
