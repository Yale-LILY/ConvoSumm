
python ./scripts/summarization.py \
  --data_dir ../../data/data-processed/ami/vanilla/ \
  --batch_size=1 \
  --save_prefix=longformer-encdec-large-1200_poly_attention_340_label_smoothing_nooverlap_512_said_summary_act_cnndm \
  --model_path=./converted-model-bart-large-cnn-long-16384 \
  --gpus=4 \
  --dataset_size=97 \
  --max_input_len=12000 \
  --max_output_len=512 \
  --epochs 66 \
  --grad_accum 8 \
  --grad_ckpt \
  --label_smoothing 0.1 \
  --poly_schedule \
  --attention_mode sliding_chunks_no_overlap \
  --attention_window 340 \
  --warmup 20;

# For running inference
#--test \
#--from_pretrained /gpfs/loomis/project/radev/af726/convosumm/Argument-Graph-Mining/longformer/summarization/longformer-encdec-large-1200_poly_attention_340_label_smoothing_nooverlap_512_said_summary_act_cnndm/_ckpt_epoch_52.ckpt \
