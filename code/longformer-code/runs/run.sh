



#MODEL_PATH="./converted-model-longbart-4096"
##MAX_POS=$((4096 * 4))
#python ./scripts/summarization.py \
#  --batch_size=1 \
#  --model_path=${MODEL_PATH} \
#  --gpus=8 \
#  --max_input_len=4096 \
#  --max_output_len=256 \
#  --grad_ckpt \
#  --epochs 67 \
#  --grad_accum 4 \
#  --warmup 20 \
#  #--test \
#  #--model_path=summarization/test/ \



#MODEL_PATH="./converted-model-longbart-8192"
#MAX_POS=$((4096 * 4))
#python ./scripts/summarization.py \
#  --batch_size=1 \
#  --from_pretrained /private/home/alexfabbri/convosumm/Argument-Graph-Mining/longformer/summarization/test/save/_ckpt_epoch_12.ckpt \
#  --model_path=${MODEL_PATH} \
#  --gpus=8 \
#  --max_input_len=8192 \
#  --max_output_len=256 \
#  --grad_ckpt \
#  --epochs 20 \
#  --grad_accum 4 \
#  --warmup 20 \
#  --test



#MODEL_PATH="./converted-model-longbart-4096"
#python ./scripts/summarization.py \
#  --batch_size=1 \
#  --from_pretrained /private/home/alexfabbri/convosumm/Argument-Graph-Mining/longformer/summarization/test/_ckpt_epoch_12.ckpt \
#  --model_path=${MODEL_PATH} \
#  --gpus=8 \
#  --max_input_len=4096 \
#  --max_output_len=256 \
#  --grad_ckpt \
#  --epochs 20 \
#  --grad_accum 4 \
#  --warmup 20 \
  


#12/28
  
#python  scripts/summarization.py --num_workers 12  --save_prefix eval_long16k_large-sliding_chunks --model_path bart-large-long-16384/ --max_input_len 16000 --batch_size 1 --grad_accum 8 --grad_ckpt   --attention_mode sliding_chunks --attention_window 1024 --val_every 0.333333333  --val_percent_check 1.0 




MODEL_PATH="./bart-large-long-16384"
python ./scripts/summarization.py \
  --batch_size=1 \
  --save_prefix=bart-large-long-16384-1-8-67-20-1024 \
  --model_path=bart-large-long-16384 \
  --gpus=8 \
  --max_input_len=16384 \
  --max_output_len=256 \
  --grad_ckpt \
  --epochs 67 \
  --grad_accum 8 \
  --batch_size 1 \
  --warmup 20 \
  --attention_window 1024 ;
