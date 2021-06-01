


export TASK_NAME=CLPR
#   --task_name $TASK_NAME \

# --model_name_or_path bert-base-uncased \
python /private/home/alexfabbri/convosumm/transformers/examples/text-classification/run_glue.py \
  --model_name_or_path /private/home/alexfabbri/convosumm/AMPERSAND-EMNLP2019/models \
  --do_train \
  --do_eval \
  --train_file /private/home/alexfabbri/convosumm/AMPERSAND-EMNLP2019/data/training_data/train_combined.csv \
  --validation_file /private/home/alexfabbri/convosumm/AMPERSAND-EMNLP2019/data/training_data/dev.csv \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /private/home/alexfabbri/convosumm/transformers/$TASK_NAME/


#12/16/2020 06:42:50 - INFO - __main__ -     eval_loss = 0.8684926629066467
#12/16/2020 06:42:50 - INFO - __main__ -     eval_accuracy = 0.6016949415206909
#12/16/2020 06:42:50 - INFO - __main__ -     epoch = 3.0


#12/16/2020 06:45:13 - INFO - __main__ -     eval_loss = 0.69954913854599
#12/16/2020 06:45:13 - INFO - __main__ -     eval_accuracy = 0.7033898234367371
#12/16/2020 06:45:13 - INFO - __main__ -     epoch = 3.0
