
# Argument classification and graph creation
### Run argument classifier to label as claim, premise, or non-argumentative unit
The classifier was trained on data found in [AMPERSAND: Argument Mining for PERSuAsive oNline Discussions](https://www.aclweb.org/anthology/D19-1291.pdf) and [Parsing Argumentation Structures in Persuasive Essays](https://www.aclweb.org/anthology/J17-3005.pdf), which can be found on S3: `s3://convosumm/data/claim-premise-data/`. </br>

Download the model and run on a dummy sentence: </br>

```
aws s3 cp --recursive s3://convosumm/checkpoints/CLPR/ $CLPR_PATH
python scripts/arg_classifier_test.py $CLPR_PATH "This is a test sentence."
```

### Run claim/premise prediction for filtering without graph creation </br>
```
python scripts/arg_classifier.py
python scripts/process_arg_classifier_results.py
```


### Run claim/premise prediction for each comment and then join claims into a graph, potentially across comments
*See the* `_process()` *function for details about the expected input format.*
```
python Argument-Graph-Mining-code/recap_am/app.join_separate_graphs.py INPUT_FILENAME OUTPUT_DIR 
```

### Run claim/premise prediction for each commment, create subject node for each comment and connect all subject nodes to conversation node (no connections between comments)
*See the* `_process()` *function for details about the expected input format.*
```
python Argument-Graph-Mining-code/recap_am/app.separate_graphs.py INPUT_FILENAME OUTPUT_DIR
```

### Load the graph from .json format and save source as {train,val,test}.graph
```
python scripts/load_graph2txt.py $OUTPUT_DIR
```


</br></br>
# BART 
### Run fairseq preprocessing (for vanilla BART using 2048 tokens) </br>
```
./scripts/prep.sh
```

### Run BART training in fairseq (num of visible devices * UPDATE_FREQ = 32) </br>
```
./scripts/finetune.sh $BART_PATH $DATA_DIR $CHECKPOINT_DIR $TENSORBOARD_DIR $CUDA_VISIBLE_DEVICES $UPDATE_FREQ  3e-5 20 200 -1
```

### Run BART inference  </br>
```
python scripts/inference.py $MODEL_DIR checkpoint_best.pt $DATA-BIN $TEST_SOURCE_FILE $OUTPUT_FILE 4 1 80 120 $BATCH_SIZE 2048 ./misc/encoder.json ./misc/vocab.bpe 
```

</br></br>
# Longformer
See `./longformer-code/run.sh` for an example of running the Longformer.
