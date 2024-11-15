#!/bin/bash

MODEL_PATH="./paligemma-3b-pt-224"
PROMPT="this building is "
IMAGE_FILE_PATH="./images/taj-mahal.jpg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.7
TOP_P=0.9
DO_SAMPLE="False"
DEVICE="cpu"


python inference.py \
    --model_path $MODEL_PATH \
    --prompt $PROMPT \
    --image_file_path $IMAGE_FILE_PATH \
    --max_tokens $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --device $DEVICE
