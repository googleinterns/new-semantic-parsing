set -e
cd ..


export TOKENIZERS_PARALLELISM=false

SET_NAME=path_99_debug
CLASSES=SL:PATH
BATCH_SIZE=32

DATA=data-bin/"$SET_NAME"
MODEL=output_dir/"$SET_NAME"

rm -rf $DATA
rm -rf $MODEL
rm -rf output_dir/finetuned


python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing-1000 \
  --text-tokenizer bert-base-cased \
  --output-dir $DATA \
  --split-class SL:PATH \
  --split-amount 0.99 \

python cli/train.py \
  --data-dir $DATA  \
  --decoder-lr 0.2 \
  --encoder-lr 0.02 \
  --batch-size $BATCH_SIZE \
  --layers 2 \
  --hidden 32 \
  --dropout 0.2 \
  --heads 2 \
  --epochs 3 \
  --warmup-steps 100 \
  --freeze-encoder 0 \
  --unfreeze-encoder 500 \
  --log-every 150 \
  --early-stopping 10 \
  --output-dir $MODEL \
  --tags train,debug \
  --new-classes $CLASSES \
  --seed 1 \


old_data_amount=0.2
ewc=0.1

python cli/retrain.py \
  --data-dir $DATA \
  --model-dir $MODEL \
  --batch-size $BATCH_SIZE \
  --dropout 0.2 \
  --epochs 40 \
  --early-stopping 1 \
  --log-every 100 \
  --new-data-amount 1.0 \
  --old-data-amount $old_data_amount \
  --weight-consolidation $ewc \
  --new-classes $CLASSES \
  --tags finetune,debug,ewc_"$ewc" \
  --output-dir output_dir/finetuned \
  --old-data-sampling-method sample \
