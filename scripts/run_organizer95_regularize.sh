set -e
cd ..


SET_NAME=organizer_95
DATE=Jan9
CLASSES=SL:ORGANIZER_EVENT

DATA=data-bin/"$SET_NAME"_"$DATE"
MODEL=output_dir/"$SET_NAME"_"$DATE"
BATCH_SIZE=112
SPLIT=0.95


python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing \
  --text-tokenizer bert-base-cased \
  --output-dir $DATA \
  --split-class $CLASSES \
  --split-amount $SPLIT \

TAG="$SET_NAME"_"$DATE"

python cli/train.py \
  --data-dir $DATA  \
  --encoder-model bert-base-cased \
  --decoder-lr 0.2 \
  --encoder-lr 0.02 \
  --batch-size $BATCH_SIZE \
  --layers 4 \
  --hidden 256 \
  --dropout 0.2 \
  --heads 4 \
  --label-smoothing 0.0 \
  --epochs 100 \
  --early-stopping 10 \
  --warmup-steps 1500 \
  --freeze-encoder 0 \
  --unfreeze-encoder 500 \
  --log-every 150 \
  --output-dir $MODEL \
  --tags train,$TAG \
  --new-classes $CLASSES \
  --track-grad-square \
  --seed 1 \


TAG="$SET_NAME"_"$DATE"_ewc_find

for old_data_amount in 0.0 0.05 0.1 0.2 0.5
do

for ewc in 10 0 100
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --epochs 50 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --weight-consolidation $ewc \
      --new-classes $CLASSES \
      --tags finetune,$TAG,ewc_"$ewc" \
      --output-dir output_dir/finetuned \
      --old-data-sampling-method merge_subset \

done
done


# dropout
TAG="$SET_NAME"_"$DATE"_higher_dropout
# path_99_Aug19_bert_run_baseline


for old_data_amount in 0.0 0.05 0.1 0.2 0.5
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.4 \
      --epochs 50 \
      --early-stopping 20 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done


TAG="$SET_NAME"_"$DATE"_move_norm

for old_data_amount in 0.0 0.05 0.1 0.2 0.5
do

for move_norm in 0.01 0.1
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --epochs 50 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes $CLASSES \
      --tags finetune,$TAG,move_norm_"$move_norm" \
      --output-dir output_dir/finetuned \
      --old-data-sampling-method merge_subset \

done
done


TAG="$SET_NAME"_"$DATE"_ewc_find_sample

for old_data_amount in 0.0 0.05 0.1 0.2 0.5
do

for ewc in 10 0 100
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --epochs 50 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --weight-consolidation $ewc \
      --new-classes $CLASSES \
      --tags finetune,$TAG,ewc_"$ewc" \
      --output-dir output_dir/finetuned \
      --old-data-sampling-method sample \

done
done
