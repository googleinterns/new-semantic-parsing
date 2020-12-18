set -e
cd ..


SET_NAME=road_95
DATE=Dec18
CLASSES=IN:GET_INFO_ROAD_CONDITION

DATA=data-bin/"$SET_NAME"_"$DATE"
MODEL=output_dir/"$SET_NAME"_"$DATE"
BATCH_SIZE=128
SPLIT=0.95


python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing \
  --text-tokenizer bert-base-cased \
  --output-dir $DATA \
  --split-class $CLASSES \
  --split-amount $SPLIT \


python cli/train_lightning.py \
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
  --warmup-steps 1500 \
  --freeze-encoder 0 \
  --unfreeze-encoder 500 \
  --log-every 150 \
  --early-stopping 10 \
  --output-dir $MODEL \
  --tags train,$TAG \
  --new-classes $CLASSES \
  --track-grad-square \
  --seed 1 \


TAG="$SET_NAME"_"$DATE"_ewc_find
# path_99_Aug19_bert_run_baseline


for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

for ewc in 1000000 0 100000 1000
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


done
done


# baseline
TAG="$SET_NAME"_"$DATE"_baseline
# path_99_Aug19_bert_run_baseline


for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --epochs 50 \
      --early-stopping 20 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done


# sample
TAG="$SET_NAME"_"$DATE"_sample


for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --epochs 50 \
      --early-stopping 20 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --old-data-sampling-method sample \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done

TAG="$SET_NAME"_"$DATE"_ewc_find_sample
# path_99_Aug19_bert_run_baseline


for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

for ewc in 1000000 0 100000 1000
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


# move norm 0.05
TAG="$SET_NAME"_"$DATE"_move_norm

for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --move-norm 0.1 \
      --epochs 50 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done

# freeze encoder
TAG="$SET_NAME"_"$DATE"_freeze

for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --freeze-encoder 0 \
      --epochs 50 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done


# new data
TAG="$SET_NAME"_"$DATE"_new_data


for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do
for new_data_amount in 0.1 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size $BATCH_SIZE \
      --dropout 0.2 \
      --epochs 50 \
      --early-stopping 20 \
      --log-every 100 \
      --new-data-amount $new_data_amount \
      --old-data-amount $old_data_amount \
      --old-data-sampling-method sample \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \

done
done
