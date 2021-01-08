set -e
cd ..


SET_NAME=path_99
DATE=Aug19
CLASSES=SL:PATH

DATA=data-bin/"$SET_NAME"_"$DATE"
MODEL=output_dir/"$SET_NAME"_"$DATE"_bert_run


python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing \
  --text-tokenizer bert-base-cased \
  --output-dir $DATA \
  --split-class SL:PATH \
  --split-amount 0.99 \


python cli/train.py \
  --data-dir $DATA  \
  --encoder-model bert-base-cased \
  --decoder-lr 0.2 \
  --encoder-lr 0.02 \
  --batch-size 192 \
  --layers 4 \
  --hidden 256 \
  --dropout 0.2 \
  --heads 4 \
  --label-smoothing 0.1 \
  --epochs 100 \
  --warmup-steps 1500 \
  --freeze-encoder 0 \
  --unfreeze-encoder 500 \
  --log-every 150 \
  --early-stopping 10 \
  --output-dir $MODEL \
  --tags train,$TAG \
  --new-classes $CLASSES \
  --seed 1 \


# baseline
TAG="$SET_NAME"_"$DATE"_bert_run_baseline
# path_99_Aug19_bert_run_baseline


for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size 128 \
      --dropout 0.2 \
      --epochs 40 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done


# sample
TAG="$SET_NAME"_"$DATE"_bert_run_sample


for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size 128 \
      --dropout 0.2 \
      --epochs 40 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --old-data-sampling-method sample \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done

# move norm 0.05
TAG="$SET_NAME"_"$DATE"_bert_run_move_norm

for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size 128 \
      --dropout 0.2 \
      --move-norm 0.1 \
      --epochs 40 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done

# freeze encoder
TAG="$SET_NAME"_"$DATE"_bert_run_freeze

for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size 128 \
      --dropout 0.2 \
      --freeze-encoder 0 \
      --epochs 40 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done

# best recipe
TAG="$SET_NAME"_"$DATE"_bert_run_best

for old_data_amount in 0.0 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size 128 \
      --dropout 0.2 \
      --move-norm 0.1 \
      --freeze-encoder 0 \
      --epochs 40 \
      --early-stopping 10 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --old-data-sampling-method sample \
      --new-classes $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done
