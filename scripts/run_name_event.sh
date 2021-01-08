# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

set -e
cd ..

DATA=data-bin/top_name_event_95
MODEL=output_dir/name_event_95_bert_run
CLASSES=SL:NAME_EVENT
TAG=name_event_bert_run

# Preprocess

python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing \
  --text-tokenizer bert-base-cased \
  --split-amount 0.95 \
  --split-class $CLASSES \
  --output-dir $DATA \


# Train


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
  --new-classes-file $CLASSES \


# Finetune

for new_data_amount in 0.1 0.3 0.5 0.7 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size 128 \
      --dropout 0.2 \
      --epochs 40 \
      --log-every 100 \
      --new-data-amount $new_data_amount \
      --new-classes-file $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done

for old_data_amount in 0.01 0.05 0.1 0.15 0.2 0.5 1.0
do

    rm -rf output_dir/finetuned

    python cli/retrain.py \
      --data-dir $DATA \
      --model-dir $MODEL \
      --batch-size 128 \
      --dropout 0.2 \
      --epochs 40 \
      --log-every 100 \
      --new-data-amount 1.0 \
      --old-data-amount $old_data_amount \
      --new-classes-file $CLASSES \
      --tags finetune,$TAG \
      --output-dir output_dir/finetuned \


done


for SEED in 2 3 4 5
do

  rm -rf "$MODEL"_seed

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
    --new-classes-file $CLASSES \
    --seed $SEED \

done
