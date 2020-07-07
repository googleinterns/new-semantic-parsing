set -e
cd ..

python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing \
  --text-tokenizer bert-base-cased \
  --output-dir data-bin/top_jul7 \


python cli/train_lightning.py \
  --data-dir data-bin/top_jul7 \
  --encoder-model bert-base-cased \
  --decoder-lr 0.2 \
  --encoder-lr 0.02 \
  --batch-size 128 \
  --gradient-accumulation-steps 2 \
  --layers 4 \
  --hidden 256 \
  --dropout 0.3 \
  --heads 4 \
  --epochs 100 \
  --label-smoothing 0.2 \
  --warmup-steps 1500 \
  --num-frozen-encoder-steps 500 \
  --log-every 500 \
  --early-stopping 10 \
  --output-dir output_dir/jul7_run \


python cli/predict.py \
  --data data/top-dataset-semantic-parsing/test.tsv \
  --model output_dir/jul7_run \
  --output-file output_dir/jul7_run/test_predictions.tsv \
