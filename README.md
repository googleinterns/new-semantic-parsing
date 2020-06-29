# New Project Template

"This is not an officially supported Google product"

This template uses the Apache license, as is Google's default.  See the
documentation for instructions on using alternate license.


### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Usage
```bash
# download data

sh scripts/download_data.sh

# preprocess

python cli/preprocess.py \
  --data data/top-dataset-semantic-parsing \
  --text-tokenizer bert-base-cased \
  --save-dir data-bin/top_dataset \


# train

python cli/train.py \
  --data-dir data-bin/top_dataset \
  --encoder-model bert-base-cased \
  --decoder-lr 0.006 \
  --encoder-lr 0.00001 \
  --batch-size 32 \
  --gradient-accumulation-steps 4 \
  --dropout 0.1 \
  --decoder-layers 4 \
  --decoder-hidden 128 \
  --decoder-heads 2 \
  --epochs 150 \
  --log-every 100 \
  --output-dir output-dir/hodpxokz \


# predict

python cli/predict.py \
  --data data/top-dataset-semantic-parsing/test.tsv \
  --model output_dir/hodpxokz \
  --output-file output_dir/hodpxokz/predictions.tsv \


```

## Source Code Headers

Every file containing source code must include copyright and license
information. This includes any JS/CSS files that you might be serving out to
browsers. (This is to help well-intentioned people avoid accidental copying that
doesn't comply with the license.)

Apache header:

    Copyright 2020 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
