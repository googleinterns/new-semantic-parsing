set -e
cd ..

mkdir data
cd data

curl -JLO http://fb.me/semanticparsingdialog
unzip semanticparsingdialog  # creates top-dataset-semantic-parsing folder
cd ..
