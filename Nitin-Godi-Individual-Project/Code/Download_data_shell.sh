#!/bin/sh
pip install -q kaggle
mkdir -p ~/.kaggle
cp $1/kaggle.json ~/.kaggle/
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d ilhamfp31/yelp-review-dataset -p $2
cd $2
unzip yelp-review-dataset.zip
cp yelp_review_polarity_csv/test.csv yelp_review_polarity_csv/train.csv $2
rm yelp-review-dataset.zip 
rm -R yelp_review_polarity_csv

