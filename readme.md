U.SUN bank does not allow participants to release their dataset, so we are not allowed to upload the dataset. Hence, the code cannot function properly.

The following README file are for those we have the right to use the U.SAN bank dataset.

1. Download all training dataset, and submission_sample.csv and put them under the /data folder.
2. run preprocess.py first
ex. python3 src/preprocess.py -n 10 -p 20 -d 3 --train --test
n: number of negative samples, p:number of positive samples, d:number of day to aggregate, --train: generating train_data.pkl, --test: generate test_data.pkl
3. run runner.py
ex. python3 runner.py
4. the generated data/prediction.csv file is the predicted results.