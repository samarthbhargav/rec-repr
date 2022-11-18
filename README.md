
## Env setup

```
pyenv virtualenv 3.7.10 rec-repr
echo rec-repr > ./.python-version

pip install -r main_reqs.txt
```

## Data Preprocessing
```
mkdir datasets/
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip  -O datasets/ml-1m.zip
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip -O datasets/ml-20m.zip
cd datasets/
unzip ml-1m.zip
unzip ml-20m.zip
cd ../

mkdir datasets/raw_goodreads/
## download the following files from https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home and move them into the raw_goodreads folder
# goodreads_interactions_comics_graphic.json.gz to datasets/raw_goodreads/comic
# goodreads_interactions_children.json.gz to datasets/raw_goodreads/children
# goodreads_reviews_dedup.json.gz to datasets/raw_goodreads/goodreads


mkdir datasets/msd
cd datasets/msd
wget http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip
unzip train_triplets.txt.zip
rm *.zip
cd ../
 
python preprocessing.py --dataset datasets/ml-1m/ --output_dir datasets/ml-1m/processed/ --threshold 3.5 --min_items_per_user 0 --heldout_users 500 --dataset_name ml-1m 
python preprocessing.py --dataset datasets/ml-20m/ --output_dir datasets/ml-20m/processed/ --threshold 3.5 --heldout_users 10000 --dataset_name ml-20m
python preprocessing.py --dataset_name gr --dataset datasets/raw_goodreads/goodreads/ --output_dir datasets/gr --threshold 3.5  --min_users_per_item 15 --min_items_per_user 10 --heldout_users  10000
python preprocessing.py --dataset_name gr-children --dataset datasets/raw_goodreads/children/ --output_dir datasets/gr-children --threshold 3.5  --min_users_per_item 10 --min_items_per_user 10 --heldout_users  7500
python preprocessing.py --dataset_name gr-comics --dataset datasets/raw_goodreads/comic/ --output_dir datasets/gr-comics --threshold 3.5 --min_users_per_item 10 --min_items_per_user 10 --heldout_users  5000
python preprocessing.py --dataset datasets/msd/ --output_dir datasets/msd/processed/ --threshold 3.5 --min_items_per_user 20 --min_users_per_item 200 --heldout_users 50000 --dataset_name msd

```
## Example commands to reproduce

```
## train model, model_config needs to be provided
## see python run.py --help for full list of args!
python run.py --dataset ml-1m --model_config ./example_model_config.json --out test_results/ml-1m/out.json --save test_results/ml-1m/model.pt --device cpu --n_epochs 5

# test the model, test results are stored!
python test.py --run test_results/ml-1m/out.json --result_loc test_results/ml-1m/out_test.json --device cpu
```

