#!/bin/bash

for i in {1..100}
do
   echo "Running iteration $i"
   #python devnet_kfold.py --network_depth=2 --epochs=20 --known_outliers=30 --cont_rate=0.02 --data_format=0 --output=./results/result_kfold.csv --data_set=annthyroid_21feat_normalised --k_folds=5
   python iForest.py
done
