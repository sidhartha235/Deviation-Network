## Instructions to run the devnet model

* Create a virtual environment:
```
python -m venv env
```

* Activate the virtual environment (in Windows):
```
.\env\Scripts\activate
```

* Install requirements:
```
pip install -r requirements.txt
```

* Run the model (this is an example, change dataset and params accordingly):
```
python devnet.py --network_depth=2 --runs=10 --known_outliers=30 --cont_rate=0.02 --data_format=0 --output=./results/result.csv --data_set=annthyroid_21feat_normalised
```