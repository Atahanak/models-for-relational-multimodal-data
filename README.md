# models-for-relational-multimodal-data

## Setup

Install the conda environment:
```
conda env create -f env.yml
conda activate rel-mm
conda develop ./src/
conda develop ./pytorch-frame/
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.0.0%2Bcu118.html
```

Init submodules:
```
git submodule update --init pytorch-frame
```

Finally, if you havent already create a wandb account from the following link https://docs.wandb.ai/quickstart.

## Usage

As an example dataset, download the transactions for Anti Money Laundering (AML) dataset from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data). After downloading it, you first need to perform a pre-processing step:
```
python ./data/prepare_AML_transactions.py -i <path to dataset>/HI-Small_Trans.csv -o <path to dataset>/HI-Small_Trans-c.csv
```

The preprocessed table is saved under `HI-Small_Trans-c.csv`. Now you can go ahead and run the `supervised.ipynb` and `slef-supervised.ipynb`. Additionally you can also take a look at `link-_prediction.ipynb`.

## License

MIT License

