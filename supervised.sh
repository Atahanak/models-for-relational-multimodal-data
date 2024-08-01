data=/mnt/data/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv
batch_size=4096

python cagri_train.py --batch_size 4096 --data /mnt/data/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans-c.csv --model gin --save_model