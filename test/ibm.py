from src.datasets.ibm_transactions_for_aml import IBMTransactionsAML

from icecream import ic
from torch_frame.data import DataLoader

dataset = IBMTransactionsAML('/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy.csv', pretrain=True)
dataset.materialize()
#is_classification = dataset.task_type.is_classification
#ic(is_classification)
ic(dataset.col_stats['Payment Currency'])
ic(dataset.col_stats['Amount Paid'])
ic(dataset.df.head(5))

train_dataset, val_dataset, test_dataset = dataset.split()
train_tensor_frame = train_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=32, shuffle=True)
example = next(iter(train_loader))
ic(example)
ic(example.get_col_feat('Payment Currency'))
ic(example.get_col_feat('Amount Paid'))