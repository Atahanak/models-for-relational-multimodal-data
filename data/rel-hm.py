import sys
sys.path.append('../')
import logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Specify the log message format
    datefmt='%Y-%m-%d %H:%M:%S',  # Specify the date format
    handlers=[
        #logging.FileHandler('app.log'),  # Log messages to a file
        logging.StreamHandler()  # Also output log messages to the console
    ]
)
logger = logging.getLogger(__name__)
# import time
# from src.datasets.util.mask import PretrainType
# from src.datasets import RelHM
# dataset = RelHM(
#     root='/mnt/data/rel-hm/rel-hm.csv', 
#     pretrain={PretrainType.MASK, PretrainType.LINK_PRED},
#     split_type='daily_temporal',
#     splits=[0.6,0.2,0.2], 
#     khop_neighbors=[100,100],
#     ports=False
# )
# logger.info(f"Materialzing dataset...")
# s = time.time()
# dataset.materialize()
# logger.info(f"Materialized in {time.time() - s:.2f} seconds")
# dataset.df.head(5)
# num_columns = len(dataset.num_columns)
# cat_columns = len(dataset.cat_columns)
# logger.info(f"Number of numerical columns: {num_columns}")
# logger.info(f"Number of categorical columns: {cat_columns}")

from src.datasets import EllipticBitcoin, IBMTransactionsAML

dataset = EllipticBitcoin(root='/mnt/data/elliptic_bitcoin_dataset/', ports=True)
# dataset = IBMTransactionsAML(
#     root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', 
#     split_type='daily_temporal',
#     splits=[0.6,0.2,0.2], 
#     khop_neighbors=[100,100],
#     ports=True
# )