"""The data stored in pickle format with version: 0.7.5 (python 3.7).
The type of graph object：networkx.classes.multidigraph.MultiDiGraph
Numbers of nodes: 2973489
Numbers of edges: 13551303
Average degree:   4.5574
Nodes' features：
    // The label. 1 means fishing mark node, otherwise 0.
    G.nodes[nodeName]['isp']；

Edges' features:
    G[node1][node2][0]['amount']        // The amount mount of the transaction.
    G[node1][node2][0]['timestamp']     // The timestamp of the transaction.				
							
* Notes * 
"""

import logging
# Configure logging
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
import pickle 
import networkx as nx
import time
from datetime import datetime, timezone
import os
import pandas as pd

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
logger.info("Loading the graph")
start = time.time()
G = load_pickle('/mnt/data/ethereum-phishing-transaction-network/MulDiGraph.pkl')
logger.info(f"Graph loaded in {time.time()-start} seconds.")
logger.info(f'Number of nodes: {G.number_of_nodes()}')
logger.info(f'Number of edges: {G.number_of_edges()}')

logger.info("Processing the graph")
uniq = set()
start = time.time()
min_time = 1e20
max_time = 0
from_address = []
to_address = []
time_stamps = []
amounts = []

for ind, edge in enumerate(nx.edges(G)):
    (u, v) = edge
    eg = G[u][v][0]
    amo, tim = eg['amount'], eg['timestamp']
    uniq.add((u, v, tim))
    #uniq.add((u, v, amo, tim))
    min_time = min(min_time, tim)
    max_time = max(max_time, tim)
min_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(min_time))
max_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(max_time))
# print(min_time, max_time)
# print(len(uniq))
logger.info(f"Graph processed in {time.time()-start} seconds.")
logger.info(f'Number of unique transactions: {len(uniq)}')
#print an element from uniq
logger.info(f"An element from uniq: {list(uniq)[0]}")
logger.info(f'Minimum timestamp: {min_time}')
logger.info(f'Maximum timestamp: {max_time}')

transactions_path = '/mnt/data/ethereum-phishing-transaction-network/transactions.csv'
if os.path.exists(transactions_path):
    columns_to_read = ['from_address', 'to_address', 'value', 'block_timestamp']
    columns_to_read = ['from_address', 'to_address', 'block_timestamp']
    dtype_dict = {
        'from_address': 'str',
        'to_address': 'str',
        #'value': 'str',
        'block_timestamp': 'str'  # assuming block_timestamp is initially read as string
    }
    df = pd.read_csv(transactions_path, usecols=columns_to_read, dtype=dtype_dict) 
    logger.info(f"Read {len(df)} transactions from the csv")
    #df['value'] = df['value'].astype(float)
    df['block_timestamp'] = pd.to_datetime(df['block_timestamp']).view('int64') / 10**9
    # print a row of the dataframe
    logger.info(f"A row from the dataframe: {df.iloc[0]}")
    #exists = set(zip(df['from_address'], df['to_address'], df['value'], df['block_timestamp']))
    exists = set(zip(df['from_address'], df['to_address'], df['block_timestamp']))
    uniq = uniq - exists
logger.info(f"Removed {len(exists)} transactions that are already in the csv")
logger.info(f"Remaining transactions: {len(uniq)}")

uniql = list(uniq)
from_address = [x[0] for x in uniql]
to_address = [x[1] for x in uniql]
#amounts = [x[2] for x in uniql]
time_stamps = [datetime.fromtimestamp(x[2], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z") for x in uniql]

from google.cloud import bigquery
client = bigquery.Client(project='windy-nation-428806-q0')

def get_transactions(from_address, to_address, time_stamps):
    assert len(from_address) == len(to_address) == len(time_stamps)

    conditions = " OR ".join(
        f'(from_address = "{from_addr}" AND to_address = "{to_addr}" AND block_timestamp = TIMESTAMP("{time_stamp}"))'
        for from_addr, to_addr, time_stamp in zip(from_address, to_address, time_stamps)
    )
    #print(conditions)

    query = (
        'SELECT `hash`, nonce, transaction_index, from_address, to_address, value, gas, gas_price, receipt_gas_used, receipt_contract_address, receipt_status, block_timestamp, block_number, max_fee_per_gas, max_priority_fee_per_gas, transaction_type, receipt_effective_gas_price FROM `bigquery-public-data.crypto_ethereum.transactions`'
        f'WHERE block_timestamp BETWEEN TIMESTAMP("2015-08-07") AND TIMESTAMP("2019-01-20") AND ({conditions})'
    )
    #print(query)

    query_job = client.query(query)  # API request
    rows = list(query_job.result())  # Waits for query to finish
    return rows

# write rows to csv
import csv
import time
batch = 30
read = 0
start = time.time()
with open(transactions_path, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['hash', 'nonce', 'transaction_index', 'from_address', 'to_address', 'value', 'gas', 'gas_price', 'receipt_gas_used', 'receipt_contract_address', 'receipt_status', 'block_timestamp', 'block_number', 'max_fee_per_gas', 'max_priority_fee_per_gas', 'transaction_type', 'receipt_effective_gas_price'])
    writer.writeheader()
    for i in range(0, 100000, batch):
        start = time.time()
        rows = get_transactions(from_address[i:i+batch], to_address[i:i+batch], time_stamps[i:i+batch])
        read += len(rows)
        for row in rows:
           writer.writerow(row)
        writer.writerow(row)
        logger.info(f"Processed transactions {i}-{i+batch} in {time.time()-start} seconds")
        logger.info(f"Found {len(rows)} transactions, missing {batch-len(rows)} transactions")        
        logger.info(f"Total transactions processed: {read}")
logger.info(f"Total time taken: {time.time()-start}")

