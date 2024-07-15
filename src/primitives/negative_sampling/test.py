import sys
sys.path.append('./build/')
import negative_samples as example
import json

# Read from the file
with open('edge_index.json', 'r') as f:
    edge_index = json.load(f)
with open('pos_edge_index.json', 'r') as f:
    pos_edge_index = json.load(f)
num_neg_samples = 64

import time
start = time.time()
neg_edges = example.generate_negative_samples(edge_index, pos_edge_index, num_neg_samples)
print('Time taken: ', time.time() - start)
print(len(neg_edges), len(neg_edges[0]))
