
from src.nn.models import FTTransformerGINeFused
from torch_frame.datasets import Dota2

from icecream import ic

#dataset = Dota2(root='/mnt/data')
from src.datasets import IBMTransactionsAML
dataset = IBMTransactionsAML(root='/mnt/data/ibm-transactions-for-anti-money-laundering-aml/dummy-c.csv', pretrain='mask+lp')
dataset.materialize()
ic(dataset.tensor_frame.col_names_dict)

ic(dataset.tensor_frame.num_cols)
#import sys
#sys.exit()
model = FTTransformerGINeFused(
    channels=128,
    out_channels=None,
    col_stats=dataset.col_stats,
    col_names_dict=dataset.tensor_frame.col_names_dict,
    edge_dim=128*dataset.tensor_frame.num_cols,
    num_layers=3, 
    dropout=0.5,
    pretrain=True
)

learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
ic(learnable_params)
common_params_num = 0
tab_params_num = 0
gnn_params_num = 0
fuse_params_num = 0
decoder_params_num = 0
node_emb_params_num = 0
edge_emb_params_num = 0

lp_optimizer_num = 0
lp_optimizer_num_nwd = 0
mcm_optimizer_num = 0
mcm_optimizer_num_nwd = 0

for name, param in model.named_parameters():
    ic(name)
    if 'gnn' in name:
        gnn_params_num += param.numel()
    elif 'tab' in name:
        tab_params_num += param.numel()
    elif 'fuse' in name:
        fuse_params_num += param.numel()
    elif 'decoder' in name:
        decoder_params_num += param.numel()
    elif 'node_emb' in name:
        node_emb_params_num += param.numel()
    elif 'edge_emb' in name:
        edge_emb_params_num += param.numel()
    else:
        common_params_num += param.numel()

    no_decay = ['bias', 'LayerNorm.weight']
    if 'tab_conv' not in name and 'mcm_decoder' not in name and not any(nd in name for nd in no_decay):
        lp_optimizer_num += param.numel()
    elif 'tab_conv' not in name and 'mcm_decoder' not in name and any(nd in name for nd in no_decay):
        lp_optimizer_num_nwd += param.numel()

    if 'gnn_conv' not in name and 'lp_decoder' not in name and not any(nd in name for nd in no_decay):
        mcm_optimizer_num += param.numel()
    elif 'gnn_conv' not in name and 'lp_decoder' not in name and any(nd in name for nd in no_decay):
        mcm_optimizer_num_nwd += param.numel()

ic(common_params_num, tab_params_num, gnn_params_num, fuse_params_num, decoder_params_num, node_emb_params_num, edge_emb_params_num)
ic(lp_optimizer_num, lp_optimizer_num_nwd, mcm_optimizer_num, mcm_optimizer_num_nwd)
