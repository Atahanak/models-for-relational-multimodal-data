from src.nn.models.ft_transformer import FTTransformer

from icecream import ic
from torch_frame.datasets import Yandex
from torch_frame.data import DataLoader
import sys

from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame import TensorFrame, stype

dataset = Yandex(root='/tmp/yandex', name='adult')
ic(dataset)
# ic(dataset.feat_cols)
dataset.materialize()
is_classification = dataset.task_type.is_classification

train_dataset, val_dataset, test_dataset = dataset.split()
train_tensor_frame = train_dataset.tensor_frame
train_loader = DataLoader(train_tensor_frame, batch_size=1, shuffle=True)
example = next(iter(train_loader))
# ic(example)
# ic(example.get_col_feat('C_feature_1'))
# ic(example.get_col_feat('N_feature_1'))

numerical_encoder = LinearEncoder()
stype_encoder_dict = {
    stype.categorical: EmbeddingEncoder(),
    stype.numerical: numerical_encoder
}

if is_classification:
    output_channels = dataset.num_classes
else:
    output_channels = 1

model = FTTransformer(
    channels=32,
    out_channels=output_channels,
    num_layers=3,
    col_stats=dataset.col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
    stype_encoder_dict=stype_encoder_dict,
    pretrain=True
)

pred = model(example)
ic(example.y)
ic(pred[0])
ic(pred[1])