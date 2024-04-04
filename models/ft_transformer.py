from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential, ModuleList

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import FTTransformerConvs
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

class FTTransformer(Module):
    r"""The FT-Transformer model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using FTTransformer, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_layers (int): Number of layers.  (default: :obj:`3`)
        col_stats(dict[str,dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:obj:`torch_frame.stype`, list[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call
            :class:`torch_frame.nn.encoder.EmbeddingEncoder()` for categorical
            feature and :class:`torch_frame.nn.encoder.LinearEncoder()`
            for numerical feature)
        pretrain (bool): If :obj:`True`, the model will be pre-trained, otherwise it will be trained end-to-end. (default: :obj:`False`)
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder] | None = None,
        pretrain: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")
        self.pretrain = pretrain
        if pretrain:
            num_numerical = len(col_names_dict[stype.numerical])
            #num_categorical = len(col_names_dict[stype.categorical])
            #ic(num_numerical, num_categorical)
            num_classes = [len(col_stats[col][StatType.COUNT][0]) for col in col_names_dict[stype.categorical]]
            #ic(num_classes)

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        self.backbone = FTTransformerConvs(channels=channels,
                                           num_layers=num_layers)
        
        if pretrain:
            self.num_decoder = Sequential(
                LayerNorm(channels),
                ReLU(),
                Linear(channels, num_numerical),
            )
            self.cat_decoder = ModuleList([Sequential(
                LayerNorm(channels),
                ReLU(),
                Linear(channels, num_classes),
            ) for num_classes in num_classes])
        else:
            self.decoder = Sequential(
                LayerNorm(channels),
                ReLU(),
                Linear(channels, out_channels),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.backbone.reset_parameters()
        if self.pretrain:
            for m in self.num_decoder:
                if not isinstance(m, ReLU):
                    m.reset_parameters()
            for m in self.cat_decoder:
                for n in m:
                    if not isinstance(n, ReLU):
                        n.reset_parameters()
        else:
            for m in self.decoder:
                if not isinstance(m, ReLU):
                    m.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)
        x, x_cls = self.backbone(x)
        out = 0 #self.decoder(x_cls)
        return out

if __name__ == '__main__':
    from icecream import ic
    from torch_frame.datasets import Yandex
    from torch_frame.data import DataLoader
    import sys

    dataset = Yandex(root='/tmp/yandex', name='adult')
    ic(dataset)
    ic(dataset.feat_cols)
    dataset.materialize()
    is_classification = dataset.task_type.is_classification

    train_dataset, val_dataset, test_dataset = dataset.split()
    train_tensor_frame = train_dataset.tensor_frame
    train_loader = DataLoader(train_tensor_frame, batch_size=32, shuffle=True)
    example = next(iter(train_loader))
    ic(example)
    ic(example.get_col_feat('C_feature_1'))
    ic(example.get_col_feat('N_feature_1'))
    ic(example.y)
    sys.exit()

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