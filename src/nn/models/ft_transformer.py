from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.nn import Module

import torch_frame
from src.datasets.util.mask import PretrainType
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.conv import FTTransformerConvs
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

from ..decoder import SelfSupervisedHead, SelfSupervisedMVHead
from ..decoder import SupervisedHead

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
        #out_channels: int,
        num_layers: int,
        # encoder,
        # decoder
        # col_stats: dict[str, dict[StatType, Any]],
        # col_names_dict: dict[torch_frame.stype, list[str]],
        # stype_encoder_dict: dict[torch_frame.stype, StypeEncoder] | None = None,
        #pretrain: set[PretrainType] = set(),
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")
        
        #self.encoder = encoder
        #self.decoder = decoder

        # if stype_encoder_dict is None:
        #     stype_encoder_dict = {
        #         stype.categorical: EmbeddingEncoder(),
        #         stype.numerical: LinearEncoder(),
        #     }

        # self.encoder = StypeWiseFeatureEncoder(
        #     out_channels=channels,
        #     col_stats=col_stats,
        #     col_names_dict=col_names_dict,
        #     stype_encoder_dict=stype_encoder_dict,
        # )
        
        self.backbone = FTTransformerConvs(channels=channels,
                                           num_layers=num_layers)
        
        # if pretrain:
        #     num_numerical = len(col_names_dict[stype.numerical])
        #     num_categorical = [len(col_stats[col][StatType.COUNT][0]) for col in col_names_dict[stype.categorical]]

        #     if PretrainType.MASK_VECTOR in pretrain:
        #         self.decoder = SelfSupervisedMVHead(channels, num_numerical, num_categorical)
        #     else:
        #         self.decoder = SelfSupervisedHead(channels, num_numerical, num_categorical)
        #     # self.num_decoder = Sequential(
        #     #     LayerNorm(channels),
        #     #     ReLU(),
        #     #     Linear(channels, num_numerical),
        #     # )
        #     # self.cat_decoder = ModuleList([Sequential(
        #     #     LayerNorm(channels),
        #     #     ReLU(),
        #     #     Linear(channels, num_classes),
        #     # ) for num_classes in num_categorical])
        # else:
        #     # self.decoder = Sequential(
        #     #     LayerNorm(channels),
        #     #     ReLU(),
        #     #     Linear(channels, out_channels),
        #     # )
        #     self.decoder = SupervisedHead(channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #self.encoder.reset_parameters()
        self.backbone.reset_parameters()
        #self.decoder.reset_parameters()

    def forward(self, x) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        #x, _ = self.encoder(tf)
        x, x_cls = self.backbone(x)
        return x, x_cls
        # if self.pretrain:
        #     # num_out = self.num_decoder(x_cls)
        #     # cat_out = [decoder(x_cls) for decoder in self.cat_decoder]
        #     # out = (num_out, cat_out)
        # else:
        # out = self.decoder(x_cls)
        # return out