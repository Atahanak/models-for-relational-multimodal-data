from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn import TransformerEncoderLayer, LayerNorm


class RCTransformer(Module):
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
        num_layers: int,
        nhead: int = 8,
        feedforward_channels: int = None,
        dropout: float = 0.1,
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")
        
        row_backbone = ModuleList()
        row_norms = ModuleList()
        col_backbone = ModuleList()
        col_norms = ModuleList()
        for l in num_layers:
            row_backbone.append(TransformerEncoderLayer(
                d_model=channels,
                nhead=nhead,
                dim_feedforward=feedforward_channels or channels,
                dropout=dropout,
                activation=activation,
                # Input and output tensors are provided as
                # [batch_size, seq_len, channels]
                batch_first=True,
            ))
            row_norms.append(LayerNorm(channels))
            col_backbone.append(TransformerEncoderLayer(
                d_model=channels,
                nhead=nhead,
                dim_feedforward=feedforward_channels or channels,
                dropout=dropout,
                activation=activation,
                # Input and output tensors are provided as
                # [batch_size, seq_len, channels]
                batch_first=True,
            ))
            col_norms.append(LayerNorm(channels))

        self.reset_parameters()
        

    def reset_parameters(self) -> None:
        self.row_backbone.reset_parameters()
        self.col_backbone.reset_parameters()

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