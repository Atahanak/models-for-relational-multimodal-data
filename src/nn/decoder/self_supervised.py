from torch.nn import Module, Sequential, LayerNorm, ReLU, Linear, ModuleList, Softmax
from torch import Tensor
from ..gnn.decoder import LinkPredHead


class SelfSupervisedHead(Module):
    r"""Used for pretraining the FT-Transformer model."""
    def __init__(self, channels: int, num_numerical: int, num_categorical: list[int]) -> None:
        super().__init__()
        self.num_decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, num_numerical),
        )
        self.cat_decoder = ModuleList([Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, num_classes),
        ) for num_classes in num_categorical])
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        for m in self.num_decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()
        for m in self.cat_decoder:
            for n in m:
                if not isinstance(n, ReLU):
                    n.reset_parameters()

    def forward(self, x_cls: Tensor) -> tuple[Tensor, list[Tensor]]:
        r"""Forward pass.

        Args:
            x_cls (torch.Tensor): Output of the transformer.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: Numerical and categorical
            outputs.
        """
        num_out = self.num_decoder(x_cls)
        cat_out = [decoder(x_cls) for decoder in self.cat_decoder]
        return num_out, cat_out


class SelfSupervisedMVHead(Module):
    r"""Used for pretraining the FT-Transformer model with mask vector prediction"""

    def __init__(self, channels: int, num_numerical: int, num_categorical: list[int]) -> None:
        super().__init__()
        self.mcm_decoder = SelfSupervisedHead(channels, num_numerical, num_categorical)
        self.mask_vector_decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, num_numerical + len(num_categorical))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.mcm_decoder.reset_parameters()
        for m in self.mask_vector_decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, x_cls: Tensor) -> tuple[Tensor, list[Tensor], Tensor]:
        r"""Forward pass.

        Args:
            x_cls (torch.Tensor): Output of the transformer.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]: Numerical and categorical
            outputs, and the prediction of the mask vector.
        """
        num_out, cat_out = self.mcm_decoder(x_cls)
        mv_out = self.mask_vector_decoder(x_cls)
        return num_out, cat_out, mv_out
    

class SelfSupervised_MCM_MV_LP_Head(Module):
    r"""Used for pretraining the FT-Transformer model with mask vector prediction"""

    def __init__(self, channels: int, num_numerical: int, num_categorical: list[int], nhidden: int, dropout: float) -> None:
        super().__init__()
        self.mv_decoder = SelfSupervisedMVHead(channels, num_numerical, num_categorical)
        self.lp_decoder = LinkPredHead(n_hidden=nhidden, final_dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.mv_decoder.reset_parameters()
        self.lp_decoder.reset_parameters()

    def forward(self, x_cls: Tensor, x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr) -> tuple[Tensor, list[Tensor], Tensor]:
        r"""Forward pass.

        Args:
            x_cls (torch.Tensor): Output of the transformer.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]: Numerical and categorical
            outputs, and the prediction of the mask vector.
        """
        num_out, cat_out, mv_out = self.mv_decoder(x_cls)
        pos_pred, neg_pred = self.lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
        return num_out, cat_out, mv_out, pos_pred, neg_pred


class SelfSupervisedLPHead(Module):
    r"""Used for pretraining the FT-Transformer model."""
    def __init__(self, channels: int, num_numerical: int, num_categorical: list[int], nhidden: int, dropout: float) -> None:
        super().__init__()
        self.mcm_decoder = SelfSupervisedHead(channels, num_numerical, num_categorical)
        self.lp_decoder = LinkPredHead(n_hidden=nhidden, final_dropout=dropout)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        self.mcm_decoder.reset_parameters()
        self.lp_decoder.reset_parameters()

    def forward(self, x_tab, x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr) -> tuple[Tensor, list[Tensor], tuple[Tensor, Tensor]]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Output of the fused model.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], tuple[Tensor, Tensor]]: numerical, categorical, link prediction
            outputs.
        """
        num_out, cat_out = self.mcm_decoder(x_tab)
        pos_pred, neg_pred = self.lp_decoder(x_gnn, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
        return num_out, cat_out, pos_pred, neg_pred

class MCMHead(Module):
    r"""Used for pretraining the FT-Transformer model."""
    def __init__(self, channels: int, num_numerical: int, num_categorical: list[int], w: int = 1) -> None:
        super().__init__()
        self.num_decoder = Sequential(
            LayerNorm(w*channels),
            ReLU(),
            Linear(channels, num_numerical),
        )
        self.cat_decoder = ModuleList([Sequential(
            LayerNorm(w*channels),
            ReLU(),
            Linear(channels, num_classes),
        ) for num_classes in num_categorical])
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        for m in self.num_decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()
        for m in self.cat_decoder:
            for n in m:
                if not isinstance(n, ReLU):
                    n.reset_parameters()

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Features.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: Numerical and categorical
            outputs.
        """
        num_out = self.num_decoder(x)
        cat_out = [decoder(x) for decoder in self.cat_decoder]
        return (num_out, cat_out)


class MVHead(Module):
    r"""Used for pretraining the FT-Transformer model with mask vector prediction"""

    def __init__(self, channels: int, num_numerical: int, num_categorical: list[int]) -> None:
        super().__init__()
        self.mask_vector_decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, num_numerical + len(num_categorical))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.mask_vector_decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, x_cls: Tensor) -> tuple[Tensor, list[Tensor], Tensor]:
        r"""Forward pass.

        Args:
            x_cls (torch.Tensor): Output of the transformer.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]: Numerical and categorical
            outputs, and the prediction of the mask vector.
        """
        mv_out = self.mask_vector_decoder(x_cls)
        return mv_out
