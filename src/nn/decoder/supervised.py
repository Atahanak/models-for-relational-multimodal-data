from torch.nn import Module, Sequential, LayerNorm, ReLU, Linear
from torch import Tensor

class SupervisedHead(Module):
    r"""Used for supervised training of the FT-Transformer model."""
    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, out_channels),
        )
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        for m in self.decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, x_cls: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x_cls (torch.Tensor): Output of the transformer.

        Returns:
            torch.Tensor: Output of the model.
        """
        return self.decoder(x_cls)