import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose


class AdapterLayer(BaseTunerLayer):
    adapter_layer_names = ["bottleneck"]

    def __init__(self, in_features: int, out_features: int, rank: int, **kwargs):
        self._disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.bottleneck = nn.ModuleDict({})
        self.rank = rank
        self.kwargs = kwargs

    def update_layer(self, adapter_name):
        self.bottleneck[adapter_name] = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.rank),
            nn.ReLU(),
            nn.Linear(in_features=self.rank, out_features=self.in_features)
        )

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)


class Linear(nn.Linear, AdapterLayer):

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        rank: int,
        fan_in_fan_out: bool = False  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    ) -> None:

        super(nn.Linear, self).__init__()

        AdapterLayer.__init__(self, in_features=in_features, out_features=out_features, rank=rank)

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name)
        self.set_adapter(adapter_name)

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            result = self._linear(x)
        else:
            result = self._linear(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.bottleneck.keys():
                    continue
                bottleneck = self.bottleneck[active_adapter]
                result = result.to(bottleneck[0].weight.dtype)
                residual = result.clone()
                result = bottleneck(result) + residual

        result = result.to(previous_dtype)
        return result
