import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from hydra.utils import instantiate
from torch.nn.utils import parametrizations
from .mixture_layers import InputLayer, ParallelLayer, OrthogonalLayer1D
from typing import List, Optional, Union, Dict, Any
from tensordict import from_modules
from copy import deepcopy


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.ln(x)
        if self.act:
            x = self.act(x)
        return x

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, simnorm_dim_or_cfg):
        super().__init__()
        # Support both old interface (cfg) and new interface (simnorm_dim)
        if hasattr(simnorm_dim_or_cfg, 'simnorm_dim'):
            self.dim = simnorm_dim_or_cfg.simnorm_dim
        else:
            self.dim = simnorm_dim_or_cfg

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        units: List[int],
        activation_cfg: dict = {"_target_": "torch.nn.Mish", "inplace": True},
        activation_cfg_last: Optional[dict] = None,
        spectral: bool = False,
    ):
        super(MLP, self).__init__()
        self.layer_dims = [input_dim] + units + [output_dim]

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))
                modules.append(instantiate(activation_cfg))

        if spectral:
            modules[-1] = spectral_norm(modules[-1])

        if activation_cfg_last:
            modules.append(activation_cfg_last)

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class SpectralNormLinear(nn.Module):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(
        self, in_features, out_features, dropout=0.0, act=None, layer_norm=True
    ):
        super(SpectralNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        linear_layer = nn.Linear(in_features, out_features)
        if layer_norm:
            self.layer_norm = True
            self.ln = nn.LayerNorm(self.out_features)
        else:
            self.layer_norm = False
        # Apply spectral normalization
        self.linear = parametrizations.spectral_norm(linear_layer)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        if self.layer_norm:
            x = self.ln(x)
        if self.act:
            x = self.act(x)
        return x

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        repr_act = f"{self.act.__class__.__name__}" if self.act else ""
        return (
            f"SpectralNormLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={repr_act})"
        )


LAYERS = {
    "linear": nn.Linear,
    "normedlinear": NormedLinear,
    "spectralnormlinear": SpectralNormLinear,
}


def mlp(in_dim, mlp_dims, out_dim, last_layer, last_layer_kwargs, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))

    # TODO this is hot-bodged. come up with a better way
    layer_cls = LAYERS[last_layer.lower()]
    mlp.append(layer_cls(dims[-2], dims[-1], **last_layer_kwargs))
    return nn.Sequential(*mlp)


class MoEBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        gate_dim: int,
        units: Union[int, List[int]],
        out_dim: int,
        *,
        n_experts: int = 4,
        head_last_layer: str = "normedlinear",
        head_last_layer_kwargs: Optional[Dict[str, Any]] = None,
        use_orthogonal: bool = False,
        tau_init: float = 1.0,          # softmax 初始温度
        tau_decay: float = 0.99995,     # 每次 forward 乘的衰减系数
    ):
        super().__init__()
        self.n_experts = n_experts
        self.tau         = tau_init          # ← ① 记录当前温度
        self.tau_decay   = tau_decay

        # ---------------- Gate ----------------
        self.gate = nn.Linear(gate_dim, n_experts, bias=False)
        # 打破完全对称
        with torch.no_grad():
            self.gate.weight += 1e-3 * torch.randn_like(self.gate.weight)

        # ---------------- Expert MLP ----------------
        if isinstance(units, int):
            units = [units]
        expert_layers: List[nn.Module] = []
        last = in_dim
        for h in units:
            expert_layers.append(NormedLinear(last, h))
            last = h
        self.unit_model = nn.Sequential(*expert_layers)

        self.trunk = nn.Sequential(
            InputLayer(n_models=n_experts),   # 复制 inputs → [K,N,D]
            ParallelLayer(self.unit_model)      # 并行推理        [K,N,H]
        )
        if use_orthogonal:
            self.trunk = nn.Sequential(self.trunk, OrthogonalLayer1D())

        # ---------------- Head ----------------
        if head_last_layer_kwargs is None:
            head_last_layer_kwargs = {}
        layer_cls = LAYERS[head_last_layer.lower()]
        head_layers: List[nn.Module] = [
            layer_cls(units[-1], out_dim, **head_last_layer_kwargs)
        ]
        self.head = nn.Sequential(*head_layers)


    def forward(self, z, a=None, task_emb=None):
        if a is not None:
            x = torch.cat([z, a], dim=-1)
        else:
            x = z
        if task_emb is not None:
            # broadcast task_emb 到与 x 对齐
            if task_emb.ndim == 2 and x.ndim == 3:
                task_emb = task_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
            elif task_emb.ndim == 2 and x.ndim == 2 and task_emb.shape[0] == 1:
                task_emb = task_emb.repeat(x.shape[0], 1)
            x = torch.cat([x, task_emb], dim=-1)

        # -------- 扁平序列 -------------------------------------------
        is_seq = (x.ndim == 3)
        if is_seq:
            T, B, D = x.shape
            x = x.view(T * B, D)

        # -------- Expert 并行 & Gate ---------------------------------
        feats = self.trunk(x)              # [K,N,H]
        feats = feats.permute(1, 0, 2)     # [N,K,H]

        gate_in = task_emb if task_emb is not None else x
        logits  = self.gate(gate_in) / self.tau          # ← ② 加温度
        w       = F.softmax(logits, dim=-1)

        if self.training:
            # 每个 batch 求一次熵，存到一个 buffer 里
            entropy = (-w * w.clamp_min(1e-8).log()).sum(-1).mean()  # 标量
            # 把它挂到模块上，外面就能访问到
            self._last_entropy = entropy.detach()
            # Only decay temperature during training, not during every forward pass
            self.tau = max(self.tau * self.tau_decay, 0.5)  # ← ③ 退火到下限 0.5

        # -------- 聚合 & Head ----------------------------------------
        out = (w.unsqueeze(-1) * feats).sum(dim=1)  # [N,H]
        out = self.head(out)

        if is_seq:
            out = out.view(T, B, -1)
        return out


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        # combine_state_for_ensemble causes graph breaks
        self.params = from_modules(*modules, as_module=True)
        with self.params[0].data.to("meta").to_module(modules[0]):
            self.module = deepcopy(modules[0])
        self._repr = str(modules[0])
        self._n = len(modules)

    def __len__(self):
        return self._n

    def _call(self, params, *args, **kwargs):
        with params.to_module(self.module):
            return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

    def __repr__(self):
        return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad
        self.padding = tuple([self.pad] * 4)

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        x = F.pad(x, self.padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.).sub(0.5)


# Compatibility function for layers.py interface
def mlp_legacy(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    Compatible with layers.py interface.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp_layers = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp_layers.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
    mlp_layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp_layers)


def conv(in_shape, num_channels, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """
    assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(), PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)


def enc(cfg, out={}):
    """
    Returns a dictionary of encoders for each observation in the dict.
    """
    for k in cfg.obs_shape.keys():
        if k == 'state':
            out[k] = mlp_legacy(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
        elif k == 'rgb':
            out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
        else:
            raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
    return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
    """
    Converts a checkpoint from our old API to the new torch.compile compatible API.
    """
    # check whether checkpoint is already in the new format
    if "_detach_Qs_params.0.weight" in source_state_dict:
        return source_state_dict

    name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
    new_state_dict = dict()

    # rename keys
    for key, val in list(source_state_dict.items()):
        if key.startswith('_Qs.'):
            num = key[len('_Qs.params.'):]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            new_total_key = "_Qs.params." + new_key
            del source_state_dict[key]
            new_state_dict[new_total_key] = val
            new_total_key = "_detach_Qs_params." + new_key
            new_state_dict[new_total_key] = val
        elif key.startswith('_target_Qs.'):
            num = key[len('_target_Qs.params.'):]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            new_total_key = "_target_Qs_params." + new_key
            del source_state_dict[key]
            new_state_dict[new_total_key] = val

    # add batch_size and device from target_state_dict to new_state_dict
    for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
        for key in ('__batch_size', '__device'):
            new_key = prefix + 'params.' + key
            new_state_dict[new_key] = target_state_dict[new_key]

    # check that every key in new_state_dict is in target_state_dict
    for key in new_state_dict.keys():
        assert key in target_state_dict, f"key {key} not in target_state_dict"
    # check that all Qs keys in target_state_dict are in new_state_dict
    for key in target_state_dict.keys():
        if 'Qs' in key:
            assert key in new_state_dict, f"key {key} not in new_state_dict"
    # check that source_state_dict contains no Qs keys
    for key in source_state_dict.keys():
        assert 'Qs' not in key, f"key {key} contains 'Qs'"

    # copy log_std_min and log_std_max from target_state_dict to new_state_dict
    new_state_dict['log_std_min'] = target_state_dict['log_std_min']
    new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
    new_state_dict['_action_masks'] = target_state_dict['_action_masks']

    # copy new_state_dict to source_state_dict
    source_state_dict.update(new_state_dict)

    return source_state_dict
