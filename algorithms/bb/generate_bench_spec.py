from dataclasses import dataclass
from pathlib import Path
from itertools import product

import torch
from torch import nn, fx
import pandas as pd
import numpy as np

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.bb.bb_ops import *
from quantlib.editing.fx.util import named_module_nodes

__all__ = ['LayerIdentifier',
           'extract_unique_layers',
           'export_bench_spec',
           'ident_of_layer']


@dataclass(frozen=True)
class LayerIdentifier:
    is_lin : bool
    k : tuple
    in_size : torch.Size
    out_size : torch.Size
    dw : bool
    groups : int
    padding : int
    stride : int
    ch_in : int
    ch_out : int

    def spec_dict(self, in_bits : int, out_bits : int, wt_bits : int):
        chin = max(int(np.ceil(self.ch_in/(8//in_bits))*8//in_bits), int(np.ceil(self.ch_in/(8//wt_bits))*8//wt_bits))
        chout = int(np.ceil(self.ch_out/(8//out_bits))*8//out_bits)
        return {'groups':self.groups,
                'DW': self.dw,
                'kernel_size': self.k,
                'chin': chin,
                'chout': chout,
                'input_size': self.in_size,
                'output_size': self.out_size,
                'padding': self.padding,
                'stride': self.stride,
                'IN_BITS': in_bits,
                'OUT_BITS': out_bits,
                'W_BITS': wt_bits}

    @classmethod
    def from_series(cls, df : pd.Series):
        is_lin = df['kernel_size'].item() == 1 and df['input_size'].item() == 1 and df['output_size'].item() == 1
        return cls(is_lin, *[df[k].item() for k in ['kernel_size', 'input_size', 'output_size', 'DW', 'groups', 'padding', 'stride', 'chin', 'chout']])


def ident_of_conv2d(m : nn.Module, in_size : torch.Size, out_size : torch.Size):
    dw = m.in_channels == m.groups and m.in_channels == m.out_channels
    assert m.kernel_size[0] == m.kernel_size[1], f"ident_of_conv2d got module with non-quadratic kernel_size {m.kernel_size} - this is currently not supported"
    assert in_size[2] == in_size[3], f"ident_of_conv2d got non-quadratic input shape {in_size[2:]} - this is currently not supported"
    assert out_size[2] == out_size[3], f"ident_of_conv2d got non-quadratic output shape {out_size[2:]} - this is currently not supported"
    return LayerIdentifier(False, m.kernel_size[0], in_size[2], out_size[2], dw, m.groups, int(m.padding[0]), int(m.stride[0]), m.in_channels, m.out_channels)

def ident_of_lin(m : nn.Module, in_size : torch.Size, out_size : torch.Size):
    # linear layers are described as K=in_size=out_size=1
    return LayerIdentifier(True, 1, 1, 1, False, 1, 0, 1, m.in_features, m.out_features)

__LAYER_IDENT_FUNS = {
    nn.Conv2d : ident_of_conv2d,
    PACTConv2d : ident_of_conv2d,
    BBConv2d : ident_of_conv2d,
    nn.Linear : ident_of_lin,
    PACTLinear : ident_of_lin,
    BBLinear : ident_of_lin
}

def ident_of_layer(n : fx.Node, m : nn.Module):
    try:
        ident_fun = __LAYER_IDENT_FUNS[type(m)]
    except KeyError:
        ident_fun = lambda m, in_s, out_s : None

    return ident_fun(m, n.meta['shape_in'], n.meta['tensor_meta'].shape)

def extract_unique_layers(network : fx.GraphModule, *shapes_in, dtype_in : torch.dtype = torch.float32, tracer : callable = None):
    from quantlib.editing.fx.passes import ShapePropPass

    if tracer is not None:
        network = tracer(network)
    spp = ShapePropPass(*shapes_in, dtype_in=dtype_in)
    gm = spp(network)
    unique_layers = set()
    for name, n, m in named_module_nodes(gm):
        l_ident = ident_of_layer(n, m)
        if l_ident is not None:
            unique_layers.add(l_ident)
            print(f"Layer {name} has identifier {l_ident}")

    return unique_layers

def export_bench_spec(layers : set, export_folder : str, out_fn : str, in_bits : list, wt_bits : list):
    out_df = pd.DataFrame(columns=['groups', 'DW', 'kernel_size', 'chin', 'chout', 'input_size', 'output_size', 'IN_BITS', 'OUT_BITS', 'W_BITS'])
    for ib, wb in product(in_bits, wt_bits):
        for l in layers:
            out_df = out_df.append(l.spec_dict(in_bits=ib, out_bits=8, wt_bits=wb), ignore_index=True)

    out_path = Path(export_folder)
    out_path.mkdir(exist_ok=True, parents=True)
    out_df.to_excel(out_path.joinpath(out_fn))

def import_bench_results(filename : str, latency_key : str = 'latency'):
    # returns a dict of dicts:
    # {identifier: {(prec_in, prec_W): latency}}
    result_df = pd.read_excel(filename)
    result_df['identifier'] = list(zip(*tuple(result_df[k] for k in ['groups', 'DW', 'kernel_size', 'chin', 'chout', 'input_size', 'output_size', 'padding', 'stride'])))
    unique_idents = result_df['identifier'].unique()
    results_by_layer = {}
    max_latency = 0
    for ui in unique_idents:
        ident_stats = result_df[result_df['identifier'] == ui]
        identifier = LayerIdentifier.from_series(ident_stats.iloc[0])
        curr_min_latency = 100000000000000
        curr_max_latency = 0
        curr_min_lat_cfg = (-1, -1)
        curr_max_lat_cfg = (-1, -1)
        curr_prec_dict = {}
        for idx in range(len(ident_stats.index)):
            bi, bw = ident_stats.iloc[idx]['IN_BITS'].item(), ident_stats.iloc[idx]['W_BITS'].item()
            curr_latency = ident_stats.iloc[idx][latency_key].item()
            curr_prec_dict[(bi, bw)] = curr_latency
            if curr_latency > max_latency:
                max_latency = curr_latency
            if curr_latency > curr_max_latency:
                curr_max_latency = curr_latency
                curr_max_lat_cfg = (bi, bw)
            if curr_latency < curr_min_latency:
                curr_min_latency = curr_latency
                curr_min_lat_cfg = (bi, bw)
        results_by_layer[identifier] = curr_prec_dict
        print(f"Layer {identifier}:\nmax latency for (bi, bw)={curr_max_lat_cfg} is {curr_max_latency}\nmin latency for (bi, bw)={curr_min_lat_cfg} is {curr_min_latency}")
    results_by_layer['max_latency'] = max_latency
    return results_by_layer



