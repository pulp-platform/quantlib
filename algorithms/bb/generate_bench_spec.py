from dataclasses import dataclass
from pathlib import Path
from itertools import product
from copy import deepcopy
from typing import Union, Tuple, List

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


def int_log2(val):
    return int(np.ceil(np.log2(val)))

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

    @property
    def macs(self):
        macs = self.k**2 * self.out_size**2 * self.ch_in * self.ch_out
        if self.dw:
            macs = macs//self.groups
        return macs

    def norm_macs(self, precs):
        return self.macs * max(precs)//2


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

def build_ident_dict(network : fx.GraphModule, *shapes_in, dtype_in : torch.dtype = torch.float32, tracer : callable = None):
    from quantlib.editing.fx.passes import ShapePropPass

    if tracer is not None:
        network = tracer(network)
    spp = ShapePropPass(*shapes_in, dtype_in=dtype_in)
    gm = spp(network)
    ident_dict = {}
    for name, n, m in named_module_nodes(gm):
        l_ident = ident_of_layer(n, m)
        if l_ident is not None:
            ident_dict[name] = l_ident

    return ident_dict

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
        #print(f"Layer {identifier}:\nmax latency for (bi, bw)={curr_max_lat_cfg} is {curr_max_latency}\nmin latency for (bi, bw)={curr_min_lat_cfg} is {curr_min_latency}")
    results_by_layer['max_latency'] = max_latency
    return results_by_layer

def build_net_latency_dict(net : nn.Module, shape_in : Union[Tuple[int], List[int], torch.Size], latency_spec : str, input_prec : int = 8):
    # nasty hack to avoid circular imports...
    from quantlib.editing.fx.passes.bb import BB_symbolic_trace, BBControllerPrepPass, find_layer_sets
    if not isinstance(net, fx.GraphModule):
        net = BB_symbolic_trace(net)
    prep_pass = BBControllerPrepPass(shape_in=shape_in, latency_spec=latency_spec)
    net = prep_pass(net)
    prop_dict = deepcopy(prep_pass.property_dict)
    layer_pairs = find_layer_sets(net)
    layer_pd = {}
    for lp in layer_pairs:
        if len(lp) == 2:
            lin_k = lp[1]
            act_k = lp[0]
        else:
            lin_k = lp[0]
            act_k = 'input'
            # prune away precision configs that can't be used, i.e. all configurations
            # with act_prec != input_prec
            lat_dict = {(ap, wp): lat for (ap, wp), lat in prop_dict[lin_k]['latency'].items() if ap == input_prec}
            prop_dict[lin_k]['latency'] = lat_dict

        layer_pd[lin_k] = {'linked_act' : act_k, 'props' : prop_dict[lin_k]}
    pd_out = {'input_prec' : input_prec, 'layers': layer_pd}

    return net, pd_out

def add_prec_spec(gm : fx.GraphModule, pd : dict, spec_dict : dict):

    is_datap = all(k.startswith('module.') for k in spec_dict["layer_levels"].keys())
    levels_dict = {k.rstrip('$') : v for k, v in spec_dict["layer_levels"].items()}
    if is_datap:
        levels_dict = {k.lstrip("module.") : v for k, v in levels_dict.items()}

    ld = pd['layers']
    kl = [k for k in ld.keys()]
    tot_latency = 0
    latency_8b = 0
    latency_4b = 0
    input_prec = pd['input_prec']

    for k in kl:
        lin_prec = int_log2(levels_dict[k])
        act_name = ld[k]['linked_act']
        if act_name == 'input':
            act_prec = input_prec
        else:
            try:
                act_prec = int_log2(levels_dict[act_name])
            except KeyError:
                # the activation is not a BB activation ==> get it directly
                # from the network
                print(f"Activation {act_name} is not in the precision dict - reading from net")
                act = gm.get_submodule(act_name)
                act_prec = int_log2(act.n_levels)
                # remove the unavailable precisions from the latency dict
                ld[k]['props']['latency'] = {(ap, wp) : lat for (ap, wp), lat in ld[k]['props']['latency'].items() if ap == act_prec}
        ld[k]['props']['prec_from_spec'] = (act_prec, lin_prec)
        latency_from_spec = ld[k]['props']['latency'][(act_prec, lin_prec)]
        ld[k]['props']['latency_from_spec'] = latency_from_spec
        tot_latency += latency_from_spec
        try:
            latency_8b += ld[k]['props']['latency'][(8,8)]
        except KeyError:
            latency_8b += ld[k]['props']['latency'][(input_prec,8)]
        try:
            latency_4b += ld[k]['props']['latency'][(4,4)]
        except KeyError:
            latency_4b += ld[k]['props']['latency'][(input_prec,4)]
    pd['tot_latency_from_spec'] = tot_latency
    pd['latency_8b'] = latency_8b
    pd['latency_4b'] = latency_4b
    return pd

def add_freebie_cfg(pd : dict):
    ld = pd['layers']
    kl = [k for k in ld.keys()]
    n_fl = 0
    tot_savings = 0

    for k in kl:
        cur_lat = ld[k]['props']['latency_from_spec']
        cur_precs = ld[k]['props']['prec_from_spec']
        cur_lat_dict = ld[k]['props']['latency']
        better_lats = {(ap,wp):v for (ap, wp), v in cur_lat_dict.items() if ap >= cur_precs[0] and wp >= cur_precs[1] and v < cur_lat}
        if len(better_lats):
            freebie_cfg, freebie_lat = min(better_lats.items(), key=lambda kv:kv[1])
            savings = cur_lat - freebie_lat
            n_fl += 1
            ld[k]['props']['freebie_precs'] = freebie_cfg
            ld[k]['props']['freebie_savings_abs'] = savings
            ld[k]['props']['freebie_savings_rel'] = savings/cur_lat
            tot_savings += savings
    tot_freebie_lat = pd['tot_latency_from_spec'] - tot_savings
    pd['tot_freebie_latency'] = tot_freebie_lat
    pd['num_freebie_layers'] = n_fl
    pd['freebie_lat_saved_abs'] = tot_savings
    pd['freebie_lat_saved_rel'] = tot_savings/pd['tot_latency_from_spec']

    return pd
