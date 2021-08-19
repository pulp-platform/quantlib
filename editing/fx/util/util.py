from torch import fx

__all__ = ['gm_modules',
           'module_of_node',
           'get_qualified_prefix']

def gm_modules(gm : fx.GraphModule):
    return dict(gm.named_modules())

def module_of_node(gm : fx.GraphModule, node : fx.Node):
    assert node.op == "call_module", "module_of_node can only be called on 'call_module' nodes!"

    return gm_modules(gm)[node.target]

def get_qualified_prefix(target : str):
    spl = target.split('.')
    return '.'.join(spl[:-1])
