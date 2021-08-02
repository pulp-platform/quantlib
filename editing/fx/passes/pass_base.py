from typing import Optional

from torch import fx, nn
from torch.fx.subgraph_rewriter import Match

from ..util import get_ordered_active_nodes, get_qualified_prefix, module_of_node, SequentialMatcher

__all__ = ['FxPass',
           'SequentialPass',
           'ModifyMatchedModulesPass',
           'ModifySequentialPatternPass',
           'ReplaceMatchWithModulePass',
           'ReplaceSequentialPatternPass']

#TODO implement logging!

class FxPass:

    def __init__(self):
        self.parent = None
        self._subpasses = {}


    def __setattr__(self, attribute, value):
        if isinstance(value, FxPass) and attribute != 'parent':
            self.register_subpass(attribute, value)
        super(FxPass, self).__setattr__(attribute, value)

    def register_subpass(self, name, value):
        subpasses = self.__dict__.get('_subpasses')
        if subpasses is None:
            raise AttributeError("Cannot assign sub-pass before calling FxPass.__init__!")
        if name in self._subpasses.keys():
            del self._subpasses[name]

        value.parent = self
        self._subpasses[name] = value

    def remove_subpass(self, name):
        try:
            del self._subpasses[name]
        except KeyError:
            print(f"No subpass with name {name}, cannot remove!")
        except AttributeError:
            raise AttributeError("Cannot remove sub-pass before calling FxPass.__init__!")


    def __getattr__(self, attribute):
        subpasses = self.__dict__.get('_subpasses')
        if subpasses is not None and attribute in subpasses.keys():
            return subpasses[attribute]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attribute}")

    def named_subpasses(self):
        return self._subpasses.copy()

    # overwrite this function in custom pass subclasses!
    # run_pass should return the modified graphModule.
    def run_pass(self, gm : fx.GraphModule):
        raise NotImplementedError("Can't apply FxPass base class. Inherit from this class")

    # DO NOT OVERWRITE this function in custom pass subclasses unless you have
    # a very good reason!
    def apply(self, gm : fx.GraphModule):
        self.retarget(gm)
        gm = self.run_pass(gm)
        gm.recompile()
        gm.graph.lint()
        return gm

    # overwrite this if your pass is specific to a graph instance (most
    # "dynamic" SequentialPass derivatives will be, as the list of passes to
    # execute probably depends on the graph. See e.g.
    # ReplaceSequentialPatternPass for an example)
    def retarget(self, gm : fx.GraphModule):
        pass

    def __call__(self, gm : fx.GraphModule):
        self.apply(gm)

class SequentialPass(FxPass):
    def __init__(self, *passes, name_prefix = ''):
        super(SequentialPass, self).__init__()
        self.name_prefix = name_prefix
        self.setup_passes(passes)

    def run_pass(self, gm : fx.GraphModule):
        for p in self.named_subpasses().values():
            gm = p.apply(gm)
        return gm

    def setup_passes(self, passes):
        for i, p in enumerate(passes):
            self.register_subpass(self.name_prefix+'_'+str(i), p)



class ModifyMatchedModulesPass(FxPass):
    # applies mod_fun to the list of matched modules in a Match object
    def __init__(self, match : Match, mod_fun : callable, **kwargs):
        super(ModifyMatchedModulesPass, self).__init__()
        active_nodes = get_ordered_active_nodes(match)
        self.nodes = active_nodes
        self.mod_fun = mod_fun
        self.kwargs = kwargs

    def run_pass(self, gm : fx.GraphModule):
        modules = [module_of_node(gm, n) for n in self.nodes if n.op == 'call_module']
        self.mod_fun(modules, **self.kwargs)
        return gm

class ModifySequentialPatternPass(SequentialPass):
    def __init__(self, pattern : callable, trace : callable, mod_fun : callable, name : str = '', **kwargs):
        super(ModifySequentialPatternPass, self).__init__(name_prefix=name)
        self.matcher = SequentialMatcher(pattern, trace)
        self.mod_fun = mod_fun
        self.name = name
        self.kwargs = kwargs

    def retarget(self, gm : fx.GraphModule):
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)
        self.matches = self.matcher.match_graph(gm)
        passes = []
        for i, m in enumerate(self.matches):
            passes.append(ModifyMatchedModulesPass(m, self.mod_fun, **self.kwargs))


class ReplaceMatchWithModulePass(FxPass):
    #Matches are specific to graph instances, so don't use this type of pass on its
    #own if you want to reuse it!
    def __init__(self, match : Match, module : nn.Module, name : str, insert_target : Optional[str] = None):
        # this class needs a name field because the inserted submodules will be named
        super(ReplaceMatchWithModulePass, self).__init__()
        self.match = match
        self.module = module
        self.name = name
        self.insert_target = insert_target

    def run_pass(self, gm : fx.GraphModule):
        matched_nodes = get_ordered_active_nodes(self.match)
        first_matched_node = matched_nodes[0]
        out_node = matched_nodes[-1]
        # we use the first matched node as the hierarchy level to insert the
        # new submodule. if the first matched node is not a module, try to use
        # the insert_target field
        target_list = []
        if self.insert_target is None and first_matched_node.op == 'call_module':
            first_pattern_node_target = get_qualified_prefix(first_matched_node.target)
            if len(first_pattern_node_target):
                target_list.append(get_qualified_prefix(first_matched_node.target))
        elif self.insert_target is not None:
            target_list.append(self.insert_target)

        target_list.append(f"_QL_REPLACED_{self.name.upper()}")
        target = '.'.join(target_list)
        #add the submodule
        gm.add_submodule(target, self.module)

        with gm.graph.inserting_after(first_matched_node.all_input_nodes[0]):
            new_node = gm.graph.call_module(target, args=first_matched_node.args, kwargs=first_matched_node.kwargs)

        #replace all uses of the output node with the new module's output
        out_users = [u for u in out_node.users]
        for u in out_users:
            u.replace_input_with(out_node, new_node)

        for n in reversed(matched_nodes):
            gm.graph.erase_node(n)
            if n.op == 'call_module':
                gm.delete_submodule(n.target)

        return gm

class ReplaceSequentialPatternPass(SequentialPass):
    def __init__(self, pattern : callable, trace : callable, replacement_fn : callable, name : '', **kwargs):
        super(ReplaceSequentialPatternPass, self).__init__(name_prefix=name)
        self.matcher = SequentialMatcher(pattern, trace)
        self.replacement_fn = replacement_fn
        self.name = name
        self.kwargs = kwargs

    def retarget(self, gm : fx.GraphModule):
        # to retarget to a new graph, clear all registered subpasses.
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)
        self.matches = self.matcher.match_graph(gm)
        passes = []
        for i, m in enumerate(self.matches):
            replacement_module = self.replacement_fn(gm, m, **self.kwargs)
            passes.append(ReplaceMatchWithModulePass(m, replacement_module, f"{self.name}_{i}"))

        self.setup_passes(passes)
