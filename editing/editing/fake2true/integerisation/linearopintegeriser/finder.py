from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern, PathGraphMatcher


class LinearOpIntegeriserMatcher(PathGraphMatcher):

    def __init__(self, pattern: NNSequentialPattern):
        super(LinearOpIntegeriserMatcher, self).__init__(pattern)
        # Despite the fact that `eps_out` is in the "body" of the linear
        # pattern, we allow for its matched `fx.Node` to have multiple users
        # since its output is not meant to change after the rewriting.
        pattern.set_leakable_nodes(pattern.name_to_pattern_node()['eps_out'])
