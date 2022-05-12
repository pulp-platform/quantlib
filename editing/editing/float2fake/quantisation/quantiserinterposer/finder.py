from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern, PathGraphMatcher


class QuantiserInterposerMatcher(PathGraphMatcher):

    def __init__(self, pattern: NNSequentialPattern):
        super(QuantiserInterposerMatcher, self).__init__(pattern)
        # Despite the fact that `linear_pre` is in the "body" of the linear
        # pattern, we allow for its matched `fx.Node` to have multiple users.
        pattern.set_leakable_nodes(pattern.name_to_pattern_node()['linear_pre'])
