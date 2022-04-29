from collections import OrderedDict
import torch.fx as fx

from quantlib.editing.editing.editors.base import ApplicationPoint


class NodesMap(ApplicationPoint, OrderedDict):
    """The ``ApplicationPoint`` for pattern-matching-based rewriting rules."""

    def __setitem__(self, pn: fx.Node, dn: fx.Node):
        """A ``NodesMap`` can only map ``fx.Node``s to ``fx.Node``s."""
        if not (isinstance(pn, fx.Node) and isinstance(dn, fx.Node)):
            raise TypeError
        super(ApplicationPoint, self).__setitem__(pn, dn)
