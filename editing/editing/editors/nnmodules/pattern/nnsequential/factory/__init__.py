"""This package implements abstractions to programmatically generate
``NNSequentialPattern``s.

Rules that rewrite different patterns can share the same logic when the
patterns share some abstract structure. For example, folding the bias of an
``nn.Conv2d`` into the following ``nn.BatchNorm2d`` can be formulated in the
same way as folding the bias of an ``nn.Conv3d`` into the following
``nn.BatchNorm3d``.

In these cases, developers might want to write a single ``NNModuleApplier``,
and chain its logic to different ``NNModuleMatcher``s which have been built
around different ``NNModulePattern``s.

However, the ``NNModuleApplier`` will assume that the components of its
``NNModulePattern`` have fixed symbolic names. In this case, we need that all
the ``NNModuleMatcher``s define patterns which, although different in the
details, share the same symbolic names.

To generate these patterns programmatically, we need the following process.
* First, we establish the collection of symbolic names (the **roles**) that
  the ``NNModuleApplier`` will operate on.
* Then, for each role, we define a collection of candidates. A candidate must
  provide all the details required to assemble an ``nn.Module`` object that
  can fit the corresponding role; we group these details in an
  ``NNModuleDescription`` tuple.
* A specific mapping of candidates to roles defines a **screenplay**.

"""

from .candidates import NNModuleDescription, Candidates
from .roles import Roles
from .factory import generate_named_patterns
