"""This package implements the hierachy of base classes of QuantLib's graph
editing machinery.

Edits are partitioned into basic edits and composed edits. Basic edits are
further partitioned into annotations (which add attributes and other semantic
information but do not modify functional graph information) and rewritings
(which can modify functional graph information, and even change the topology
of the graph).

"""

#                      Editor                                                     #
#                        /\                                                       #
#                       /  \                                                      #
#          BaseEditor _/    \_ ComposedEditor                                     #
#              /\                                                                 #
#             /  \                                                                #
#  Annotator_/    \_ Rewriter                                                     #
#                    |_ ApplicationPoint + Context = ApplicationPointWithContext  #
#                    |_ Finder                                                    #
#                    |_ Applier                                                   #

from .annotator import Annotator
from .rewriter import ApplicationPoint, Finder, Applier, Rewriter
from .composededitor import ComposedEditor
