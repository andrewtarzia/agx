Graph enumeration
=================


The iterators
-------------

- :doc:`TopologyIterator <_autosummary/agx.TopologyIterator>`
- :doc:`SkewedTopologyIterator <_autosummary/agx.SkewedTopologyIterator>`

You can use :meth:`get_configurations` to iterate over node configurations
for a given iterator, which can then be mapped to a graph using the containers
below.

Containers
----------

- :doc:`TopologyCode <_autosummary/agx.TopologyCode>`
- :doc:`Configuration <_autosummary/agx.Configuration>`
- :doc:`ConfiguredCode <_autosummary/agx.ConfiguredCode>`

Isomorphism
-----------

For both :class:`agx.TopologyCode` and :class:`agx.ConfiguredCode`, you can
use the ``get_graph`` method and/or the ``is_isomorphic`` method to check if
two graphs/configured graphs are the same.

For graphs that do not contain parallel edges, you can also use the ``pynauty``
certificate.
