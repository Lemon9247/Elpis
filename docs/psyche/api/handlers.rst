===============
Handlers Module
===============

The handlers module provides server-side behavioral components.

psyche.handlers.dream_handler
-----------------------------

DreamHandler provides memory-based introspection when no clients are connected
to the Psyche server.

.. automodule:: psyche.handlers.dream_handler
   :members:
   :undoc-members:
   :show-inheritance:

Client Connection
-----------------

Client-side connection to Psyche is handled by ``RemotePsycheClient`` in the
Hermes package (``hermes.handlers``). Psyche operates as a stateless
memory-enriched inference API.
