================
Constants Module
================

Psyche-specific constants for configuration defaults and thresholds.

.. automodule:: psyche.config.constants
   :members:
   :undoc-members:
   :show-inheritance:

Constants
---------

The following constants are defined:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Constant
     - Default
     - Description
   * - ``MEMORY_CONTENT_TRUNCATE_LENGTH``
     - 300
     - Maximum length for displayed memory content
   * - ``AUTO_STORAGE_THRESHOLD``
     - 0.6
     - Minimum importance score for automatic storage

Relationship to Mnemosyne
-------------------------

For memory-related thresholds (consolidation, similarity), Psyche imports
constants from Mnemosyne as the source of truth:

.. code-block:: python

   from mnemosyne.core.constants import (
       CONSOLIDATION_IMPORTANCE_THRESHOLD,
       CONSOLIDATION_SIMILARITY_THRESHOLD,
   )

This ensures consistent behavior between the memory server and client.
