================
Constants Module
================

Memory system constants used across the Elpis ecosystem. Mnemosyne is the
source of truth for memory-related threshold values.

.. automodule:: mnemosyne.core.constants
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
   * - ``MEMORY_SUMMARY_LENGTH``
     - 500
     - Maximum length for memory summaries
   * - ``CONSOLIDATION_IMPORTANCE_THRESHOLD``
     - 0.6
     - Minimum importance score for memory consolidation
   * - ``CONSOLIDATION_SIMILARITY_THRESHOLD``
     - 0.85
     - Similarity threshold for clustering memories

Usage
-----

Other packages (like Psyche) import these constants from Mnemosyne to ensure
consistent threshold values:

.. code-block:: python

   from mnemosyne.core.constants import (
       CONSOLIDATION_IMPORTANCE_THRESHOLD,
       CONSOLIDATION_SIMILARITY_THRESHOLD,
   )
