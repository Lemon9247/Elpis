============
Memory Types
============

Mnemosyne implements a memory system inspired by human cognitive architecture. This
guide explains the different memory types, lifecycle states, importance scoring,
and emotional context.

Memory Type Enum
----------------

The :class:`~mnemosyne.core.models.MemoryType` enum defines four categories of memories:

.. list-table:: Memory Types
   :header-rows: 1
   :widths: 20 20 60

   * - Type
     - Value
     - Description
   * - ``EPISODIC``
     - ``"episodic"``
     - Specific events and conversations. "I talked to Alice yesterday about Python."
   * - ``SEMANTIC``
     - ``"semantic"``
     - General knowledge and facts. "Python is a programming language."
   * - ``PROCEDURAL``
     - ``"procedural"``
     - How-to knowledge. "To commit code, use git commit -m 'message'."
   * - ``EMOTIONAL``
     - ``"emotional"``
     - Emotional associations. "User felt frustrated when the build failed."

Choosing the Right Type
^^^^^^^^^^^^^^^^^^^^^^^

**Episodic** memories are best for:

- Conversation history
- Specific events with context
- Time-bound information

**Semantic** memories are best for:

- User preferences
- Facts and definitions
- Knowledge that doesn't change often

**Procedural** memories are best for:

- Workflows and processes
- Commands and syntax
- Step-by-step instructions

**Emotional** memories are best for:

- Sentiment patterns
- User emotional states
- Reactions to events

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from mnemosyne.core.models import Memory, MemoryType

   # Create an episodic memory
   conversation_memory = Memory(
       content="User asked about project architecture on Monday",
       memory_type=MemoryType.EPISODIC,
       tags=["conversation", "architecture"],
   )

   # Create a semantic memory
   preference_memory = Memory(
       content="User prefers dark mode and large fonts",
       memory_type=MemoryType.SEMANTIC,
       tags=["preferences", "ui"],
   )

   # Create a procedural memory
   workflow_memory = Memory(
       content="To deploy: run tests, build Docker image, push to registry",
       memory_type=MemoryType.PROCEDURAL,
       tags=["deployment", "workflow"],
   )

Memory Lifecycle
----------------

The :class:`~mnemosyne.core.models.MemoryStatus` enum tracks memory state through
its lifecycle:

.. code-block:: text

   SHORT_TERM --> CONSOLIDATING --> LONG_TERM --> ARCHIVED
        |                              ^
        +------------------------------+
              (high importance)

Status Definitions
^^^^^^^^^^^^^^^^^^

.. list-table:: Memory Status
   :header-rows: 1
   :widths: 20 20 60

   * - Status
     - Value
     - Description
   * - ``SHORT_TERM``
     - ``"short_term"``
     - Recent memory, not yet consolidated. May be forgotten.
   * - ``CONSOLIDATING``
     - ``"consolidating"``
     - Memory is being processed for long-term storage.
   * - ``LONG_TERM``
     - ``"long_term"``
     - Consolidated memory, persists indefinitely.
   * - ``ARCHIVED``
     - ``"archived"``
     - Memory is archived, lower priority in search.

Storage Location
^^^^^^^^^^^^^^^^

Memory status determines storage location:

- ``SHORT_TERM``, ``CONSOLIDATING``: Stored in ``short_term_memory`` collection
- ``LONG_TERM``: Stored in ``long_term_memory`` collection
- ``ARCHIVED``: Currently stored with long-term (future: separate archive)

Importance Scoring
------------------

Mnemosyne computes an importance score for each memory using a weighted algorithm.
Higher importance memories are more likely to be consolidated to long-term storage
and rank higher in search results.

Algorithm
^^^^^^^^^

The importance score is calculated using three factors:

.. math::

   importance = (salience \times 0.4) + (recency \times 0.3) + (access\_factor \times 0.3)

**Emotional Salience (40%)**
    How emotionally significant the memory is. Computed from the
    :class:`~mnemosyne.core.models.EmotionalContext` if present.

**Recency (30%)**
    How recently the memory was created. Decays linearly over one year:

    .. code-block:: python

       age_days = (now - created_at).days
       recency = max(0.0, 1.0 - (age_days / 365))

**Access Frequency (30%)**
    How often the memory has been accessed. Caps at 10 accesses:

    .. code-block:: python

       access_factor = min(1.0, access_count / 10)

Implementation
^^^^^^^^^^^^^^

.. code-block:: python

   def compute_importance(self) -> float:
       """Compute importance score (0 to 1)."""
       # Emotional salience
       salience = self.emotional_context.salience if self.emotional_context else 0.5

       # Recency (decays over time)
       age_days = (datetime.now() - self.created_at).days
       recency = max(0.0, 1.0 - (age_days / 365))

       # Access frequency
       access_factor = min(1.0, self.access_count / 10)

       # Weighted combination
       return (salience * 0.4) + (recency * 0.3) + (access_factor * 0.3)

Example Scores
^^^^^^^^^^^^^^

.. list-table:: Example Importance Scores
   :header-rows: 1
   :widths: 40 20 20 20

   * - Memory Characteristics
     - Salience
     - Recency
     - Score
   * - New, high arousal, never accessed
     - 0.9
     - 1.0
     - 0.66
   * - 6 months old, neutral, accessed 5 times
     - 0.5
     - 0.5
     - 0.50
   * - 1 year old, low arousal, rarely accessed
     - 0.5
     - 0.0
     - 0.23

Emotional Context
-----------------

The :class:`~mnemosyne.core.models.EmotionalContext` dataclass captures the emotional
state at the time of memory encoding. This enables emotion-aware memory retrieval
and importance scoring.

Valence-Arousal Model
^^^^^^^^^^^^^^^^^^^^^

Mnemosyne uses the circumplex model of affect:

- **Valence** (-1 to 1): Negative to positive emotional state
- **Arousal** (-1 to 1): Low to high activation level

.. code-block:: text

                    High Arousal
                         |
                         |
      Frustrated    -----+-----    Excited
      (-, +)             |          (+, +)
                         |
   Low Valence -------- 0,0 -------- High Valence
                         |
                         |
      Depleted      -----+-----    Calm
      (-, -)             |          (+, -)
                         |
                    Low Arousal

Quadrants
^^^^^^^^^

The ``quadrant`` field indicates which emotional quadrant applies:

- **excited**: Positive valence, high arousal (happy, enthusiastic)
- **frustrated**: Negative valence, high arousal (angry, anxious)
- **calm**: Positive valence, low arousal (relaxed, content)
- **depleted**: Negative valence, low arousal (sad, bored)

Emotional Salience
^^^^^^^^^^^^^^^^^^

The salience property computes how emotionally significant a memory is:

.. code-block:: python

   @property
   def salience(self) -> float:
       """Compute emotional salience (0.5 to 1.0)."""
       arousal_factor = (abs(self.arousal) + 1) / 2   # 0.5 to 1.0
       valence_factor = (abs(self.valence) + 1) / 2   # 0.5 to 1.0

       # Arousal matters more for salience
       return (arousal_factor * 0.7) + (valence_factor * 0.3)

**Key insight**: Arousal contributes more to salience than valence. Highly arousing
experiences (whether positive or negative) are remembered better than neutral ones.

Example: Creating Emotional Memories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mnemosyne.core.models import Memory, MemoryType, EmotionalContext

   # Create a memory with emotional context
   memory = Memory(
       content="User was excited when the feature shipped successfully",
       memory_type=MemoryType.EMOTIONAL,
       emotional_context=EmotionalContext(
           valence=0.8,    # Positive
           arousal=0.7,    # High activation
           quadrant="excited",
       ),
   )

   # Compute importance (will be high due to emotional salience)
   memory.importance_score = memory.compute_importance()
   print(f"Importance: {memory.importance_score:.2f}")  # ~0.65+

Storing via MCP
^^^^^^^^^^^^^^^

.. code-block:: python

   # MCP tool call with emotional context
   {
       "content": "Build failed and user expressed frustration",
       "memory_type": "emotional",
       "emotional_context": {
           "valence": -0.6,
           "arousal": 0.7,
           "quadrant": "frustrated"
       },
       "tags": ["build", "error", "emotion"]
   }

See Also
--------

- :doc:`architecture` - How memories flow through the system
- :doc:`api/models` - Full API reference for Memory and related classes
- :doc:`/elpis/emotional-states` - Emotional state model in Elpis
