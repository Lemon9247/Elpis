=============
Models Module
=============

.. module:: mnemosyne.core.models
   :synopsis: Core memory data structures

The models module defines the core data structures for representing memories,
their types, lifecycle states, and emotional context.

Memory Class
------------

.. autoclass:: mnemosyne.core.models.Memory
   :members:
   :undoc-members:
   :show-inheritance:

The Memory class is the fundamental data structure in Mnemosyne. Each memory
contains:

- **Content**: The actual memory text
- **Metadata**: Type, status, importance, timestamps
- **Relationships**: Links to related memories
- **Emotional Context**: Optional emotional state at encoding

Creating Memories
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mnemosyne.core.models import Memory, MemoryType, EmotionalContext

   # Simple memory
   memory = Memory(
       content="User prefers dark mode",
       summary="UI preference",
       memory_type=MemoryType.SEMANTIC,
       tags=["preferences", "ui"],
   )

   # Memory with emotional context
   memory_with_emotion = Memory(
       content="User was excited about the new feature",
       memory_type=MemoryType.EMOTIONAL,
       emotional_context=EmotionalContext(
           valence=0.8,
           arousal=0.7,
           quadrant="excited",
       ),
   )

   # Compute importance after creation
   memory.importance_score = memory.compute_importance()

Memory Fields
^^^^^^^^^^^^^

.. list-table:: Memory Fields
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``id``
     - ``str``
     - UUID, auto-generated
   * - ``content``
     - ``str``
     - The memory text content
   * - ``summary``
     - ``str``
     - Brief summary of the memory
   * - ``memory_type``
     - ``MemoryType``
     - Classification (episodic, semantic, etc.)
   * - ``status``
     - ``MemoryStatus``
     - Lifecycle state
   * - ``created_at``
     - ``datetime``
     - Creation timestamp
   * - ``last_accessed``
     - ``datetime``
     - Last access timestamp
   * - ``access_count``
     - ``int``
     - Number of times accessed
   * - ``emotional_context``
     - ``Optional[EmotionalContext]``
     - Emotional state at encoding
   * - ``related_memory_ids``
     - ``List[str]``
     - IDs of related memories
   * - ``source_memory_ids``
     - ``List[str]``
     - IDs of source memories
   * - ``tags``
     - ``List[str]``
     - User-defined tags
   * - ``metadata``
     - ``Dict[str, Any]``
     - Custom metadata
   * - ``importance_score``
     - ``float``
     - Computed importance (0-1)

Serialization
^^^^^^^^^^^^^

.. code-block:: python

   # Convert to dictionary
   data = memory.to_dict()

   # Reconstruct from dictionary
   restored = Memory.from_dict(data)

Enumerations
------------

MemoryType
^^^^^^^^^^

.. autoclass:: mnemosyne.core.models.MemoryType
   :members:
   :undoc-members:
   :show-inheritance:

Classification of memory types, modeled after human cognition:

.. code-block:: python

   class MemoryType(Enum):
       EPISODIC = "episodic"      # Specific events/conversations
       SEMANTIC = "semantic"       # General knowledge/facts
       PROCEDURAL = "procedural"   # How to do things
       EMOTIONAL = "emotional"     # Emotional associations

**Usage:**

.. code-block:: python

   from mnemosyne.core.models import MemoryType

   # Access enum values
   memory_type = MemoryType.SEMANTIC
   type_string = memory_type.value  # "semantic"

   # Create from string
   memory_type = MemoryType("episodic")

MemoryStatus
^^^^^^^^^^^^

.. autoclass:: mnemosyne.core.models.MemoryStatus
   :members:
   :undoc-members:
   :show-inheritance:

Memory lifecycle status:

.. code-block:: python

   class MemoryStatus(Enum):
       SHORT_TERM = "short_term"
       CONSOLIDATING = "consolidating"
       LONG_TERM = "long_term"
       ARCHIVED = "archived"

**Usage:**

.. code-block:: python

   from mnemosyne.core.models import MemoryStatus

   # Check status
   if memory.status == MemoryStatus.SHORT_TERM:
       print("Memory not yet consolidated")

   # Status affects storage location
   # SHORT_TERM, CONSOLIDATING -> short_term_memory collection
   # LONG_TERM -> long_term_memory collection

EmotionalContext
----------------

.. autoclass:: mnemosyne.core.models.EmotionalContext
   :members:
   :undoc-members:
   :show-inheritance:

Captures the emotional state at the time of memory encoding using the
valence-arousal model.

Fields
^^^^^^

.. list-table:: EmotionalContext Fields
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``valence``
     - ``float``
     - Negative to positive (-1 to 1)
   * - ``arousal``
     - ``float``
     - Low to high activation (-1 to 1)
   * - ``quadrant``
     - ``str``
     - Emotional quadrant name

Quadrants
^^^^^^^^^

The quadrant field should be one of:

- ``"excited"``: Positive valence, high arousal
- ``"frustrated"``: Negative valence, high arousal
- ``"calm"``: Positive valence, low arousal
- ``"depleted"``: Negative valence, low arousal

Salience Property
^^^^^^^^^^^^^^^^^

The ``salience`` property computes emotional significance:

.. code-block:: python

   @property
   def salience(self) -> float:
       """Returns value between 0.5 and 1.0."""
       arousal_factor = (abs(self.arousal) + 1) / 2   # 0.5 to 1.0
       valence_factor = (abs(self.valence) + 1) / 2   # 0.5 to 1.0

       # Arousal matters more for salience
       return (arousal_factor * 0.7) + (valence_factor * 0.3)

**Examples:**

.. code-block:: python

   # High arousal = high salience
   excited = EmotionalContext(valence=0.8, arousal=0.9, quadrant="excited")
   print(excited.salience)  # ~0.92

   # Low arousal = lower salience
   calm = EmotionalContext(valence=0.5, arousal=-0.5, quadrant="calm")
   print(calm.salience)  # ~0.60

   # Negative but high arousal = still high salience
   frustrated = EmotionalContext(valence=-0.7, arousal=0.8, quadrant="frustrated")
   print(frustrated.salience)  # ~0.88

Serialization
^^^^^^^^^^^^^

.. code-block:: python

   # Convert to dictionary
   ctx = EmotionalContext(valence=0.5, arousal=0.3, quadrant="calm")
   data = ctx.to_dict()
   # {'valence': 0.5, 'arousal': 0.3, 'quadrant': 'calm', 'salience': 0.61}

   # Reconstruct from dictionary
   restored = EmotionalContext.from_dict(data)

See Also
--------

- :doc:`../memory-types` - Conceptual guide to memory types
- :doc:`storage` - How memories are stored
- :doc:`server` - MCP tools that use these models
