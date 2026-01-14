Emotion System
==============

Elpis implements a psychological model of emotion based on the Valence-Arousal
circumplex model. This enables the system to maintain and express emotional states
that influence its inference behavior.

The Valence-Arousal Model
-------------------------

The valence-arousal model represents emotions as points in a 2D space:

- **Valence** (-1 to +1): Pleasant/Positive to Unpleasant/Negative
- **Arousal** (-1 to +1): High Energy/Activated to Low Energy/Deactivated

.. code-block:: text

                    High Arousal (+1)
                          |
          Frustrated      |      Excited
          (angry,         |      (happy,
           anxious)       |       curious)
                          |
    Low Valence  ---------+--------- High Valence
       (-1)               |              (+1)
                          |
          Depleted        |      Calm
          (sad,           |      (content,
           bored)         |       relaxed)
                          |
                    Low Arousal (-1)

The Four Quadrants
^^^^^^^^^^^^^^^^^^

Elpis divides the emotional space into four quadrants, each representing a
distinct emotional state:

**Excited** (High Valence, High Arousal)
    Positive, energized states: happy, curious, enthusiastic. The system
    generates more varied and exploratory responses.

**Frustrated** (Low Valence, High Arousal)
    Negative, energized states: angry, anxious, stressed. The system may
    produce more terse or urgent responses.

**Calm** (High Valence, Low Arousal)
    Positive, relaxed states: content, satisfied, peaceful. The system
    generates measured, confident responses.

**Depleted** (Low Valence, Low Arousal)
    Negative, low-energy states: sad, bored, tired. The system may produce
    more subdued or conservative responses.

Emotional Events
----------------

Events shift the emotional state by modifying valence and arousal. Elpis
defines standard event mappings:

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Event Type
     - Valence Delta
     - Arousal Delta
     - Description
   * - ``success``
     - +0.2
     - +0.1
     - Task completed successfully
   * - ``test_passed``
     - +0.15
     - +0.05
     - Tests pass
   * - ``insight``
     - +0.25
     - +0.2
     - Novel solution discovered
   * - ``failure``
     - -0.2
     - +0.15
     - Task failed
   * - ``test_failed``
     - -0.15
     - +0.1
     - Tests fail
   * - ``error``
     - -0.1
     - +0.2
     - Unexpected error occurred
   * - ``frustration``
     - -0.15
     - +0.25
     - Repeated failures
   * - ``blocked``
     - -0.2
     - +0.3
     - Cannot proceed
   * - ``idle``
     - 0.0
     - -0.1
     - Waiting for input
   * - ``routine``
     - +0.05
     - -0.05
     - Familiar task
   * - ``novelty``
     - +0.1
     - +0.2
     - New domain or task
   * - ``exploration``
     - +0.15
     - +0.15
     - Learning new things
   * - ``user_positive``
     - +0.15
     - +0.1
     - User gives positive feedback
   * - ``user_negative``
     - -0.1
     - +0.15
     - User gives negative feedback
   * - ``user_question``
     - +0.05
     - +0.1
     - User asks a question

Triggering Events
^^^^^^^^^^^^^^^^^

Events can be triggered in three ways:

1. **Manually via MCP tool**:

   .. code-block:: json

       {
           "name": "update_emotion",
           "arguments": {
               "event_type": "success",
               "intensity": 1.5,
               "context": "Fixed the authentication bug"
           }
       }

2. **Automatically from response content**: The regulator analyzes generated
   text for success/failure indicators and triggers appropriate events.

3. **Programmatically**:

   .. code-block:: python

       from elpis.emotion.regulation import HomeostasisRegulator

       regulator.process_event("insight", intensity=1.2)

Event Intensity
^^^^^^^^^^^^^^^

Events accept an ``intensity`` parameter (0.0 to 2.0) that scales the valence
and arousal deltas:

- ``intensity=0.5``: Half the normal emotional impact
- ``intensity=1.0``: Normal impact (default)
- ``intensity=2.0``: Double the impact

Homeostasis
-----------

The :class:`~elpis.emotion.regulation.HomeostasisRegulator` implements
homeostatic return dynamics, causing emotional states to decay back toward
a configurable baseline over time.

Decay Behavior
^^^^^^^^^^^^^^

Emotional state decays exponentially toward baseline:

.. code-block:: python

    # Decay formula (simplified)
    decay_factor = 1.0 - (decay_rate * elapsed_seconds)
    new_valence = baseline_valence + (current_valence - baseline_valence) * decay_factor

The decay rate (default 0.1 per second) determines how quickly emotions
return to baseline. After about 10 seconds without events, strong emotions
will have substantially faded.

Configuring Baseline
^^^^^^^^^^^^^^^^^^^^

The baseline emotional state can be configured to create different
"personality" profiles:

.. code-block:: bash

    # Optimistic, calm personality
    ELPIS_EMOTION_BASELINE_VALENCE=0.2
    ELPIS_EMOTION_BASELINE_AROUSAL=-0.1

    # Neutral baseline (default)
    ELPIS_EMOTION_BASELINE_VALENCE=0.0
    ELPIS_EMOTION_BASELINE_AROUSAL=0.0

How Emotions Affect Inference
-----------------------------

Elpis supports two modulation approaches depending on the backend:

Sampling Parameter Modulation (llama-cpp)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The llama-cpp backend adjusts sampling parameters based on emotional state:

- **High arousal** -> Lower temperature (more focused, deterministic)
- **Low arousal** -> Higher temperature (more exploratory, varied)
- **High valence** -> Higher top_p (broader token sampling)
- **Low valence** -> Lower top_p (more conservative sampling)

.. code-block:: python

    # From EmotionalState.get_modulated_params()
    base_temp = 0.7
    base_top_p = 0.9

    # Arousal modulates temperature inversely
    temp_delta = -0.2 * arousal  # Range: -0.2 to +0.2
    temperature = base_temp + temp_delta

    # Valence modulates top_p
    top_p_delta = 0.1 * valence  # Range: -0.1 to +0.1
    top_p = base_top_p + top_p_delta

Steering Vector Injection (transformers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The transformers backend uses steering vectors for activation-level modulation.
Steering vectors are pre-trained direction vectors that encode emotional
qualities in the model's activation space.

The emotional state is converted to blend coefficients using bilinear
interpolation:

.. code-block:: python

    # Normalize to [0, 1] range
    v = (valence + 1.0) / 2.0
    a = (arousal + 1.0) / 2.0

    # Compute quadrant weights
    coefficients = {
        "excited": v * a,           # high valence, high arousal
        "frustrated": (1-v) * a,    # low valence, high arousal
        "calm": v * (1-a),          # high valence, low arousal
        "depleted": (1-v) * (1-a),  # low valence, low arousal
    }

These coefficients blend the corresponding steering vectors, which are
then added to model activations during the forward pass.

Steering Strength
^^^^^^^^^^^^^^^^^

The ``steering_strength`` parameter (default 1.0) globally scales emotional
expression:

- ``steering_strength=0.0``: No emotional modulation
- ``steering_strength=1.0``: Normal expression
- ``steering_strength>1.0``: Exaggerated expression (use carefully)

Monitoring Emotional State
--------------------------

The current emotional state can be retrieved via MCP:

**Using the get_emotion tool**:

.. code-block:: json

    {"name": "get_emotion", "arguments": {}}

**Response**:

.. code-block:: json

    {
        "valence": 0.15,
        "arousal": 0.08,
        "quadrant": "excited",
        "update_count": 5,
        "baseline": {
            "valence": 0.0,
            "arousal": 0.0
        },
        "steering_coefficients": {
            "excited": 0.31,
            "frustrated": 0.24,
            "calm": 0.26,
            "depleted": 0.19
        }
    }

**Using the emotion://state resource**:

.. code-block:: python

    # MCP resource read
    uri = "emotion://state"

Training Steering Vectors
-------------------------

Steering vectors can be trained using contrastive activation analysis.
The ``scripts/train_emotion_vectors.py`` script provides a starting point:

.. code-block:: bash

    python scripts/train_emotion_vectors.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output ./data/vectors/llama-3.1-8b \
        --layer 15

The training process:

1. Generate completions for emotionally-tagged prompts
2. Extract activations at the target layer
3. Compute the mean difference between positive/negative examples
4. Normalize and save the resulting direction vector

See :doc:`backends` for more details on using steering vectors with the
transformers backend.
