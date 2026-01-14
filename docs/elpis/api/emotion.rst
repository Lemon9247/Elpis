Emotion Module
==============

The emotion module implements the valence-arousal emotional model and
homeostatic regulation for Elpis.

.. module:: elpis.emotion
   :synopsis: Emotional state and regulation

Emotional State
---------------

.. automodule:: elpis.emotion.state
   :synopsis: Valence-arousal emotional state model

.. autoclass:: elpis.emotion.state.EmotionalState
   :members:
   :undoc-members:
   :show-inheritance:

The ``EmotionalState`` class represents a point in 2D emotional space:

.. code-block:: python

    from elpis.emotion.state import EmotionalState

    # Create a state
    state = EmotionalState(
        valence=0.3,   # Slightly positive
        arousal=-0.1,  # Slightly calm
    )

    # Check the quadrant
    print(state.get_quadrant())  # "calm"

    # Get inference parameters
    params = state.get_modulated_params()
    print(params)  # {"temperature": 0.72, "top_p": 0.93}

    # Get steering coefficients
    coeffs = state.get_steering_coefficients()
    print(coeffs)
    # {"excited": 0.29, "frustrated": 0.21, "calm": 0.32, "depleted": 0.18}

State Attributes
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Attribute
     - Type
     - Description
   * - ``valence``
     - float
     - Pleasant (+1) to Unpleasant (-1)
   * - ``arousal``
     - float
     - High energy (+1) to Low energy (-1)
   * - ``last_update``
     - float
     - Timestamp of last state change
   * - ``update_count``
     - int
     - Number of state updates
   * - ``baseline_valence``
     - float
     - Homeostatic baseline valence
   * - ``baseline_arousal``
     - float
     - Homeostatic baseline arousal
   * - ``steering_strength``
     - float
     - Global steering multiplier

State Methods
^^^^^^^^^^^^^

``get_quadrant() -> str``
    Returns the current emotional quadrant: "excited", "frustrated",
    "calm", or "depleted".

``get_modulated_params() -> Dict[str, float]``
    Returns sampling parameters adjusted for emotional state:
    ``{"temperature": float, "top_p": float}``.

``get_steering_coefficients() -> Dict[str, float]``
    Returns blend weights for steering vectors using bilinear
    interpolation across quadrants.

``shift(valence_delta, arousal_delta)``
    Shift state by the given deltas, clamping to [-1, 1].

``reset()``
    Reset to baseline state.

``distance_from_baseline() -> float``
    Euclidean distance from baseline (0 = at baseline).

``get_dominant_emotion() -> Tuple[str, float]``
    Get the strongest emotional component (name, coefficient).

``to_dict() -> Dict[str, Any]``
    Serialize state to dictionary.

Homeostasis Regulator
---------------------

.. automodule:: elpis.emotion.regulation
   :synopsis: Homeostatic regulation for emotional state

.. autoclass:: elpis.emotion.regulation.HomeostasisRegulator
   :members:
   :undoc-members:
   :show-inheritance:

The ``HomeostasisRegulator`` processes emotional events and manages
decay toward baseline:

.. code-block:: python

    from elpis.emotion.state import EmotionalState
    from elpis.emotion.regulation import HomeostasisRegulator

    state = EmotionalState()
    regulator = HomeostasisRegulator(
        state=state,
        decay_rate=0.1,  # Return to baseline at 0.1/second
        max_delta=0.5,   # Max change per event
    )

    # Process an event
    regulator.process_event("success", intensity=1.5)

    # Automatically analyzes response content
    regulator.process_response("Successfully fixed the bug!")

    # Get available event types
    events = regulator.get_available_events()
    print(events["insight"])  # (0.25, 0.2) -> valence, arousal delta

Event Mappings
^^^^^^^^^^^^^^

.. autodata:: elpis.emotion.regulation.EVENT_MAPPINGS
   :no-value:

The module defines standard event mappings as a dictionary:

.. code-block:: python

    EVENT_MAPPINGS = {
        # Success events
        "success": (0.2, 0.1),
        "test_passed": (0.15, 0.05),
        "insight": (0.25, 0.2),

        # Failure events
        "failure": (-0.2, 0.15),
        "test_failed": (-0.15, 0.1),
        "error": (-0.1, 0.2),

        # Frustration events
        "frustration": (-0.15, 0.25),
        "blocked": (-0.2, 0.3),

        # Calm events
        "idle": (0.0, -0.1),
        "routine": (0.05, -0.05),

        # Novelty events
        "novelty": (0.1, 0.2),
        "exploration": (0.15, 0.15),

        # User interaction
        "user_positive": (0.15, 0.1),
        "user_negative": (-0.1, 0.15),
        "user_question": (0.05, 0.1),
    }

Regulator Methods
^^^^^^^^^^^^^^^^^

``process_event(event_type, intensity=1.0, context=None)``
    Process an emotional event and update state. The ``intensity``
    parameter (0.0-2.0) scales the event's impact.

``process_response(content)``
    Analyze generated text content for emotional indicators and
    trigger appropriate events automatically.

``get_available_events() -> Dict[str, Tuple[float, float]]``
    Return the event mappings dictionary.

Decay Behavior
^^^^^^^^^^^^^^

Before processing any event, the regulator applies time-based decay:

.. code-block:: python

    # Exponential decay toward baseline
    elapsed = time.time() - state.last_update
    decay_factor = max(0.0, 1.0 - (decay_rate * elapsed))

    new_valence = baseline + (current - baseline) * decay_factor

This ensures emotions naturally fade over time, preventing the system
from getting "stuck" in extreme emotional states.

Usage Example
-------------

Complete example of emotional state management:

.. code-block:: python

    from elpis.emotion.state import EmotionalState
    from elpis.emotion.regulation import HomeostasisRegulator

    # Initialize with custom baseline
    state = EmotionalState(
        baseline_valence=0.1,   # Slightly optimistic
        baseline_arousal=0.0,  # Neutral energy
        steering_strength=1.2, # Slightly stronger expression
    )

    regulator = HomeostasisRegulator(
        state=state,
        decay_rate=0.05,  # Slow decay
        max_delta=0.3,    # Limited event impact
    )

    # Simulate some events
    regulator.process_event("novelty", intensity=1.0)
    print(f"After novelty: {state.get_quadrant()}")  # "excited"

    regulator.process_event("error", intensity=0.5)
    print(f"After error: {state.get_quadrant()}")  # Still "excited" probably

    # Get current state for inference
    if state.arousal > 0.5:
        # Very activated - maybe more focused generation
        params = state.get_modulated_params()
    else:
        # Calmer - maybe more exploratory
        coeffs = state.get_steering_coefficients()
