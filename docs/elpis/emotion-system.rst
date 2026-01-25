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

Context-Aware Intensity (Event Compounding)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The regulator tracks recent events to dynamically adjust intensity based on patterns:

**Event Compounding** (negative events):
    Repeated failures increase intensity: 1.0 → 1.2 → 1.4 → ... (capped at 2.0)

**Success Dampening** (positive events):
    Repeated successes decrease intensity: 1.0 → 0.8 → 0.6 → ... (floor at 0.5)

This creates realistic emotional dynamics where:

- Repeated failures build frustration over time
- Repeated successes feel less exciting (diminishing returns)
- Events older than 10 minutes are forgotten

.. code-block:: python

    # EventHistory tracks recent events
    from elpis.emotion.regulation import EventHistory

    history = EventHistory(
        compounding_factor=0.2,  # Intensity increase per repeated failure
        dampening_factor=0.2,    # Intensity decrease per repeated success
        max_compounding=2.0,     # Maximum intensity multiplier
        min_dampening=0.5,       # Minimum intensity multiplier
    )

    # Get streak type
    streak = history.get_streak_type()  # "failure_streak", "success_streak", or None

Configuration:

.. code-block:: bash

    ELPIS_EMOTION_STREAK_COMPOUNDING_ENABLED=true
    ELPIS_EMOTION_STREAK_COMPOUNDING_FACTOR=0.2

Mood Inertia
^^^^^^^^^^^^

Mood inertia resists rapid emotional swings based on current trajectory:

**Aligned events**: Events matching current momentum get a slight boost (1.1x)
**Counter events**: Events opposing strong momentum face resistance (0.6x-0.8x)

This prevents jarring emotional whiplash and creates smoother transitions:

.. code-block:: python

    # When trending positive (positive valence_velocity):
    # - A success event gets 1.1x intensity (aligned)
    # - A failure event gets 0.6-0.8x intensity (resisted)

    # When trending negative:
    # - A failure event gets 1.1x intensity (aligned)
    # - A success event gets 0.6-0.8x intensity (resisted)

Configuration:

.. code-block:: bash

    ELPIS_EMOTION_MOOD_INERTIA_ENABLED=true
    ELPIS_EMOTION_MOOD_INERTIA_RESISTANCE=0.4  # Max resistance factor

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

Quadrant-Specific Decay Rates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different emotional states decay at different rates, reflecting psychological
reality where some emotions persist longer than others:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Quadrant
     - Multiplier
     - Effect
   * - ``excited``
     - 1.0
     - Baseline decay rate
   * - ``frustrated``
     - 0.7
     - Frustration persists longer (30% slower decay)
   * - ``calm``
     - 1.2
     - Calm states fade faster (20% faster decay)
   * - ``depleted``
     - 0.8
     - Depletion persists (20% slower decay)

Lower multiplier = emotion persists longer. Higher multiplier = faster return
to baseline.

Configuration:

.. code-block:: bash

    ELPIS_EMOTION_DECAY_MULTIPLIER_EXCITED=1.0
    ELPIS_EMOTION_DECAY_MULTIPLIER_FRUSTRATED=0.7
    ELPIS_EMOTION_DECAY_MULTIPLIER_CALM=1.2
    ELPIS_EMOTION_DECAY_MULTIPLIER_DEPLETED=0.8

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

Emotional Trajectory Tracking
-----------------------------

Beyond raw valence-arousal values, Elpis tracks emotional momentum and patterns
over time. This enables richer personality modeling and early detection of
concerning states like emotional spirals.

Trajectory Components
^^^^^^^^^^^^^^^^^^^^^

The :class:`~elpis.emotion.state.EmotionalTrajectory` dataclass captures:

**Velocity** (rate of change per minute)
    - ``valence_velocity``: Positive = improving, negative = declining
    - ``arousal_velocity``: Positive = energizing, negative = calming

**Trend Detection**
    - ``improving``: Sustained positive valence movement
    - ``declining``: Sustained negative valence movement
    - ``stable``: Little change in valence
    - ``oscillating``: Alternating direction changes

**Spiral Detection**
    Detects sustained movement away from baseline with direction awareness:

    - ``positive``: Spiraling toward high valence states
    - ``negative``: Spiraling toward low valence states
    - ``escalating``: Spiraling toward high arousal
    - ``withdrawing``: Spiraling toward low arousal

**Additional Metrics**
    - ``time_in_current_quadrant``: Seconds in current emotional quadrant
    - ``momentum``: Overall classification ("positive", "negative", "neutral")

Trajectory Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Trajectory detection thresholds can be configured:

.. code-block:: python

    from elpis.emotion.state import TrajectoryConfig

    config = TrajectoryConfig(
        history_size=20,                    # States to retain for analysis
        momentum_positive_threshold=0.01,   # Velocity threshold for positive momentum
        momentum_negative_threshold=-0.01,  # Velocity threshold for negative momentum
        trend_improving_threshold=0.02,     # Velocity threshold for improving trend
        trend_declining_threshold=-0.02,    # Velocity threshold for declining trend
        spiral_history_count=5,             # States to check for spiral
        spiral_increasing_threshold=3,      # Consecutive increases to detect spiral
    )

Or via environment variables:

.. code-block:: bash

    ELPIS_EMOTION_TRAJECTORY_HISTORY_SIZE=20
    ELPIS_EMOTION_MOMENTUM_POSITIVE_THRESHOLD=0.01
    ELPIS_EMOTION_SPIRAL_HISTORY_COUNT=5

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
        },
        "trajectory": {
            "valence_velocity": 0.05,
            "arousal_velocity": 0.02,
            "trend": "improving",
            "spiral_detected": false,
            "spiral_direction": "none",
            "time_in_quadrant": 45.2,
            "momentum": "positive"
        }
    }

**Using the emotion://state resource**:

.. code-block:: python

    # MCP resource read
    uri = "emotion://state"

Enhanced Response Analysis
--------------------------

The regulator analyzes generated content to infer emotional events using
multi-factor weighted scoring instead of simple keyword matching.

Multi-Factor Scoring
^^^^^^^^^^^^^^^^^^^^

Response analysis scores content across multiple emotion categories:

- **Success words**: "successfully", "completed", "fixed", "working", "solved"
- **Error words**: "error", "failed", "cannot", "exception", "broken"
- **Frustration words**: "still", "again", "yet", "another", "keeps"
- **Exploration words**: "interesting", "discover", "learn", "novel"
- **Uncertainty words**: "unsure", "uncertain", "might", "perhaps"

Scores use diminishing returns (sqrt normalization) to prevent gaming by
repeating keywords. Only the dominant emotion is triggered, and only if
its score exceeds the threshold.

Frustration Pattern Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The analyzer detects frustration patterns where amplifiers like "still" or
"again" appear near error words:

- "still getting the same error" -> triggers frustration event
- "yet another failure occurred" -> triggers frustration event

Configuration:

.. code-block:: bash

    ELPIS_EMOTION_RESPONSE_ANALYSIS_THRESHOLD=0.3  # Score threshold to trigger

Behavioral Monitoring
---------------------

The :class:`~elpis.emotion.behavioral_monitor.BehavioralMonitor` detects patterns
in tool usage and generation behavior to trigger appropriate emotional events.

Detected Patterns
^^^^^^^^^^^^^^^^^

**Retry Loops**
    Same tool called 3+ times in succession triggers a frustration event.
    Indicates the system is stuck repeating the same action.

**Failure Streaks**
    2+ consecutive tool failures trigger compounding error events.
    Intensity increases with streak length.

**Long Generations**
    Generation taking >30 seconds triggers a mild "blocked" event.
    Suggests the system is struggling with the task.

**Idle Periods**
    No activity for >2 minutes triggers a calming "idle" event.
    Helps the system return to baseline when not actively working.

.. code-block:: python

    from elpis.emotion.behavioral_monitor import BehavioralMonitor

    monitor = BehavioralMonitor(
        on_event=regulator.process_event,
        retry_loop_threshold=3,
        failure_streak_threshold=2,
        long_generation_seconds=30.0,
        idle_period_seconds=120.0,
    )

    # Record tool calls
    monitor.record_tool_call("search", success=False)

    # Track generation timing
    monitor.start_generation()
    # ... generation happens ...
    monitor.end_generation()

Configuration:

.. code-block:: bash

    ELPIS_EMOTION_BEHAVIORAL_MONITORING_ENABLED=true
    ELPIS_EMOTION_RETRY_LOOP_THRESHOLD=3
    ELPIS_EMOTION_FAILURE_STREAK_THRESHOLD=2
    ELPIS_EMOTION_LONG_GENERATION_SECONDS=30.0
    ELPIS_EMOTION_IDLE_PERIOD_SECONDS=120.0

Sentiment Analysis (Optional)
-----------------------------

The :class:`~elpis.emotion.sentiment.SentimentAnalyzer` provides deeper
emotional inference using either a local sentiment model or LLM self-analysis.

Local Sentiment Model
^^^^^^^^^^^^^^^^^^^^^

Uses a lightweight DistilBERT model fine-tuned for sentiment classification:

- Fast inference (~50ms per response)
- No external API calls
- Maps sentiment score to emotional events

.. code-block:: python

    from elpis.emotion.sentiment import SentimentAnalyzer

    analyzer = SentimentAnalyzer(
        use_local_model=True,
        min_length=200,  # Skip short responses
    )

    result = analyzer.analyze(response_text)
    if result:
        event = analyzer.get_emotional_event(result)
        if event:
            event_type, intensity = event
            regulator.process_event(event_type, intensity)

LLM Self-Analysis
^^^^^^^^^^^^^^^^^

Alternatively, use the inference engine itself for emotion analysis:

- Deeper understanding of nuanced content
- More expensive (additional inference call)
- Better for complex or domain-specific responses

Configuration:

.. code-block:: bash

    ELPIS_EMOTION_LLM_EMOTION_ANALYSIS_ENABLED=false  # Disabled by default
    ELPIS_EMOTION_LLM_ANALYSIS_MIN_LENGTH=200
    ELPIS_EMOTION_USE_LOCAL_SENTIMENT_MODEL=true

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
