LLM Module
==========

The LLM module provides the inference engine abstraction and backend
implementations for Elpis.

.. module:: elpis.llm
   :synopsis: LLM inference engines and backends

Base Interface
--------------

.. automodule:: elpis.llm.base
   :members:
   :undoc-members:
   :show-inheritance:
   :synopsis: Abstract base class for inference engines

The ``InferenceEngine`` abstract base class defines the interface that all
inference backends must implement:

.. code-block:: python

    from elpis.llm.base import InferenceEngine

    class InferenceEngine(ABC):
        # Capability flags
        SUPPORTS_STEERING: bool = False
        MODULATION_TYPE: Literal["none", "sampling", "steering"] = "none"

        @abstractmethod
        async def chat_completion(
            self,
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            emotion_coefficients: Optional[Dict[str, float]] = None,
        ) -> str: ...

        @abstractmethod
        async def chat_completion_stream(
            self,
            messages: List[Dict[str, str]],
            ...
        ) -> AsyncIterator[str]: ...

        @abstractmethod
        async def function_call(
            self,
            messages: List[Dict[str, str]],
            tools: List[Dict[str, Any]],
            ...
        ) -> Optional[List[Dict[str, Any]]]: ...

Capability Flags
^^^^^^^^^^^^^^^^

``SUPPORTS_STEERING``
    Whether the backend supports steering vector injection. If ``True``,
    the ``emotion_coefficients`` parameter will be used for activation
    modulation. If ``False``, it will be logged but ignored.

``MODULATION_TYPE``
    How the backend achieves emotional modulation:

    - ``"none"``: No modulation support
    - ``"sampling"``: Modulates temperature/top_p
    - ``"steering"``: Injects steering vectors

Backend Registry
----------------

.. automodule:: elpis.llm.backends
   :noindex:
   :synopsis: Backend registry and factory

.. autofunction:: elpis.llm.backends.create_backend

.. autofunction:: elpis.llm.backends.register_backend

.. autofunction:: elpis.llm.backends.get_available_backends

.. autofunction:: elpis.llm.backends.is_backend_available

The backend registry provides a plugin-style architecture:

.. code-block:: python

    from elpis.llm.backends import create_backend, get_available_backends
    from elpis.config.settings import ModelSettings

    # Check available backends
    available = get_available_backends()
    print(available)  # {"llama-cpp": True, "transformers": True}

    # Create a backend
    settings = ModelSettings(backend="llama-cpp", path="./model.gguf")
    engine = create_backend(settings)

    # Use the engine
    response = await engine.chat_completion([
        {"role": "user", "content": "Hello!"}
    ])

llama-cpp Backend
-----------------

.. automodule:: elpis.llm.backends.llama_cpp
   :noindex:
   :synopsis: llama-cpp-python inference backend

.. autoclass:: elpis.llm.backends.llama_cpp.inference.LlamaInference
   :members:
   :undoc-members:
   :show-inheritance:

The llama-cpp backend provides fast inference for GGUF quantized models:

.. code-block:: python

    from elpis.llm.backends.llama_cpp import LlamaInference
    from elpis.llm.backends.llama_cpp.config import LlamaCppConfig

    config = LlamaCppConfig(
        path="./model.gguf",
        gpu_layers=35,
        n_threads=8,
    )

    engine = LlamaInference(config)

    # Synchronous style
    response = await engine.chat_completion([
        {"role": "user", "content": "Explain quantum computing."}
    ])

    # Streaming
    async for token in engine.chat_completion_stream(messages):
        print(token, end="", flush=True)

Backend Characteristics
^^^^^^^^^^^^^^^^^^^^^^^

- ``SUPPORTS_STEERING = False``
- ``MODULATION_TYPE = "sampling"``

Emotional modulation is achieved by adjusting ``temperature`` and ``top_p``
based on emotional state. The ``emotion_coefficients`` parameter is logged
but ignored.

Transformers Backend
--------------------

.. automodule:: elpis.llm.backends.transformers
   :noindex:
   :synopsis: HuggingFace Transformers inference backend

.. autoclass:: elpis.llm.backends.transformers.inference.TransformersInference
   :members:
   :undoc-members:
   :show-inheritance:

The transformers backend provides full HuggingFace model support with
steering vector injection:

.. code-block:: python

    from elpis.llm.backends.transformers import TransformersInference
    from elpis.llm.backends.transformers.config import TransformersConfig

    config = TransformersConfig(
        path="meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype="bfloat16",
        steering_layer=15,
        emotion_vectors_dir="./data/vectors",
    )

    engine = TransformersInference(config)

    # With emotional modulation
    response = await engine.chat_completion(
        messages=[{"role": "user", "content": "Hello!"}],
        emotion_coefficients={
            "excited": 0.7,
            "calm": 0.2,
            "frustrated": 0.05,
            "depleted": 0.05,
        },
    )

Backend Characteristics
^^^^^^^^^^^^^^^^^^^^^^^

- ``SUPPORTS_STEERING = True``
- ``MODULATION_TYPE = "steering"``

Emotional modulation is achieved by injecting steering vectors into model
activations during the forward pass. The ``emotion_coefficients`` parameter
controls the blend of pre-trained vectors.

Steering Manager
----------------

.. automodule:: elpis.llm.backends.transformers.steering
   :noindex:
   :synopsis: Steering vector management

.. autoclass:: elpis.llm.backends.transformers.steering.SteeringManager
   :members:
   :undoc-members:
   :show-inheritance:

The ``SteeringManager`` handles loading, blending, and injection of
steering vectors:

.. code-block:: python

    from elpis.llm.backends.transformers.steering import SteeringManager

    # Initialize
    steering = SteeringManager(device="cuda", steering_layer=15)

    # Load pre-trained vectors
    steering.load_vectors("./data/vectors/llama-3.1-8b")
    print(steering.available_emotions)  # ['excited', 'calm', 'frustrated', 'depleted']

    # Compute blended vector
    coefficients = {"excited": 0.6, "calm": 0.4}
    blended = steering.compute_blended_vector(coefficients)

    # Apply to model during inference
    steering.apply_hook(model, blended)
    output = model.generate(**inputs)
    steering.remove_hook()

Vector File Format
^^^^^^^^^^^^^^^^^^

Steering vectors are stored as ``.pt`` files (PyTorch tensors):

- Each file contains a 1D tensor of shape ``(hidden_dim,)``
- Filename (without extension) becomes the emotion name
- Example: ``excited.pt``, ``calm.pt``, ``frustrated.pt``, ``depleted.pt``

Hook Mechanism
^^^^^^^^^^^^^^

The steering manager uses PyTorch forward hooks to inject vectors:

1. ``apply_hook()`` registers a hook on the target layer
2. During forward pass, the hook adds the steering vector to activations
3. ``remove_hook()`` removes the hook after generation

.. warning::

    Always call ``remove_hook()`` after generation to avoid memory leaks
    and unexpected behavior in subsequent generations.

Usage Examples
--------------

Basic Chat Completion
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from elpis.llm.backends import create_backend
    from elpis.config.settings import ModelSettings

    settings = ModelSettings(
        backend="llama-cpp",
        path="./model.gguf",
    )

    engine = create_backend(settings)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    response = await engine.chat_completion(messages)
    print(response)

Streaming with Emotional Modulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from elpis.emotion.state import EmotionalState

    # Get emotional parameters
    state = EmotionalState(valence=0.3, arousal=0.2)
    coefficients = state.get_steering_coefficients()

    # Stream with emotional modulation
    async for token in engine.chat_completion_stream(
        messages=messages,
        emotion_coefficients=coefficients,
    ):
        print(token, end="", flush=True)

Function Calling
^^^^^^^^^^^^^^^^

.. code-block:: python

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What's the weather in Paris?"}
    ]

    tool_calls = await engine.function_call(messages, tools)
    if tool_calls:
        print(tool_calls)
        # [{"name": "get_weather", "arguments": {"location": "Paris"}}]
