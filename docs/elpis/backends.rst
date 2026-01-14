Inference Backends
==================

Elpis supports multiple inference backends, each with different trade-offs
in terms of performance, features, and resource requirements.

Backend Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Feature
     - llama-cpp
     - transformers
   * - **Model Format**
     - GGUF quantized models
     - HuggingFace models
   * - **Memory Usage**
     - Lower (4-8 bit quantization)
     - Higher (full precision)
   * - **GPU Support**
     - CUDA, ROCm, CPU
     - CUDA, ROCm, CPU
   * - **Emotional Modulation**
     - Sampling parameters
     - Steering vectors
   * - **Modulation Depth**
     - Surface-level
     - Activation-level
   * - **Streaming**
     - Yes
     - Yes
   * - **Best For**
     - Fast inference, limited VRAM
     - Research, steering vectors

Choosing a Backend
^^^^^^^^^^^^^^^^^^

**Use llama-cpp when:**

- You need fast inference on limited hardware
- You want to run quantized models (4-bit, 5-bit)
- Sampling parameter modulation is sufficient
- You don't need activation-level steering

**Use transformers when:**

- You have sufficient GPU VRAM for full models
- You want activation-level emotional modulation
- You need steering vector support
- You're doing research on model behavior

llama-cpp Backend
-----------------

The llama-cpp backend uses the `llama-cpp-python <https://github.com/abetlen/llama-cpp-python>`_
library to run GGUF quantized models.

Installation
^^^^^^^^^^^^

.. code-block:: bash

    # CPU-only
    pip install llama-cpp-python

    # With CUDA support
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

    # With ROCm support (AMD GPUs)
    CMAKE_ARGS="-DGGML_HIP=on" pip install llama-cpp-python

Configuration
^^^^^^^^^^^^^

.. code-block:: bash

    ELPIS_MODEL_BACKEND=llama-cpp
    ELPIS_MODEL_PATH=./data/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
    ELPIS_MODEL_GPU_LAYERS=35      # Layers on GPU (0 for CPU-only)
    ELPIS_MODEL_N_THREADS=8        # CPU threads
    ELPIS_MODEL_CONTEXT_LENGTH=8192

Key Configuration Options
"""""""""""""""""""""""""

``gpu_layers``
    Number of model layers to offload to GPU. Set to 0 for CPU-only,
    or match your model's layer count for full GPU offloading.

``n_threads``
    CPU threads for matrix operations. Typically set to your physical
    core count.

``chat_format``
    Chat template format. Use ``llama-3`` for Llama 3.x models, or
    consult your model's documentation.

Emotional Modulation
^^^^^^^^^^^^^^^^^^^^

The llama-cpp backend modulates emotional state through sampling parameters:

.. code-block:: python

    # High arousal -> more focused (lower temperature)
    # Low arousal -> more exploratory (higher temperature)
    temperature = 0.7 + (-0.2 * arousal)  # Range: 0.5 to 0.9

    # High valence -> broader sampling
    # Low valence -> more conservative
    top_p = 0.9 + (0.1 * valence)  # Range: 0.8 to 1.0

This approach is simple and efficient but provides surface-level modulation.
The model's internal representations are not directly affected.

Hardware Detection
^^^^^^^^^^^^^^^^^^

The backend automatically detects available hardware:

.. code-block:: python

    from elpis.utils.hardware import detect_hardware, HardwareBackend

    backend = detect_hardware()
    # Returns: HardwareBackend.CUDA, HardwareBackend.ROCM, or HardwareBackend.CPU

Override with ``ELPIS_MODEL_HARDWARE_BACKEND=cuda|rocm|cpu``.

Transformers Backend
--------------------

The transformers backend uses HuggingFace Transformers for inference with
support for steering vector injection.

Installation
^^^^^^^^^^^^

.. code-block:: bash

    pip install torch transformers

    # For GPU acceleration
    pip install torch --index-url https://download.pytorch.org/whl/cu121

Configuration
^^^^^^^^^^^^^

.. code-block:: bash

    ELPIS_MODEL_BACKEND=transformers
    ELPIS_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
    ELPIS_MODEL_TORCH_DTYPE=bfloat16
    ELPIS_MODEL_STEERING_LAYER=15
    ELPIS_MODEL_EMOTION_VECTORS_DIR=./data/vectors/llama-3.1-8b

Key Configuration Options
"""""""""""""""""""""""""

``torch_dtype``
    Model weight precision. Use ``bfloat16`` for modern GPUs, ``float16``
    for older GPUs, or ``float32`` for CPU/debugging.

``steering_layer``
    Layer index for steering vector injection. Typically middle layers
    (layer 15 for 32-layer models) work well.

``emotion_vectors_dir``
    Directory containing trained steering vectors as ``.pt`` files.
    Each file should be named after its emotion (e.g., ``excited.pt``).

Steering Vector System
^^^^^^^^^^^^^^^^^^^^^^

The transformers backend implements activation-level emotional modulation
through steering vectors.

**How it works:**

1. Steering vectors are pre-trained direction vectors in activation space
2. During inference, vectors are blended based on emotional coefficients
3. The blended vector is added to activations at the target layer
4. This directly influences the model's internal representations

**Loading vectors:**

.. code-block:: python

    from elpis.llm.backends.transformers.steering import SteeringManager

    steering = SteeringManager(device="cuda", steering_layer=15)
    steering.load_vectors("./data/vectors/llama-3.1-8b")

    # Check available emotions
    print(steering.available_emotions)  # ['excited', 'calm', 'frustrated', 'depleted']

**Blending vectors:**

.. code-block:: python

    # Coefficients from emotional state
    coefficients = {
        "excited": 0.6,
        "calm": 0.3,
        "frustrated": 0.05,
        "depleted": 0.05,
    }

    blended = steering.compute_blended_vector(coefficients)

**Applying during inference:**

The steering manager uses PyTorch forward hooks to inject vectors:

.. code-block:: python

    # Apply hook before generation
    steering.apply_hook(model, blended_vector)

    # Generate (steering is active)
    output = model.generate(**inputs)

    # Always remove hook after
    steering.remove_hook()

Training Steering Vectors
^^^^^^^^^^^^^^^^^^^^^^^^^

Steering vectors can be trained using contrastive activation analysis:

1. Create paired prompts with contrasting emotional content
2. Run both through the model, extracting activations
3. Compute the mean difference vector
4. Normalize and save

Example training script usage:

.. code-block:: bash

    python scripts/train_emotion_vectors.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --layer 15 \
        --output ./data/vectors/llama-3.1-8b

The resulting ``.pt`` files can be loaded by the SteeringManager.

Backend API
-----------

All backends implement the :class:`~elpis.llm.base.InferenceEngine` interface:

.. code-block:: python

    from elpis.llm.base import InferenceEngine

    class InferenceEngine(ABC):
        SUPPORTS_STEERING: bool  # Does this backend support steering?
        MODULATION_TYPE: str     # "none", "sampling", or "steering"

        async def chat_completion(
            self,
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            emotion_coefficients: Optional[Dict[str, float]] = None,
        ) -> str: ...

        async def chat_completion_stream(
            self,
            messages: List[Dict[str, str]],
            ...
        ) -> AsyncIterator[str]: ...

        async def function_call(
            self,
            messages: List[Dict[str, str]],
            tools: List[Dict[str, Any]],
            ...
        ) -> Optional[List[Dict[str, Any]]]: ...

Backend Factory
^^^^^^^^^^^^^^^

Use the backend factory to create engines:

.. code-block:: python

    from elpis.llm.backends import create_backend, get_available_backends
    from elpis.config.settings import ModelSettings

    # Check what's available
    print(get_available_backends())
    # {'llama-cpp': True, 'transformers': True}

    # Create an engine
    settings = ModelSettings(backend="llama-cpp", path="./model.gguf")
    engine = create_backend(settings)

    # Use it
    response = await engine.chat_completion([
        {"role": "user", "content": "Hello!"}
    ])

Performance Tips
----------------

llama-cpp Performance
^^^^^^^^^^^^^^^^^^^^^

- Set ``gpu_layers`` to offload as many layers as VRAM allows
- Use quantized models (Q4_K_M, Q5_K_M) for speed/quality balance
- Match ``n_threads`` to physical core count
- Enable mmap by default for faster loading

Transformers Performance
^^^^^^^^^^^^^^^^^^^^^^^^

- Use ``bfloat16`` on Ampere+ GPUs for best performance
- Consider Flash Attention 2 for long contexts
- Use ``device_map="auto"`` for multi-GPU setups
- Enable compile mode for faster repeated inference:

  .. code-block:: python

      model = torch.compile(model)
