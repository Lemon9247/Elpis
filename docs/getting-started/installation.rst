Installation
============

This guide covers installing the Elpis ecosystem from source.

Prerequisites
-------------

Before installing Elpis, ensure you have:

- **Python 3.11 or higher** (3.10 minimum, 3.11+ recommended)
- **8-10GB RAM** (for loading the LLM model)
- **GPU recommended** (NVIDIA CUDA or AMD ROCm) for faster inference

Basic Installation
------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/elpis.git
      cd elpis

2. Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the base package:

   .. code-block:: bash

      pip install -e .

This installs all three components (elpis, mnemosyne, psyche) with their
core dependencies.

GPU Support
-----------

For GPU-accelerated inference, you need to install llama-cpp-python with
the appropriate backend.

NVIDIA CUDA
^^^^^^^^^^^

.. code-block:: bash

   CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir
   pip install -e .

AMD ROCm
^^^^^^^^

.. code-block:: bash

   CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python --no-cache-dir
   pip install -e .

Optional Dependencies
---------------------

Elpis has optional dependencies for different backends and features:

Steering Vectors (Transformers Backend)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For activation steering with emotional modulation:

.. code-block:: bash

   pip install torch transformers

Or install the transformers extra:

.. code-block:: bash

   pip install -e ".[transformers]"

This enables the transformers backend which supports steering vectors for
more nuanced emotional expression.

llama-cpp Backend
^^^^^^^^^^^^^^^^^

For the default llama-cpp backend (if not using GPU installation above):

.. code-block:: bash

   pip install -e ".[llama-cpp]"

All Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^

To install all optional dependencies:

.. code-block:: bash

   pip install -e ".[all]"

Development Installation
------------------------

For development, install the dev dependencies:

.. code-block:: bash

   pip install -e ".[dev]"
   pre-commit install

This includes:

- pytest and coverage tools
- ruff for linting and formatting
- mypy for type checking
- pre-commit hooks

Downloading a Model
-------------------

Elpis requires an LLM model for inference. For the llama-cpp backend,
download a GGUF quantized model:

.. code-block:: bash

   mkdir -p data/models

   # Using huggingface-cli
   huggingface-cli download \
     TheBloke/Llama-3.1-8B-Instruct-GGUF \
     Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
     --local-dir data/models --local-dir-use-symlinks False

Alternatively, use the provided download script:

.. code-block:: bash

   python scripts/download_model.py

Verifying Installation
----------------------

Verify the installation by checking that the CLI commands are available:

.. code-block:: bash

   elpis-server --help
   mnemosyne-server --help
   psyche --help

Run the test suite to ensure everything is working:

.. code-block:: bash

   pytest tests/ -v

Next Steps
----------

Once installed, proceed to the :doc:`quickstart` guide to learn how to
use the system.
