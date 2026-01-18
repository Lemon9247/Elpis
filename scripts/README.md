# Psyche Memory Database Utilities

Scripts for querying and analyzing Psyche's ChromaDB memory database.

## Setup

These scripts should be run from the project root with the venv activated:

```bash
cd /path/to/Elpis
source venv/bin/activate
cd utils
```

## Scripts

### list_memories.py

List and count memories in the database.

```bash
# Count all memories
python list_memories.py --count

# List recent memories (default 20)
python list_memories.py

# List more memories with full content
python list_memories.py -n 50 --full

# Filter by collection
python list_memories.py --collection short
python list_memories.py --collection long

# Filter by memory type
python list_memories.py --type episodic
python list_memories.py --type semantic
```

### search_memories.py

Semantic search across memories.

```bash
# Basic search
python search_memories.py "conversation about music"

# Search with more results
python search_memories.py "technical discussion" -n 20

# Search only long-term memories
python search_memories.py "important events" --collection long

# Show full content
python search_memories.py "emotional moment" --full
```

### read_memories.py

Interactive reader for browsing memories with full content.

```bash
# Interactive mode - browse with n/p/q/g/s commands
python read_memories.py

# Dump all memories to stdout
python read_memories.py --dump

# Show only the 10 most recent
python read_memories.py --latest 10

# Filter by text search
python read_memories.py --search "Willow"

# Combine filters
python read_memories.py --search "conversation" --latest 5 --dump
```

Interactive commands:
- `n` or Enter: next memory
- `p`: previous memory
- `g 5`: go to memory #5
- `s query`: search for "query" in content
- `q`: quit

### get_memory.py

Retrieve a specific memory by ID.

```bash
python get_memory.py abc12345-6789-...
```

### analyze_emotions.py

Analyze emotional distribution of memories.

```bash
# Basic analysis
python analyze_emotions.py

# Include ASCII scatter plot
python analyze_emotions.py --scatter

# Break down by memory type
python analyze_emotions.py --by-type

# All analyses
python analyze_emotions.py --scatter --by-type
```

### export_memories.py

Export memories to various formats.

```bash
# Export to JSON (default)
python export_memories.py

# Export to CSV
python export_memories.py --format csv

# Export to Markdown (human-readable)
python export_memories.py --format markdown

# Specify output file
python export_memories.py -o my_export.json

# Export only short-term memories
python export_memories.py --collection short
```

## Memory Structure

Each memory contains:

- **id**: Unique identifier (UUID)
- **content**: The memory content text
- **memory_type**: episodic, semantic, procedural, or emotional
- **status**: short_term or long_term
- **importance_score**: 0.0 to 1.0
- **created_at**: ISO timestamp
- **emotional_context**: valence, arousal, quadrant
- **tags**: List of tags
- **summary**: Optional summary text

## Emotional Quadrants

- **excited**: High valence (+), high arousal (+)
- **frustrated**: Low valence (-), high arousal (+)
- **calm**: High valence (+), low arousal (-)
- **depleted**: Low valence (-), low arousal (-)
