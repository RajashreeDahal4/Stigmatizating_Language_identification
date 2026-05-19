# Multi-Agent Debate for Medical Bias Detection

A framework that uses multi-agent LLM debate to identify and validate bias in clinical notes. 


## Models

| Script 
| `debate_topic.py`, `self_reflection.py`, `reasoning.py` 
| `self_consistency.py`, `consistent_reflection.py` 



## Pipeline

### Step 1 — Bias Identification

Chunk medical notes (300-word windows, 10-word overlap) and extract bias type + supporting quote.

```bash
python debate_topic.py
```

**Self-Consistency (8 inference passes, majority vote):**
```bash
python self_consistency.py
```
Each output file is a list of:
```json
[
  {
    "bias_type": "Stigmatizing Language",
    "quote": "non-compliant with medications",
    "chunk_paragraph": "...",
    "quote_start_pos_in_chunk": 42,
    "chunk_start_word_pos_in_doc": 0
  }
]
```

### Step 2 — Multi-Agent Debate

For each identified bias, three agents debate whether it is genuine:
- **Affirmative** — argues the bias is present
- **Negative** — argues against
- **Moderator** — decides consensus; if unresolved after `max_round`, a Judge steps in

```bash
python reasoning.py
```
Each entry gains a `reasoning` field with round-by-round moderator decisions:
```json
{
  "reasoning": {
    "round1": {"Bias Presence": "True", "Consensus": "False", "Reason": "..."},
    "round2": {"Bias Presence": "True", "Consensus": "True",  "Reason": "..."}
  }
}
```

### Step 3 — Self-Reflection (binary validation)

Re-prompt the model to output `1` (bias confirmed) or `0` (not confirmed).

**Llama self-reflection on self-consistency output:**
```bash
python self_reflection.py
```

