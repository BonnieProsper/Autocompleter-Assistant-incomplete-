# Autocompleter-Assistant

An intelligent, self-learning autocompleter that uses a Trie, BK-tree (for fuzzy matching), and contextual bigram learning to improve text suggestions over time.

---

## Features

- Fast prefix-based search using Trie
- Fuzzy spelling correction via BK-tree
- Context learning with bigrams and reinforcement
- Adaptive fuzzy logic
- CLI interface with colorized output and statistics
- Evaluation harness for performance comparison

---

## Setup

```bash
git clone https://github.com/BonnieProsper/Autocompleter-Assistant.git
cd Autocompleter-Assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
