---
name: qbot
type: repo
agent: CodeActAgent
---

Repository: qbot
Description: A bot to beat you at Quoridor

Directory Structure:
- src/: Main MCTS implementation
- train/: Code to train the neural net value function
- QuoridorClient/: GUI code and integration

Setup:
- Run `make noviz` to compile a binary without visualization
- Use `./qbot -b` to initiate a self-play run

Guidelines:
- When making recommendations for features, read and consider the papers in the root directory
- Consider my implementation of StateNode when making suggestions

When relevant, source directly from the academic papers in the root directory.
