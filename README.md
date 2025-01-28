# QBOT
### A bot to beat you at Quoridor.

Uses MCTS to simulate possible moves, evaluate them, and choose the best.
Homegrown, on-off development started in 2018, the code is messy and full of non-best practices.
But it will still beat you at Quoridor
<br>
<br>
<br>

## Compile and run
environment.yml specifies the python3.12 conda/mamba env required to run the current form of the GUI.
```
cd qbot
conda env create --name qenv --file=environment.yml
make qbot
./qbot -p 2
```
The `-p 2` starts it such that it is the players turn first. Otherwise the computer will play first.

If you don't have python3.12 installed outside of the env, you may need to add the `conda_env_base/lib` to your LD_LIBRARY_PATH when starting the binary.

For a lower overhead, easier to compile version, `make noviz` compiles a binary that accepts player input via text, and shows output solely on the terminal.



