
#include "Global.h"
#include "Game.h"

#ifndef NOVIZ
#include "Python.h"
#endif


void Game::run_game(){
	int depth = 1;
	std::time_t time = std::time(0);
	bool gameOver = false;

	StateNode* currState = this->root;
	//currState->generate_valid_children();
	//build_tree(currState, depth, time);

#ifndef NOVIZ
	//make it so all threads can use python interpreter. mutex to govern GIL
	Py_Initialize();
	// Build the name object
	PyObject* sysPath = PySys_GetObject("path");
	PyList_Append(sysPath, PyUnicode_FromString("/home/eamon/repos/Quoridor-Online/quoridor/client"));
#endif

	Move player_move;
	std::string viz_retval;
	while(!gameOver){
		if(currState->turn){
			StateNode* next_state = currState->get_best_move();
			if (next_state != nullptr){
				next_state->print_node();
				viz_retval = next_state->visualize_gamestate();
			}else{
				std::cout << "best_node is nullptr";
				return;
			}

			player_move = Move(viz_retval);
			std::cout << player_move << "\n";

			//find the move in the tree, create it if it doesn't exist yet.
			bool move_exists = false;
			for (auto &child : next_state->children){
				if (player_move == child.move){
					currState = &child;
					move_exists = true;
				}
			}
			if (!move_exists){
				move_exists = test_and_add_move(next_state, player_move);
				for (auto &child : next_state->children){
					if (player_move == child.move){
						currState = &child;
					}
				}
			}

			if (!move_exists){
				std::cout << "Could not produce player move state\n";
				return;
			}
		} else {
			std::cout << "reached bad turn state\n";
			return;
		}


		//gameOver = true;
	}
#ifndef NOVIZ
	Py_Finalize();
#endif
}



//recursive function for building the state tree
void build_tree(StateNode* currState, int depth, std::time_t starttime){
	std::time_t time = std::time(0) - starttime;
	if (depth >= MAXDEPTH || time > MAXTIME)
		return;

	//places a list of valid, evaluated moves in currState->children
	currState->generate_valid_children();

	//removes moves I don't think are good
	//currState->prune_children();

	std::deque<StateNode>::iterator it;
	for (it = currState->children.begin(); it != currState->children.end(); it++) {
		//must be careful when passing pointer to element in vector, if vector resizes this pointer will become invalid.
		//should be okay here but in case any unexplained phenoms happen, check this
		//this comment was prophetic^ too bad i didn't remember it was here til i changed the structure to fix this lol
		//it should be good now as list doesn't change element positions
		build_tree(&(*it), depth + 1, starttime);
	}	
}

//receives opposing move and sets the resulting gamestate as this->root
bool Game::receive_opposing_move(){
	return false;
}

//outputs move
bool Game::play_move(){
	return false;
}

Game::Game(StateNode* start){
	this->root = start;
	this->ply = start->ply;
}