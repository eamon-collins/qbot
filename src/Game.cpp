
#include "Global.h"
#include "Game.h"
#include "storage.h"
#include "utility.h"
#include <chrono>


#ifndef NOVIZ
#include "Python.h"
#endif

using namespace std::chrono;

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
	    PyRun_SimpleString(R"(
import sys
import os
sys.path.append(os.path.join(sys.prefix, 'lib', 'python3.12', 'site-packages'))
		)");
	// Build the name object
	PyObject* sysPath = PySys_GetObject("path");
	PyList_Append(sysPath, PyUnicode_FromString("Quoridor-Online/quoridor/client"));
	PyList_Append(sysPath, PyUnicode_FromString("Quoridor-Online"));
#endif

	Move player_move;
	while(!gameOver){
		StateNode* next_state;
		if(!currState->turn){
			if(currState->p1.numFences == 0 && currState->p2.numFences == 0){
				vector<Move> testpath;
				int diff = pathfinding(currState, testpath, true);
			}
			int ret = currState->get_best_move();
			next_state = &(currState->children[ret]);
			if (next_state != nullptr){
				next_state->print_node();
				if (next_state->game_over()){
					std::cout << "!!! qbot Wins !!!" << std::endl;
					gameOver = true;
					break;
				}
				player_move = get_player_move(next_state);	
			}else{
				std::cout << "best_node is nullptr";
				return;
			}
		} else { //should only be run if we start game with player move
			next_state = currState;
			next_state->print_node();
			player_move = get_player_move(next_state);
		}

		//player_move = Move(viz_retval);
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
		//std::cout << "AFTER playermove";
		currState->print_node();

		if (!move_exists){
			std::cout << "Could not produce player move state\n";
			return;
		} else if (currState->game_over()) {
			//check win condition
			std::cout << "!!! Player1 Wins !!!" << std::endl;
			gameOver = true;
		}
	
	}
#ifndef NOVIZ
	Py_Finalize();
#endif
}

Move Game::get_player_move(StateNode* currState){
	Move player_move;
	std::string viz_retval;
	time_point<system_clock> start, end;
	while( true ){ //input loop, gathers input from player until it receives a valid move
		start = system_clock::now();
		viz_retval = currState->visualize_gamestate();
		end = system_clock::now();
		duration<double> length = end - start;
		if (length.count() < .2){//trys to detect error state so it doesnt open python instances on loop
			std::cout << length.count() << " too short, loop detected, terminating" << std::endl;
			//TERMINATE PROGRAM
			//If this is short it means the python is erroring out and returning too quick
			std::terminate();
			return Move();
		}
		
		if (viz_retval == "error"){
			std::cout << "Viz gamestate returned error" << std::endl;
			break;
		}
		try {
			player_move = Move(viz_retval);
			//basic checking done in Move constructor, but we need gamestate specific
			std::vector<Move> valid_moves;
			currState->generate_valid_moves(valid_moves);
			for (auto move : valid_moves){
				if (move.type == 'p')
					cout << move  << std::endl;
				if (move == player_move){//gen_valid_moves doesn't check pathfinding so we still need that
					// vector<Move> path; // just to debug pathfinding
					// pathfinding(currState, path, true);
					int pf = pathfinding(currState, move);
					cout << "Relative length to goal " << pf << std::endl;
					if (player_move.type == 'f' &&  pf == -999){
						std::cout << "Attempted move blocks a player from their goal" << std::endl;	
					}else{
						return player_move;
					}
					break;
				}
			}
			std::cout << "Attempted move doesn't appear to be possible" << std::endl;
		}catch(InvalidMoveException& e ){
			std::cout << e.what() << std::endl;
			std::cout << "Enter a valid move" << std::endl;
		}
	}
	return Move();//bad state, should have returned earlier
	
}

void Game::self_play(const int timeout) {
	bool gameOver = false;
	StateNode* nextState, *currState = this->root;
	std::time_t startTime = std::time(0);
	std::time_t currTime = startTime;


	while(!(currTime - startTime > timeout)) {
		while(!gameOver) {
			int ret = currState->get_best_move();
			nextState = &(currState->children[ret]);
			if (nextState != nullptr){
				nextState->print_node();
			}else{
				std::cout << "best_node is nullptr";
				return;
			}
		
			currState = nextState;
			// currState->print_node();
			gameOver = currState->game_over();
		}

		gameOver = false;
		currTime = std::time(0);
	}
}

void Game::self_play(const std::string& checkpoint_file, const int games_per_checkpoint) {
    StateNode* currState = nullptr;
    int games_played = 0;
    std::time_t startTime = std::time(0);

    while (true) { // Run indefinitely
        // Start new game from root
        currState = this->root;
        bool gameOver = false;

        // Play a single game
        while (!gameOver) {
            int ret = currState->get_best_move();
            StateNode* nextState = &(currState->children[ret]);
            
            if (nextState != nullptr) {
                nextState->print_node(); // Keep this for debugging/monitoring
                gameOver = nextState->game_over();
                currState = nextState;
            } else {
                std::cout << "best_node is nullptr" << std::endl;
                break;
            }
        }

        games_played++;

        // Checkpoint the tree periodically
        if (games_played % games_per_checkpoint == 0) {
            int nodes_saved = save_tree(root, checkpoint_file);
            std::cout << "Checkpoint after " << games_played << " games. Saved " 
                     << nodes_saved << " nodes to " << checkpoint_file << std::endl;
            
            // Output some statistics
            std::time_t currentTime = std::time(0);
            double minutes = difftime(currentTime, startTime) / 60.0;
            std::cout << "Training stats after " << minutes << " minutes:" << std::endl;
            std::cout << "Games played: " << games_played << std::endl;
            std::cout << "Games per hour: " << games_played / minutes << std::endl;
            // output_tree_stats(root);
        }

        // Optional: Break if some training target is met
        // if (some_condition) break;
    }
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

	std::vector<StateNode>::iterator it;
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

Game::Game(StateNode* start, int num_threads){
	this->root = start;
	this->ply = start->ply;
	this->num_threads = num_threads;
}
