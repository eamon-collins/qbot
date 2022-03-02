
#include "Global.h"
#include "Game.h"


void Game::run_game(){
	int depth = 1;
	std::time_t time = std::time(0);
	bool gameOver = false;

	while(!gameOver){
		StateNode* currState = this->root;
		build_tree(currState, depth, time);

		gameOver = true;
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
	currState->prune_children();

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