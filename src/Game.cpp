
#include "Global.h"
#include "Game.h"


void Game::run_game(){
	int depth = 1, time;
	bool gameOver = false;

	while(!gameOver){
		StateNode* currState = this->root;
		build_tree(currState, depth, time);

		gameOver = true;
	}
}

//recursive function for building the state tree
void build_tree(StateNode* currState, int depth, int starttime){
	int time = 0; //TODO: get systime in ms here 
	if (depth >= MAXDEPTH || time > MAXTIME)
		return;

	//places a list of valid, evaluated moves in currState->children
	currState->generate_valid_children();

	//removes moves I don't think are good
	currState->prune_children();

	for(int i = 0; i < currState->children.size(); i++){
		//must be careful when passing pointer to element in vector, if vector resizes this pointer will become invalid.
		//should be okay here but in case any unexplained phenoms happen, check this
		build_tree(&currState->children[i], depth + 1, starttime);
	}	
}

//receives opposing move and sets the resulting gamestate as this->root
bool Game::receive_opposing_move(){

}

//outputs move
bool Game::play_move(){

}

Game::Game(StateNode* start){
	this->root = start;
	this->ply = start->ply;
}