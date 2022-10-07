//Game class, to run an instance of the game, deals with accepting opponents moves and outputting our moves, once it receives a move it organizes tree building as well
#pragma once

#include "Tree.h"
#include <ctime>

class Game
{
public:
	StateNode* root; //the root statenode at any given time, not the gameroot but basically state of game currently.
	int ply;

	//
	Game(StateNode* start);

	//starts playing from root
	//this method organizes most of the bot's activities at a high level
	void run_game(); 
	//kind of forgot this method existed when i made get_player_move, might be possible to make a more general
	//function that can receive input from either opposing bot or human
	bool receive_opposing_move();
	//currently asks for a human response to the currState, and attempts validation of the move.
	Move get_player_move(StateNode* currState);
	bool play_move();
};


void build_tree(StateNode* currState, int depth, std::time_t starttime);