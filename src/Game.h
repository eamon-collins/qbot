//Game class, to run an instance of the game, deals with accepting opponents moves and outputting our moves, once it receives a move it organizes tree building as well
#pragma once

#include "Tree.h"
#include "inference.h"
#include <ctime>


class Game
{
public:
	StateNode* root; //the root statenode at any given time, not the gameroot but basically state of game currently.
	int ply;
	int num_threads;
	bool humanGame{};
	bool model_loaded{};
	ModelInference model;

	//
	Game(StateNode* start, int num_threads=4, std::string model_file="");

	//starts playing from root
	//this method organizes most of the bot's activities at a high level
	void run_game(); 
	//builds a tree while playing itself, preserves tree in between games and keeps refining it
	//hopefully this will be the backbone of training, both building a tree with lots of simulated
	//knowledge built in to the params and also eventually training a NN for individual state eval
	void self_play(const int timeout);
    void self_play(const std::string& checkpoint_file, const int games_per_checkpoint);
    void train_alpha(const std::string& checkpoint_file, int iterations_before_training = 50000);
    void better_self_play(const std::string& checkpoint_file, const int games_per_checkpoint);
	//kind of forgot this method existed when i made get_player_move, might be possible to make a more general
	//function that can receive input from either opposing bot or human
	bool receive_opposing_move();
	//currently asks for a human response to the currState, and attempts validation of the move.
	Move get_player_move(StateNode* currState);
	bool play_move();
};


void build_tree(StateNode* currState, int depth, std::time_t starttime);
