/*Eamon Collins 
Functions for building/managing the state-tree
*/

#include "Tree.h"
#include "utility.h"
#include "storage.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <tuple>
#include <stack>
#include <bitset>
#include <thread>
#include <mutex>
#include <unordered_map>

#ifndef NOVIZ
#include "Python.h"
#endif

bool debugs = true;

int fenceRows = 2*NUMROWS - 1;

using std::vector;
using std::rand;
using std::thread;
using std::mutex;
using std::unordered_map;
using std::string;

mutex viz_mutex;


//finds a move in the set time limit.
//at the moment, I generate all child moves for leafs. If memory is a limiting factor, I may instead want to
//compare number of children to the number of valid moves possible from a position and expand if inequal. Need some way to select b/w unexpanded and expanded nodes tho
int StateNode::get_best_move(){
	StateNode* root = this;		
	//copying node so can generate trees in parallel.
	//might be expensive to copy if we are handed a precomputed tree, maybe benchmark this
	vector<thread> workers;
	vector<StateNode> copies;
	copies.reserve(num_threads); //necessary so addresses don't change out from under threads.
	for (int i = 0; i < num_threads; i++){
		//copy constructor takes care of deep copying/parent pointer of internal nodes,
		//just need to set root to nullptr so it knows it's a root node for the relevant subtree
		copies.push_back(*root);
		copies[i].parent = nullptr;
		copies[i].fix_parent_references();
		workers.push_back(thread(best_move_worker, i, &copies[i]));
	}

	for (auto it = workers.begin(); it != workers.end(); it++){
		it->join();
	}
	
	//gather consensus. Right now, we get the highest average score from each child of root.
	unordered_map<string, std::pair<int, float>> scores;
	for (auto &copy : copies){
		for (auto &child : copy.children){
			auto score = scores.find(child.move.unique_string());
			//cout << child.UCB() << " : " << child.visits << "\n";
			if ( score != scores.end()){
				score->second.first++;
				score->second.second += child.score;
			}else{
				std::pair<int, float> s = {1,child.score};
				scores[child.move.unique_string()] = s;
				if (!(child.move == Move(child.move.unique_string())))
					cout << child.move << "  string  " << child.move.unique_string() << " // " << Move(child.move.unique_string()) << "\n";
			}
		}
	}
	//selects move with best avg score across copies
	//if multiple are tied, chooses among them at random
	float best_avg_score = -1000;
	vector<Move> best_moves;
	Move best_move;
	for (auto score : scores){
		// cout << score.first << " : " << score.second.second << " " << score.second.first << "\n";
		float avg_score = score.second.second / score.second.first;
		//cout << score.first << " : " << score.second.second << " " << score.second.first << "\n";
		if (avg_score >= best_avg_score){
			if (avg_score > best_avg_score)
				best_moves.clear();
			best_avg_score = avg_score;
			best_moves.push_back(Move(score.first));
		}
	}
	std::uniform_int_distribution<> int_gen(0, best_moves.size()-1);
	best_move = best_moves[int_gen(rng)];
	//now with the best move, get the node of the best UCB example of that move from across the copies
	//maybe not finding bestnode is problem with horizontal bool, or with conversion
	StateNode* best_node = nullptr;
	float node_score = -1000;
	for (auto &copy : copies){
		for (StateNode &child : copy.children){
			if (child.move == best_move && child.score > node_score){
				node_score = child.score;
				best_node = &child;
				break;
			}
		}
	}

	cout << "FINAL RESULT: " << best_move << " with score: " << best_avg_score <<"\n";
	

	//VERY IMPORTANT to copy the subtree over to the parent rather than returning the pointer.
	//also in future maybe want to average out results from all subtrees? depends how much time it is to compute vs avg, but 
	//very likely worth the trouble.
	//StateNode* best_child;
	int i=0;
	bool found_match = false;
	for(auto& child : this->children){
		if (best_move == child.move){
			//child = *best_node;
			//best_child = &(this->children[i]);
			found_match = true;
			break;
		}
		i++;
	}
	if (!found_match){
		this->children.push_back(*best_node);
		i = this->children.size()-1;
	} else {
		this->children[i] = *best_node;
	}
	std::cout << "adopted child stats: " ;
	output_tree_stats(&this->children[i]);
	//maybe clear best_node's children's children?
	// for (auto& child : best_node->children){
	// 	child.children.clear();
	// }
	//takes a very long time, see why perhaps?
	//best_node->children.clear();
	//output_tree_stats(best_node);
	//best_node->print_node();

	this->children[i].fix_parent_references();

	//try v shallow?
	// this->children[i].score = best_node->score;
	// this->children[i].visits = best_node->visits;
	// this->children[i].move = best_node->move;
	// this->children[i].p1 = best_node->p1;
	// this->children[i].p2 = best_node->p2;
	// this->children[i].ply = best_node->ply;
	// this->children[i].turn = best_node->turn;
	// //this->children[i].gamestate = best_node->gamestate;
	// memcpy(this->children[i].gamestate, best_node->gamestate, (2*NUMROWS-1)*NUMCOLS * sizeof(bool));

	return i;
}

void best_move_worker(int id, StateNode* root){
	//may get passed a precomputed tree from opponents thinking time, or just a leaf node.
	if (root->children.size() == 0)
		root->generate_valid_children();

	StateNode* curr;

	std::time_t start_time = std::time(0);
	while( std::time(0) - start_time < MAXTIME){
		curr = root;
		//SELECTION
		while(curr->children.size() != 0){
			vector<StateNode*> max_list;
			double max_ucb = -999, curr_ucb = 0;
			//find highest UCB in group, random if tied, important to be able to break ties
			for (int i = 0; i < curr->children.size(); i++){
				StateNode* currChild = &(curr->children[i]);
				curr_ucb = currChild->UCB();
				if (max_ucb <= curr_ucb){
					if (max_ucb != curr_ucb)
						max_list.clear();
					max_ucb = curr_ucb;
					max_list.push_back(currChild);
				}
			}

			if (max_list.size() == 0) {
				std::cout << "No max node currchildren: " << curr->children.size() << " max_ucb " << max_ucb << std::endl;
			}
			std::uniform_int_distribution<> int_gen(0, max_list.size()-1);
			int index = int_gen(rng);
			curr = max_list[index];
			
			//cout << "-";
		}

		//we have reached a leaf node.
		//EXPANSION stage begins
		//expand the node fully so it's not a leaf, and choose one of the new children for simulation
		if (curr->generate_valid_children() == 0){
			std::cout << "No valid children during playout";
			continue;
		}

		//SIMULATION/BACKPROPAGATION stage begins
		std::uniform_int_distribution<> int_gen(0, curr->children.size()-1);
		int index = int_gen(rng);
		//after playing out, score all the way up should be updated accordingly.
		//this means we can immediately delete them.
		curr->children[index].play_out();
		curr->children[index].children.clear();
	}

	//while UCB is used to select move for playout, use score for best move selection
	//find best UCB so far for root's direct children
	vector<StateNode*> max_list;
	double max_score = -1000, curr_score = 0;
	//find highest UCB in group, random if tied, important to be able to break ties
	for (int i = 0; i < root->children.size(); i++){
		StateNode* currChild = &(root->children[i]);
		curr_score = currChild->score;
		if (max_score < curr_score){
			if (max_score != curr_score)
				max_list.clear();
			max_score = curr_score;
			max_list.push_back(currChild);
		}
	}

	std::uniform_int_distribution<> int_gen(0, max_list.size()-1);
	int index = int_gen(rng);
	cout << "Worker " << id << " proposes " << max_list[index]->move << " with score: " << max_score <<"\n";
	//max_list[index]->print_node();
	output_tree_stats(root);

}


//generates all valid moves from this state, places them in this->children, and evaluates them.
int StateNode::generate_valid_children(){
	std::vector<Move> vmoves;
	generate_valid_moves(vmoves);

	this->children.reserve(vmoves.size());
	for(auto& m : vmoves){
		test_and_add_move(this, m);
	}

	//evaluate will error with empty list
	if (this->children.size() == 0){
		printf("EMPTY CHILDREN VECTOR");
		return 0;
	}

	//here this->children should be list of all states created by valid moves.
	//pruning here?

	return this->children.size();
}

//
int StateNode::generate_random_child()
{
	Player currPlayer;
	Player otherPlayer;
	if (this->turn){
		currPlayer = this->p1;
		otherPlayer = this->p2;
	}
	else {
		currPlayer = this->p2;
		otherPlayer = this->p1;
	}
	//THIS MAY NEED TO BE MODIFIED, maybe adjusted according to other metrics?
	std::uniform_real_distribution<> float_gen(0.0, 1.0);
	float chance_to_choose_fence = currPlayer.numFences > 0 ? .3 : 0;

	vector<Move> vmoves;
	int numFenceMoves = generate_valid_moves(vmoves);
	if (vmoves.size() == 0)
		return 0;

	//FOR NOW maybe never need if this is only called in playout and then dumped
	//this->children.reserve(vmoves.size());

	float random;
	int rand_index = 0;
	bool valid_move = false;
	while(!valid_move && vmoves.size() != 0) {
		if (vmoves.size() == numFenceMoves){
			//cout << currPlayer.row << ","<<currPlayer.col << "\n" <<std::flush;
			return 0;
		}
		random = float_gen(rng);
		if( random > chance_to_choose_fence){
			std::uniform_int_distribution<> int_gen(0,vmoves.size()-numFenceMoves-1);
			rand_index = int_gen(rng);
			valid_move = test_and_add_move(this, vmoves[rand_index]);
			//remove invalid moves so no infinite loop
			if(!valid_move)
				vmoves.erase(vmoves.begin() + rand_index);
		}else{//choose a fence move, relies on fences being at back of the vector
			std::uniform_int_distribution<> int_gen(0,numFenceMoves-1);
			rand_index = int_gen(rng);
			valid_move = test_and_add_move(this, vmoves[vmoves.size()-numFenceMoves+rand_index]);
			if (!valid_move){
				vmoves.erase(vmoves.begin() + (vmoves.size() - numFenceMoves+rand_index));
				numFenceMoves--;
			}
		}
	}

	return this->children.size();
}

int StateNode::generate_valid_moves(vector<Move>& vmoves){
	Player currPlayer;
	Player otherPlayer;
	if (this->turn){
		currPlayer = this->p1;
		otherPlayer = this->p2;
	} else {
		currPlayer = this->p2;
		otherPlayer = this->p1;
	}

	//If there are no more fences to be placed, playing out is a courtesy. Should be able to speed it up.
	if (p1.numFences == 0 && p2.numFences == 0){
		// vector<Move> myTurnMoves, theirTurnMoves;
		// int difference = pathfinding(this, move, myTurnMoves, theirTurnMoves);
		vector<Move> path;
		int difference = pathfinding(this, path);
		if (difference == -999)
			std::cout << "PAWN BLOCKED" <<std::endl;
		//path[0] is current location
		vmoves.push_back(path[1]);
		return 1;
	}

	//PAWN MOVES
	//check up/down
	for(int i = -1; i < 2; i+=2){
		if(currPlayer.row +i >= 0 && currPlayer.row +i < NUMROWS && !this->gamestate[2*currPlayer.row + i][currPlayer.col] && (otherPlayer.row != currPlayer.row +i || otherPlayer.col != currPlayer.col))
			vmoves.push_back(Move('p',currPlayer.row+i, currPlayer.col, false));
		else if(otherPlayer.row == currPlayer.row +i && otherPlayer.col == currPlayer.col && !this->gamestate[2*currPlayer.row + i][currPlayer.col]){
			if(currPlayer.row+2*i >= 0 && currPlayer.row+2*i < NUMROWS && !this->gamestate[2*currPlayer.row + 2*i][currPlayer.col])//jump the other player
				vmoves.push_back(Move('p', currPlayer.row+2*i, currPlayer.col, false));
			else if(this->gamestate[2*currPlayer.row + 2*i][currPlayer.col]){ //if other player has wall behind them
				if (currPlayer.col+1 < NUMCOLS && !this->gamestate[2*currPlayer.row + 2*i][currPlayer.col]) //right
					vmoves.push_back(Move('p', currPlayer.row+i, currPlayer.col+1, false));
				if (currPlayer.col-1 >= NUMCOLS && !this->gamestate[2*currPlayer.row + 2*i][currPlayer.col-1]) //left //TOOK OUT -1 on the last index, ie currPLayer.col-1, not sure if significant
					vmoves.push_back(Move('p', currPlayer.row+i, currPlayer.col-1, false));
			}
		}
	}
	//check left/right
	for(int i = -1; i < 2; i+=2){
		if(currPlayer.col +i >= 0 && currPlayer.col +i < NUMCOLS && !this->gamestate[2*currPlayer.row][currPlayer.col + (i==1 ? 0 : -1)] && (otherPlayer.col != currPlayer.col +i || otherPlayer.row != currPlayer.row))
			vmoves.push_back(Move('p',currPlayer.row, currPlayer.col+i, false));
		else if(otherPlayer.row == currPlayer.row && otherPlayer.col == currPlayer.col+i && !this->gamestate[2*currPlayer.row][currPlayer.col + (i==1 ? 0 : -1)]){
			if(currPlayer.col+2*i >= 0 && currPlayer.col+2*i < NUMCOLS && !this->gamestate[2*currPlayer.row][currPlayer.col + (i==1 ? 1 : -2)])//jump the other player
				vmoves.push_back(Move('p', currPlayer.row, currPlayer.col+2*i, false));
			else if(this->gamestate[2*currPlayer.row][currPlayer.col + (i==1 ? 1 : -2)]){ //if other player has wall behind them
				if (currPlayer.row-1 >= 0 && !this->gamestate[2*currPlayer.row -1 ][currPlayer.col+i]) //up
					vmoves.push_back(Move('p', currPlayer.row-1, currPlayer.col+i, false));
				if (currPlayer.row+1 < NUMROWS && !this->gamestate[2*currPlayer.row + 1][currPlayer.col+i]) //down
					vmoves.push_back(Move('p', currPlayer.row+1, currPlayer.col+i, false));
			}
		}
	}

	
	//FENCE MOVES
	//need to check for intersecting fences, actually might already do that
	int fenceMoves = 0;
	bool horizontal = false;
	if(currPlayer.numFences > 0){
		for (int i = 0; i < 2*NUMROWS-2; i++){
			for (int j=0; j < NUMCOLS-1; j++){
				if(i %2 == 1){
					horizontal = true;
					if ((!this->gamestate[i-1][j] || !this->gamestate[i+1][j]) && !this->gamestate[i][j] && !this->gamestate[i][j+1]){
						vmoves.push_back(Move('f', i, j, horizontal));
						fenceMoves++;
					}
				}else if((!this->gamestate[i+1][j] || !this->gamestate[i+1][j+1]) && !this->gamestate[i][j] && !this->gamestate[i+2][j]){
					horizontal = false;
					vmoves.push_back(Move('f', i, j, horizontal));
					fenceMoves++;
				}
			}
		} 
	}

	return fenceMoves;
}

//play the game out from the current state with random moves and return winner and score
//later maybe test trying a couple close to bottom of the tree and averaging. Not canonical though?
//returns winning StateNode
StateNode* StateNode::play_out(){
	int numChildren;
	int choice;
	StateNode* currState = this;
	int scoreModifier = 0;

	std::time_t start_time = std::time(0);
	while (!currState->game_over()){
		//if there are no more fences this game is effectively over with pathfinding
		if (currState->p1.numFences == 0 && currState->p2.numFences == 0){
			// scoreModifier = pathfinding(currState);
			//cout << "ENDING PLAYOUT WITH SCORE " << scoreModifier << std::endl;
			break;
		}

		if (currState->children.size() == 0)
			numChildren = currState->generate_random_child();
			
		else
			numChildren = currState->children.size();

		if ( numChildren == 1){
			currState = &(currState->children.front());
		}
		else if (numChildren != 0){
			std::uniform_int_distribution<> int_gen(0, numChildren-1);
			choice = int_gen(rng);
			std::vector<StateNode>::iterator it = std::next(currState->children.begin(), choice);
			currState = &(*it);
			//currState = &(currState->children[choice]);
		}
		else {
			//currState->visualize_gamestate();
			//printf("No valid children during playout, or no valid pawn moves\n");
			
			break;
		}
		
		if( std::time(0) - start_time > 4){
			cout <<"VISUALIZING BROKEN STATE";
			currState->print_node();
			cout << std::flush;
			currState->visualize_gamestate();
			//currState->parent->visualize_gamestate();
			break;
		}
		//currState->print_node();
	}

	//now that we have an end state check who wins and backpropagate that info
	//value of terminal state is based on how far the opponent is from winning, 
	//so the further they are from the end the better the game
	scoreModifier = pathfinding(currState);

	StateNode* winState = currState;
	while (currState->parent != nullptr){
		currState->score += scoreModifier;
		currState->visits += 1;
		currState = currState->parent;
	}
	currState->score += scoreModifier;
	currState->visits += 1; //propagate to root MIGHT BE BAD IDEA

	return winState;
}

double StateNode::UCB() const{
	//return this->vi + 2* sqrt(log(this->parent->visits) / this->visits);
	return (this->score / this->visits) + 2* sqrt(log(this->parent->visits) / this->visits);
}

int StateNode::game_over() const{
	if (this->p1.row == 8) {
		return 1;
	} else if (this->p2.row == 0) {
		return 2;
	} else {
		return 0;
	}
}

//if a fence move, tests whether it will block either player from being able to reach the goal and doesn't add it if so
//else, adds state to the passed in state's children
bool test_and_add_move(StateNode* state, Move move){	
	// more computation to pathfind on pawn moves too, but if we input to score need to know
	int difference = pathfinding(state, move);
	// if (move.type == 'f')
	// 	difference = pathfinding(state, move);
	if (difference != -999){
		state->children.push_back(StateNode(state, move, difference));
		state->children.back().parent = state;
		return true;
	}
	else return false;
}

//attempts to evaluate the score of a gamestate
//note that when a gamenode is created from other gamestates, its score is initialized as the 
//difference between players' shortest path to respective goals. Do not call evaluate()
//on the same node more than once to avoid problems.
// void StateNode::evaluate(){
// 	Player currPlayer;
// 	Player otherPlayer;
// 	if (this->turn){
// 		currPlayer = this->p1;
// 		otherPlayer = this->p2;
// 	}
// 	else {
// 		currPlayer = this->p2;
// 		otherPlayer = this->p1;
// 	}


// 	double distanceCoeff, fenceCoeff, fence2Coeff = 1.0;

// 	this->score = distanceCoeff * this->score + fenceCoeff * currPlayer.numFences + otherPlayer.numFences;
// }


//DANGEROUS: deque will not preserve pointer validity when things are erased.
//best solution is probably to filter moves in generate stage so we don't need pruning like this.
int StateNode::prune_children(){
	int dist_threshold = 1;
	int count = 0;
	//currently prunes fence moves too far away from either player, mostly to see effect on calc time
	for(std::vector<StateNode>::iterator it = this->children.begin(); it != this->children.end();) {
			if(it->move.type == 'f' && l1_f_p(it->move, this->p1) > dist_threshold && l1_f_p(it->move, this->p2) > dist_threshold){
				it = this->children.erase(it);
				count++;
			}
			else
				++it;
	}
	return count;
}

void StateNode::fix_parent_references() {
	if (children.size() == 0)
		return;
	for (auto it = children.begin(); it != children.end(); it++){
		it->parent = this;
		it->fix_parent_references();
	}
}

//used to create the root node, starting gamestate.
StateNode::StateNode(bool turn){	
	this->parent = nullptr;
	this->turn = turn;
	this->ply = 0;
	this->visits = 1; //have to initialize to 1 to not divide by 0
	this->serial_type = '0';
	this->score = 1; //I think this should be nonzero to start

	// //THIS IS FOR TESTING READ/WRITE, take out later
	// this->score = .54321;
	// this->vi = .987654321;
	// this->visits = 12121;
	// this->ply = 699;

	this->p1.row = 0;
	this->p1.col = NUMCOLS/2;
	this->p1.numFences = NUMFENCES;
	this->p2.row = NUMROWS - 1;
	this->p2.col = NUMCOLS/2;
	this->p2.numFences = NUMFENCES;


	//only 8 possible vertical fence locations b/w the squares
	for(int i = 0; i < 2*NUMROWS-1; i++){
		for (int j = 0; j < NUMCOLS; j++){
			this->gamestate[i][j] = false;
		}
	}
}

//copy constructor, should create a deep copy of the subtree inclusive of the passed in root
// StateNode::StateNode(const StateNode &s) :
// 	move(s.move), p1(s.p1), p2(s.p2), turn(s.turn), score(s.score), vi(s.vi), visits(s.visits),
// 	ply(s.ply), serial_type(s.serial_type)
// {
// 	memcpy(this->gamestate, s.gamestate, fenceRows*NUMCOLS * sizeof(bool));

// 	for (int i = 0; i < s.children.size(); i ++){
// 		children.push_back(s.children[i]);
// 		children[i].parent = this;
// 	}
// }

//do not use this method directly, use as part of generate_valid_children after the move has been verified as valid
StateNode::StateNode(StateNode* parent, Move move, int score){
	this->p1 = parent->p1;
	this->p2 = parent->p2;


	//copies gamestate from parent, then we have to modify 
	memcpy(this->gamestate, parent->gamestate, fenceRows*NUMCOLS * sizeof(bool));
	//this->gamestate = parent->gamestate;

	//if they moved a pawn
	if (move.type == 'p'){
		if(parent->turn){ //this means the move creating this state is player 1's turn
			this->p1.row = move.row;
			this->p1.col = move.col;
		}else{
			this->p2.row = move.row;
			this->p2.col = move.col;
		}
	}else if(move.type == 'f'){
		if (move.horizontal){
			this->gamestate[move.row][move.col] = true;
			this->gamestate[move.row][move.col+1] = true;
		}else{
			this->gamestate[move.row][move.col] = true;
			this->gamestate[move.row+2][move.col] = true;
		}

		if(parent->turn)
			this->p1.numFences--;
		else
			this->p2.numFences--;
	}

	this->turn = !parent->turn;
	this->move = move;
	this->parent = parent;
	this->score = 0;
	this->visits = 1;
	this->ply = parent->ply + 1;
} 

//used to create new nodes directly from the database character string representation
StateNode::StateNode(unsigned char* node_buffer){
	//read move
	Move move;
	move.type = node_buffer[0];
	unsigned char* row = &node_buffer[1];
	sscanf((char*)row, "%2d%1d", &move.row, &move.col);
	move.horizontal = (node_buffer[4] == '1');
	this->move = move;


	//read players
	Player p1, p2;
	unsigned char* c = &node_buffer[5];
	
	sscanf((char*)c, "%1d%1d%1d", &p1.row, &p1.col, &p1.numFences);
	c += 0x003; //moves array pointer up 3 bytes
	sscanf((char*)c, "%1d%1d%1d", &p2.row, &p2.row, &p2.numFences);
	if(node_buffer[7] == 't') p1.numFences = 10;
	if(node_buffer[10] == 't') p2.numFences = 10;
	this->p1 = p1;
	this->p2 = p2;

	// read gamestate and turn
	bool temp_gamestate[(2*NUMROWS-1)*NUMCOLS];
	for (int i =0; i < 19; i++){
		unsigned char inp = (unsigned char)node_buffer[11+i]-33;
		for (int j = 0; j < 8; j++){
			temp_gamestate[i*8+j] = inp & (true<<j);
		}
	}
	unsigned char last = (unsigned char)node_buffer[30]-33;
	temp_gamestate[152] = last & (true<<0);
	this->turn = last & (true<<1);
	memcpy(&(this->gamestate[0][0]), temp_gamestate, 153);


	//score and vi normalized to 0-1, with 7 digits after the decimal stored
	//add the 0. and null terminator and atof()
	unsigned char score[10] = {'0','.','0','0','0','0','0','0','0','\0'};
	unsigned char vi[10] = {'0','.','0','0','0','0','0','0','0','\0'};
	memcpy(&score[2], &node_buffer[31], 7);
	memcpy(&vi[2], &node_buffer[38], 7);

	this->score = atof((char*)score);
	this->vi = atof((char*)vi);


	unsigned char visits[8] = {'0','0','0','0','0','0','0','\0'};
	unsigned char ply[4] = {'0','0','0','\0'};

	memcpy(visits, &node_buffer[45], 7);
	memcpy(ply, &node_buffer[52], 3);
	this->visits = atoi((char*)visits);
	this->ply = atoi((char*)ply);
	
	
	this->serial_type = node_buffer[55];
}

//0 is empty square
//1 is empty fence lane
//2 is filled fence lane
//3 is p1
//4 is p2
//NEED A LOCK ON VISUALIZATION CAUSE PYTHON GIL WILL MESS YOU UP
//SIMPLE MUTEX around pyinit and py end should do it.
#ifndef NOVIZ
string StateNode::visualize_gamestate(){
	std::vector<int> x, y, walls;
	std::vector<double> color;

	const int EMPTY_TILE = 0;
	const int EMPTY_FENCE = 1;
	const int FILL_FENCE = 2;
	const int PLAYER1 = 3;
	const int PLAYER2 = 4;
	// int square_color;
	// for (int i = 0; i < 2*NUMROWS -1; i++){
	// 	for (int j = 0; j < 2*NUMCOLS -1; j++){
	// 		square_color = -1;
	// 		x.push_back(j);
	// 		y.push_back(i);
	// 		if(i % 2 == 0 && j % 2 == 0){
	// 			square_color = EMPTY_TILE;
	// 		}
	// 		else if (i % 2 == 1 && j % 2 == 1){
	// 			square_color = FILL_FENCE;
	// 		}else if (i % 2 != j % 2){
	// 			if (i % 2 == 0) // vertical fence
	// 				square_color = gamestate[2*NUMROWS-1 - i][j/2 + 1] ? FILL_FENCE : EMPTY_FENCE;
	// 			else
	// 				square_color = gamestate[2*NUMROWS-1 - i][j/2] ? FILL_FENCE : EMPTY_FENCE;
	// 		}
	// 		// gamestate[p1.row*2][p1.col] = PLAYER1;
	// 		// gamestate[p2.row*2][p2.col] = PLAYER2;

	// 		color.push_back(square_color);

	// 	}
	// }
	//copy gamestate so we can remove walls as we add them to python list
	bool copy_gamestate[2*NUMROWS - 1][NUMCOLS];
	memcpy(copy_gamestate, this->gamestate, (2*NUMROWS-1)*NUMCOLS * sizeof(bool));
	for (int i = 2*NUMROWS-2; i >= 0; i--){
		for (int j = 0; j < NUMCOLS; j++){
			if (i % 2 == 0 && copy_gamestate[i][j] && copy_gamestate[i+2][j]){ //vert wall
				// if (j==8)
				// 	continue;
				x.push_back(j);
				x.push_back(j+1);

				y.push_back(7-(i+1)/2);
				y.push_back(7-(i+1)/2);
				// y.push_back((i+1)/2+1);
				// y.push_back((i+1)/2+1);
				copy_gamestate[i][j] = false;
				copy_gamestate[i+2][j] = false;
			}else if (i % 2 == 1 && copy_gamestate[i][j] && copy_gamestate[i][j+1]){ //horizontal wall
				// x.push_back(j);
				// x.push_back(j+1);
				// y.push_back(i/2);
				// y.push_back(i/2);
				x.push_back(j);
				x.push_back(j);
				// y.push_back((i+1)/2);
				// y.push_back((i+1)/2);
				y.push_back(8-(i+1)/2);
				y.push_back(8-(i+1)/2);
				copy_gamestate[i][j] = false;
				copy_gamestate[i][j+1] = false;
			}
		}
	}

	//lock the visualization mutex so only one instance of python interpreter
	viz_mutex.lock();


	// Set PYTHONPATH TO working directory
	//setenv("PYTHONPATH",".",1);
	PyObject *pName, *pModule, *pDict, *pFunc, *px, *py, *pcolor, *presult;
	string retval;
	// Initialize the Python Interpreter
	//Py_SetProgramName("visualization");
	// Py_Initialize();

	// // Build the name object
	// PyObject* sysPath = PySys_GetObject("path");
	// PyList_Append(sysPath, PyUnicode_FromString("/home/eamon/repos/Quoridor-Online/quoridor/client"));
	pName = PyUnicode_FromString("bot_integration");

	// Load the module object
	pModule = PyImport_Import(pName);
	if (pModule == nullptr)
	{
	    PyErr_Print();
	    std::exit(1);
	}

	// pDict is a borrowed reference 
	pDict = PyModule_GetDict(pModule);

	// pFunc is also a borrowed reference 
	pFunc = PyDict_GetItemString(pDict, "visualize_gamestate");

	if (PyCallable_Check(pFunc))
	{
		int num_wall_coords = x.size() > 0 ? x.size() : 1;
		px=PyList_New(num_wall_coords);
		py=PyList_New(num_wall_coords);
		if (x.size() == 0){
			PyList_SetItem(px, 0, Py_BuildValue("i",-1));
			PyList_SetItem(py, 0, Py_BuildValue("i",-1));
		} else {
			for (int i =0; i < num_wall_coords; i++){
				PyList_SetItem(px, i, Py_BuildValue("i",x[i]));
				PyList_SetItem(py, i, Py_BuildValue("i",y[i]));
			}
		}
		PyErr_Print();
		PyObject *p1w, *p1x, *p1y, *p2w, *p2x, *p2y;
		Player& currPlayer = this->turn ? p1 : p2;
		Player& otherPlayer = !this->turn ? p2 : p2;
		p1w = Py_BuildValue("i", currPlayer.numFences);
		p2w = Py_BuildValue("i", otherPlayer.numFences);
		p1x = Py_BuildValue("i", currPlayer.col);
		p1y = Py_BuildValue("i", currPlayer.row);
		p2x = Py_BuildValue("i", otherPlayer.col);
		p2y = Py_BuildValue("i", otherPlayer.row);

		//Actually call the method to visualize the gamestate.
		presult=PyObject_CallFunctionObjArgs(pFunc,px,py, p1w, p1x, p1y, p2w, p2x, p2y, NULL);
		if (presult != NULL){
			PyObject *bytes = PyUnicode_AsUTF8String(presult);
			retval = PyBytes_AsString(bytes);
			Py_DECREF(bytes);
			if (retval == "quit") {
				cout << "GUI exited, game over" << std::endl;
				exit(0);
			}
			cout << "Intaking move: " << retval << "\n";
		}

		Py_DECREF(p1w);
		Py_DECREF(p1x);
		Py_DECREF(p1y);
		Py_DECREF(p2w);
		Py_DECREF(p2x);
		Py_DECREF(p2y);
		PyErr_Print();
	} else 
	{
		PyErr_Print();
	}
	// Clean up
	Py_DECREF(px);
	Py_DECREF(py);
	Py_DECREF(pModule);
	Py_DECREF(pName);

	// Finish the Python Interpreter
	//Py_Finalize();

	viz_mutex.unlock();

	return retval;
}
#else
//want wasd corresponding to pawn move direction, or
//fx,y corresponding to fence play. dont give bad input.
string StateNode::visualize_gamestate(){
	string input_move;
	getline(cin, input_move);
	return input_move;
}
#endif

std::ostream& operator<<(std::ostream &strm, const StateNode &sn) {
	return strm << (sn.turn ? "player2" : "player1") << "\t"<< (sn.move.type=='f' && sn.move.horizontal ? "h " : "v ") << sn.move.type << " -> (" << sn.move.row << "," << sn.move.col <<")\n";
	//print out opposite of what turn says, because we want the player who made the move resulting in this gamestate
}

bool StateNode::operator==(const StateNode& node) {
	bool isEqual = false;
	if (node.move.row == this->move.row &&
		node.move.col == this->move.col &&
		node.p1.col == this->p1.col &&
		node.p1.row == this->p1.row &&
		node.p1.numFences == this->p1.numFences &&
		node.p2.row == this->p2.row &&
		node.p2.col == this->p2.col &&
		node.p2.numFences == this->p2.numFences &&
		node.visits == this->visits &&
		node.ply == this->ply)
	{
		isEqual = true;
	}

	return isEqual;
}

void StateNode::print_node(){
	std::ostringstream sstream; 
	sstream << "Last move: " << this->move << "\n" << "Score: " << this->score << "\n";
	sstream << "Visits: " << this->visits << "   Ply: " << this->ply << "    Turn: " << this->turn << "\n";
	int index = 0;	
	for(int i = 16; i >= 0; i--){
		for(int j = 0; j < 9; j++){
			if ( i % 2 == 0 ) {

				if ( (p1.row == i/2 && p1.col == j) || (p2.row == i/2 && p2.col == j) )
					sstream << 'X';
				else
					sstream << '0';

				if (gamestate[i][j])
					sstream << '|';
				else
					sstream << ' ';
			} else {
				if (gamestate[i][j] && j<8 && gamestate[i][j+1])
					sstream << "--";
				else if (gamestate[i][j])
					sstream << "- ";
				else //if (j % 2 == 1)
					sstream << "  ";
			}
			index++;
		}
		sstream << "\n";
	}
	sstream << "\n\n";

	std::cout << sstream.str();
}
	

std::ostream& operator<<(std::ostream& os, const Move& m){
	return os << m.type << (m.type == 'f' ? (m.horizontal ? " horizontal " : " vertical ") : " ") << m.row << "," << m.col;
}
