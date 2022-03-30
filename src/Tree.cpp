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

bool debugs = true;

//#include <cmath>

int fenceRows = 2*NUMROWS - 1;

using std::vector;
using std::rand;
using std::thread;



//finds a move in the set time limit.
//at the moment, I generate all child moves for leafs. If memory is a limiting factor, I may instead want to
//compare number of children to the number of valid moves possible from a position and expand if inequal. Need some way to select b/w unexpanded and expanded nodes tho
Move StateNode::get_best_move(){
	vector<Move> vmoves;
	this->generate_valid_moves(vmoves);

	
	StateNode* root = this;		
	//copying node so can generate trees in parallel.
	//might be expensive to copy if we are handed a precomputed tree, maybe benchmark this
	vector<thread> workers;
	vector<StateNode> copies;
	// for (int i = 0; i < NUM_THREADS; i++){
	// 	copies.push_back(*root);
	// 	thread t = thread(best_move_worker, i, &copies[i]);
	// 	workers.push_back(std::move(t));
	// }

	// for (auto it = workers.begin(); it != workers.end(); it++){
	// 	it->join();
	// }
	//FOR TESTING< SINGLE THREAD ONLY
	best_move_worker(1, root);

	return Move();


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
			double max_ucb = 0;
			//find highest UCB in group, random if tied, important to be able to break ties
			for (int i = 0; i < curr->children.size(); i++){
				StateNode* currChild = &(curr->children[i]);
				if (max_ucb <= currChild->UCB()){
					if (max_ucb != currChild->UCB())
						max_list.clear();
					
					max_list.push_back(currChild);
				}
			}

			int index = rand() % max_list.size();
			curr = max_list[index];
		}

		//we have reached a leaf node.
		//EXPANSION stage begins
		//expand the node fully so it's not a leaf, and choose one of the new children for simulation
		if (curr->generate_valid_children() == 0){
			std::cout << "No valid children during playout";
			return;
		}

		//SIMULATION/BACKPROPAGATION stage begins
		int index = rand() % curr->children.size();	
		//after playing out, score all the way up should be updated accordingly.
		//this means we can immediately delete them.
		curr->children[index].play_out();
		curr->children[index].children.clear();
	}

}


//generates all valid moves from this state, places them in this->children, and evaluates them.
int StateNode::generate_valid_children(){
	std::vector<Move> vmoves;
	generate_valid_moves(vmoves);

	for(auto it=vmoves.begin(); it != vmoves.end(); it++){
		test_and_add_move(this, *it);
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
	float chance_to_choose_fence = currPlayer.numFences > 0 ? .3 : 0;

	vector<Move> vmoves;
	int numFenceMoves = generate_valid_moves(vmoves);
	if (vmoves.size() == 0)
		return 0;

	float random;
	int rand_index = 0;
	bool valid_move = false;
	while(!valid_move && vmoves.size() != 0) {
		if (vmoves.size() == numFenceMoves){
			cout << currPlayer.col << ","<<currPlayer.row <<std::flush;
			return 0;
		}
		random = (float)rand() / RAND_MAX;
		if( random > chance_to_choose_fence){


			rand_index = rand() % (vmoves.size()-numFenceMoves);
			valid_move = test_and_add_move(this, vmoves[rand_index]);
		}else{//choose a fence move, relies on fences being at back of the vector
			rand_index = rand() % numFenceMoves;
			valid_move = test_and_add_move(this, vmoves[vmoves.size()-numFenceMoves+rand_index]);
			if (valid_move)
				numFenceMoves--;
		}

		//remove invalid moves so no infinite loop
		if(!valid_move)
			vmoves.erase(vmoves.begin() + random);
	}

	return this->children.size();
}

int StateNode::generate_valid_moves(vector<Move>& vmoves){
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
				if (currPlayer.col-1 >= NUMCOLS && !this->gamestate[2*currPlayer.row + 2*i][currPlayer.col-1]) //left
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
	if(currPlayer.numFences > 0){
		for (int i = 0; i < 2*NUMROWS-2; i++){
			for (int j=0; j < NUMCOLS-1; j++){
				bool horizontal = false;
				if(i %2 == 1){
					horizontal = true;
					if ((!this->gamestate[i-1][j] || !this->gamestate[i+1][j]) && !this->gamestate[i][j] && !this->gamestate[i][j+1]){
						vmoves.push_back(Move('f', i, j, horizontal));
						fenceMoves++;
					}
				}else if((!this->gamestate[i+1][j] || !this->gamestate[i+1][j+1]) && !this->gamestate[i][j] && !this->gamestate[i+2][j]){
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

	while (currState->p1.row != 0 && currState->p2.row != NUMROWS-1){
		if (currState->children.size() == 0)
			numChildren = currState->generate_random_child();
		else
			numChildren = currState->children.size();

		if ( numChildren == 1){
			currState = &(currState->children.front());
		}
		else if (numChildren != 0){
			choice = rand() % numChildren;
			std::deque<StateNode>::iterator it = std::next(currState->children.begin(), choice);
			//currState = &(*it);
			currState = &(currState->children[choice]);
		}
		else {
			printf("No valid children during playout, or no valid pawn moves");
			break;
		}

		currState->print_node();
	}

	//now that we have an end state check who wins and backpropagate that info
	//value of terminal state is based on how far the opponent is from winning, 
	//so the further they are from the end the better the game
	int scoreModifier;
	if (currState->p1.row == 0){
		scoreModifier = NUMROWS - currState->p2.row;
	}else{
		scoreModifier = -currState->p1.row - 1;
	}

	StateNode* winState = currState;

	while (currState->parent != nullptr){
		currState->score += scoreModifier;
		currState->visits += 1;
		currState = currState->parent;
	}
	currState->score += scoreModifier;
	currState->visits += 1; //propagate to root

	return winState;
}

double StateNode::UCB(){
	//return this->vi + 2* sqrt(log(this->parent->visits) / this->visits);
	return (this->score / this->visits) + 2* sqrt(log(this->parent->visits) / this->visits);
}


//deeply problematic, not used anywhere atm
//passes around references to objects in vector, but we are guaranteed that the vector is not resizing 
//during an update_vi pass
void StateNode::update_vi(){
	std::stack<StateNode*> s;
	StateNode* root = this;
	int count=0;
	while(root != nullptr || s.size() > 0){
		if (root != nullptr){
			s.push(root);

			if (root->children.size() >= 1)
				root = &(root->children.front());
			else
				root = nullptr;

		}

		StateNode* curr = s.top();
		s.pop();
		//traverse 
		count += curr->visits;

		// while (s.size() > 0 && curr->childrenIndex ==  
	//               s.top()->children.size() - 1) 
	//       { 
	//           curr = s.top(); 
	//           s.pop(); 
							
	//           //traverse
	//       } 
	}
}

//if a fence move, tests whether it will block either player from being able to reach the goal and doesn't add it if so
//else, adds state to the passed in state's children
bool test_and_add_move(StateNode* state, Move move){
	int difference = pathfinding(state, move);
	if (difference != -999){
		StateNode snode = StateNode(state, move, difference);
		//snode.evaluate(); //perhaps should just let score be set as difference
		state->children.push_back(snode);
		return true;
	}
	else return false;
}

//attempts to evaluate the score of a gamestate
//note that when a gamenode is created from other gamestates, its score is initialized as the 
//difference between players' shortest path to respective goals. Do not call evaluate()
//on the same node more than once to avoid problems.
void StateNode::evaluate(){
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


	double distanceCoeff, fenceCoeff, fence2Coeff = 1.0;

	this->score = distanceCoeff * this->score + fenceCoeff * currPlayer.numFences + otherPlayer.numFences;
}


//DANGEROUS: deque will not preserve pointer validity when things are erased.
//best solution is probably to filter moves in generate stage so we don't need pruning like this.
int StateNode::prune_children(){
	int dist_threshold = 1;
	int count = 0;
	//currently prunes fence moves too far away from either player, mostly to see effect on calc time
	for(std::deque<StateNode>::iterator it = this->children.begin(); it != this->children.end();) {
			if(it->move.type == 'f' && l1_f_p(it->move, this->p1) > dist_threshold && l1_f_p(it->move, this->p2) > dist_threshold){
				it = this->children.erase(it);
				count++;
			}
			else
				++it;
	}
	return count;
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

	this->p1.row = NUMROWS - 1;
	this->p1.col = NUMCOLS/2;
	this->p1.numFences = NUMFENCES;
	this->p2.row = 0;
	this->p2.col = NUMCOLS/2;
	this->p2.numFences = NUMFENCES;


	//only 8 possible vertical fence locations b/w the squares
	for(int i = 0; i < 2*NUMROWS-1; i++){
		for (int j = 0; j < NUMCOLS; j++){
			this->gamestate[i][j] = false;
		}
	}
	//starting pawn locations not notated, just fences
	// gamestate[0][4] = true;
	// gamestate[17][4] = true;
}

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
	this->score = score;
	this->visits = 1;
	this->ply = parent->ply + 1;
} 

/*StateNode::StateNode(StateNode& rhs){
	parent = rhs.parent;
	children = rhs.children;
	move = rhs.move;
	p1 = rhs.p1;
	p2 = rhs.p2;
	turn = rhs.turn;
	memcpy(gamestate, rhs.gamestate, fenceRows*NUMCOLS * sizeof(bool));
	score = rhs.score;
	vi = rhs.vi;
	visits = rhs.visits;
	ply = rhs.ply;
	serial_type = rhs.serial_type;
}*/

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

	// //read gamestate and turn
	// unsigned char bit_chars[160];
	// int index = 0;
	// string tempString;
	// for (int i = 0; i < 20; i++){ 
	// 	tempString = bitset<8>(node_buffer[11 + i]).to_string();
	// 	for(int j = 0; j<8; j++){
	// 		bit_chars[index] = tempString[j];
	// 		index++;
	// 	}
	// }

	// index = 0;	
	// for(int i = 0; i < (2*NUMROWS-1); i++){
	// 	for(int j = 0; j < NUMCOLS; j++){
	// 		this->gamestate[i][j] = bit_chars[index] == '1';
	// 		//std::cout << this->gamestate[i][j];
	// 		index++;
	// 	}
	// 	//std::cout << "\n";
	// }
	// //std::cout << "\n\n";
	// this->turn = (bit_chars[index] == '1');
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
	std::cout << this->move.type << " " << this->score << " " << this->vi << "\n";
	std::cout << "visits " << this->visits << " " << this->ply << " " << this->serial_type << "\n";
	std::cout << this->turn << "\n";
	int index = 0;	
	for(int i = 0; i < 17; i++){
		for(int j = 0; j < 9; j++){
			std::cout << this->gamestate[i][j];
			index++;
		}
		std::cout << "\n";
	}
	std::cout << "\n\n";
	
}
