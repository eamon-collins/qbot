/*Eamon Collins 
Functions for building/managing the state-tree
*/

#include "Tree.h"
#include "utility.h"
#include <cstring>
#include <cstdlib>
#include <tuple>
#include <stack>

int fenceRows = 2*NUMROWS - 1;

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

int StateNode::prune_children(){
	int dist_threshold = 1;
	//currently prunes fence moves too far away from either player, mostly to see effect on calc time
	for(std::vector<StateNode>::iterator it = this->children.begin(); it != this->children.end();) {
	    if(it->move.type == 'f' && l1_f_p(it->move, this->p1) > dist_threshold && l1_f_p(it->move, this->p2) > dist_threshold)
	    	it = this->children.erase(it);
	    else
	    	++it;
	}
}

//generates all valid moves from this state, places them in this->children, and evaluates them.
int StateNode::generate_valid_children(){
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

	std::vector<std::tuple<Move, int>> vmoves;
	

	//PAWN MOVES
	//check up/down
	for(int i = -1; i < 2; i+=2){
		if(currPlayer.row +i >= 0 && currPlayer.row +i < NUMROWS && !this->gamestate[2*currPlayer.row + i][currPlayer.col] && (otherPlayer.row != currPlayer.row +i || otherPlayer.col != currPlayer.col))
			test_and_add_move(vmoves, this, Move('p',currPlayer.row+i, currPlayer.col, false));
		else if(otherPlayer.row == currPlayer.row +i && otherPlayer.col == currPlayer.col && !this->gamestate[2*currPlayer.row + i][currPlayer.col]){
			if(currPlayer.row+2*i >= 0 && currPlayer.row+2*i < NUMROWS && !this->gamestate[2*currPlayer.row + 2*i][currPlayer.col])//jump the other player
				test_and_add_move(vmoves, this, Move('p', currPlayer.row+2*i, currPlayer.col, false));
			else if(this->gamestate[2*currPlayer.row + 2*i][currPlayer.col]){ //if other player has wall behind them
				if (currPlayer.col+1 < NUMCOLS && !this->gamestate[2*currPlayer.row + 2*i][currPlayer.col]) //right
					test_and_add_move(vmoves, this, Move('p', currPlayer.row+i, currPlayer.col+1, false));
				if (currPlayer.col-1 >= NUMCOLS && !this->gamestate[2*currPlayer.row + 2*i][currPlayer.col-1]) //left
					test_and_add_move(vmoves, this, Move('p', currPlayer.row+i, currPlayer.col-1, false));
			}
		}
	}
	//check left/right
	for(int i = -1; i < 2; i+=2){
		if(currPlayer.col +i >= 0 && currPlayer.col +i < NUMCOLS && !this->gamestate[2*currPlayer.row][currPlayer.col + (i==1 ? 0 : -1)] && (otherPlayer.col != currPlayer.col +i || otherPlayer.row != currPlayer.row))
			test_and_add_move(vmoves, this, Move('p',currPlayer.row, currPlayer.col+i, false));
		else if(otherPlayer.row == currPlayer.row && otherPlayer.col == currPlayer.col+i && !this->gamestate[2*currPlayer.row][currPlayer.col + (i==1 ? 0 : -1)]){
			if(currPlayer.col+2*i >= 0 && currPlayer.col+2*i < NUMCOLS && !this->gamestate[2*currPlayer.row][currPlayer.col + (i==1 ? 1 : -2)])//jump the other player
				test_and_add_move(vmoves, this, Move('p', currPlayer.row, currPlayer.col+2*i, false));
			else if(this->gamestate[2*currPlayer.row][currPlayer.col + (i==1 ? 1 : -2)]){ //if other player has wall behind them
				if (currPlayer.row-1 >= 0 && !this->gamestate[2*currPlayer.row -1 ][currPlayer.col+i]) //up
					test_and_add_move(vmoves, this, Move('p', currPlayer.row-1, currPlayer.col+i, false));
				if (currPlayer.row+1 < NUMROWS && !this->gamestate[2*currPlayer.row + 1][currPlayer.col+i]) //down
					test_and_add_move(vmoves, this, Move('p', currPlayer.row+1, currPlayer.col+i, false));
			}
		}
	}

	
	//FENCE MOVES
	//need to check for intersecting fences, actually might already do that
	if(currPlayer.numFences > 0){
		for (int i = 0; i < 2*NUMROWS-2; i++){
			for (int j=0; j < NUMCOLS-1; j++){
				bool horizontal = false;
				if(i %2 == 1){
					horizontal = true;
					if ((!this->gamestate[i-1][j] || !this->gamestate[i+1][j]) && !this->gamestate[i][j] && !this->gamestate[i][j+1])
						test_and_add_move(vmoves, this, Move('f', i, j, horizontal));
				}else if((!this->gamestate[i+1][j] || !this->gamestate[i+1][j+1]) && !this->gamestate[i][j] && !this->gamestate[i+2][j]){
					test_and_add_move(vmoves, this, Move('f', i, j, horizontal));
				}
			}
		} 
	}


	//evaluate all child moves now
	for(int i = 0; i < this->children.size(); i++){
		this->children[i].evaluate();
	}

	//here this->children should be list of all states created by valid moves.
	//pruning here?

	return this->children.size();
}


//play the game out from the current state with random moves and backpropagate the result
void StateNode::play_out(){
	int numChildren = this->generate_valid_children();
	int choice = rand() % numChildren;
	StateNode currState = this->children[choice];

	while (currState.p1.row != 0 && currState.p2.row != NUMROWS-1){
		numChildren = currState.generate_valid_children();
		choice = rand() % numChildren;
		currState = currState.children[choice];
	}

	//now that we have an end state check who wins and backpropagate that info
	//value of terminal state is based on how far the opponent is from winning, so the further they are from the end the better the game
	int scoreModifier;
	if (currState.p1.row == 0){
		scoreModifier = NUMROWS - currState.p2.row;
	}else{
		scoreModifier = -currState.p1.row - 1;
	}

	while (currState.parent != nullptr){
		currState.score += scoreModifier;
		currState.visits += 1;
		currState = currState.parent;
	}
	currState.score += scoreModifier;
	currState.visits += 1; //propagate to root
}

double StateNode::UCB(){
	return this->vi + 2* sqrt(math.ln(this->visits) / this->parent->visits)
}

void StateNode::update_vi(){
	std::stack<StateNode*> s;
	root = *this;
	int count=0;
	while(root != nullptr || s.size() > 0){
		if (root != nullptr){
			s.push(root);

			if (root->children.size() >= 1)
				root = root->children[0];
			else
				root = nullptr;

		}

		StateNode* curr = s.top();
		s.pop();
		//traverse 
		count += curr->vists;

		while (s.size() > 0 && curr->childrenIndex ==  
                st.top()->children.size() - 1) 
        { 
            curr = s.top(); 
            s.pop(); 
              
            //traverse
        } 
	}
}


//if a fence move, tests whether it will block either player from being able to reach the goal and doesn't add it if so
//else, adds state to the passed in state's children
bool test_and_add_move(std::vector<std::tuple<Move, int>> vmoves, StateNode* state, Move move){
	int difference = pathfinding(state, move);
	if (difference != -999){
		state->children.push_back(StateNode(state, move, difference));
		return true;
	}
	else return false;
}


StateNode::StateNode(bool turn){	
	this->turn = turn;
	this->ply = 0;
	this->visits = 0;

	this->p1.row = NUMROWS - 1;
	this->p1.col = NUMCOLS/2;
	this->p1.numFences = NUMFENCES;
	this->p2.row = 0;
	this->p2.col = NUMCOLS/2;
	this->p2.numFences = NUMFENCES;


	//only 8 possible vertical fence locations b/w the squares
	for(int i = 0; i < 2*NUMROWS-1; i+=2){
		for (int j = 0; j < NUMCOLS; j++){
			this->gamestate[i][j] = false;
		}
	}
}

//do not use this method directly, use as part of generate_valid_children after the move has been verified as valid
StateNode::StateNode(StateNode* parent, Move move, int score){
	this->p1 = parent->p1;
	this->p2 = parent->p2;


	//copies gamestate from parent, then we have to modify 
	memcpy(this->gamestate, parent->gamestate, fenceRows*NUMCOLS * sizeof(bool));

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
	this->visits = 0;
	this->ply = parent->ply + 1;
} 



std::ostream& operator<<(std::ostream &strm, const StateNode &sn) {
  return strm << (sn.turn ? "player2" : "player1") << "\t"<< (sn.move.type=='f' && sn.move.horizontal ? "h " : "v ") << sn.move.type << " -> (" << sn.move.row << "," << sn.move.col <<")\n";
  //print out opposite of what turn says, because we want the player who made the move resulting in this gamestate
}
