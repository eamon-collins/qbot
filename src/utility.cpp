

#include "utility.h"
#include <cstdlib>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cstring>


// L_1 norm (manhattan distance)
int l1_norm(int i0, int j0, int i1, int j1) {
  return std::abs(i0 - i1) + std::abs(j0 - j1);
}
//l1 for finding how far fence is from pawn
int l1_f_p(Move move1,  Player player) {
	if (move1.type == 'f')
 		return std::abs(move1.row / 2 - player.row) + std::abs(move1.col - player.col);
 	else
 		return std::abs(move1.row - player.row) + std::abs(move1.col - player.col);
}

//if the move is a fence move, we are considering this move.
//if not, basically finding the necessary moves of the passed in state
int pathfinding(StateNode* state, Move move, vector<Move> p1Moves, vector<Move> p2Moves){

	// int arr[3] = {-2, 1, 2};

	// if(true)
	// 	return arr[rand() % 3];


	
	//heuristics to try and avoid needing to do pathfinding as much as possible //TAKEN OUT BECAUSE I NEED TO HAVE PATHLENGTH OF ALL VALID NODES, ONLY DISQUALIFYING HEURISTICS ALLOWED
	// int fencesUsed = 2*NUMFENCES - state->p1.numFences - state->p2.numFences;
	// if(fencesUsed < 3 || ((move.type == 'p') && fencesUsed < 4))
	// 	return true;


	bool gamestate[2*NUMROWS - 1][NUMCOLS];
	//copy over the state in question's gamestate so we don't affect the state itself
	memcpy(gamestate, state->gamestate, (2*NUMROWS-1)*NUMCOLS*sizeof(bool));

	//apply the proposed move
	if (move.type == 'f'){
		if (move.horizontal){
			gamestate[move.row][move.col] = true;
			gamestate[move.row][move.col+1] = true;
		}else{
			gamestate[move.row][move.col] = true;
			//gamestate[move.row+1][move.col] = true;
			gamestate[move.row+2][move.col] = true;
		}
	}


	std::map<Pos, SearchMapItem> search_map;
	std::vector<SMII> found;

	MakeMap(gamestate, true, search_map); //fills search_map for player1
	int pathLength = FindGoalFrom(Pos(2*(state->p1.row), 2*(state->p1.col)), search_map, found);
	if (pathLength == -1){ //
		return -999;
	}else if((state->p1.numFences == 0 && state->p2.numFences == 0) && state->turn && !found.empty()){
		Move pawn_move = Move();
		pawn_move.type = 'p';
		for (SMII& square : found){
			pawn_move.row = found[0]->first.y / 2;
			pawn_move.col = found[0]->first.x / 2;
			p1Moves.push_back(pawn_move);
		}
	}

	//all that needs to be different for player 2 is the bottom row is the goal, then we can reuse the search_map
	for(int i = 0; i < 2*NUMCOLS-1; i++){
		search_map[Pos(0,i)].goal = false;
		search_map[Pos(2*NUMROWS-1,i)].goal = true;
	}
	found.clear();

	//MakeMap(gamestate, false, search_map); //for player2 
	int p2pathLength = FindGoalFrom(Pos(2*(state->p2.row), 2*(state->p2.col)), search_map, found);
	if (p2pathLength == -1){ 
		return -999;
	}else if((state->p1.numFences == 0 && state->p2.numFences == 0) && !state->turn && !found.empty()){ //player2 turn
		Move pawn_move = Move();
		pawn_move.type = 'p';
		for (SMII& square : found){
			pawn_move.row = found[0]->first.y / 2;
			pawn_move.col = found[0]->first.x / 2;
			p2Moves.push_back(pawn_move);
		}
	}

	int score = p2pathLength/2 - pathLength/2;

	return score;
}

int pathfinding(StateNode* state, Move move){
	
	//heuristics to try and avoid needing to do pathfinding as much as possible //TAKEN OUT BECAUSE I NEED TO HAVE PATHLENGTH OF ALL VALID NODES, ONLY DISQUALIFYING HEURISTICS ALLOWED
	// int fencesUsed = 2*NUMFENCES - state->p1.numFences - state->p2.numFences;
	// if(fencesUsed < 3 || ((move.type == 'p') && fencesUsed < 4))
	// 	return true;


	bool gamestate[2*NUMROWS - 1][NUMCOLS];
	//copy over the state in question's gamestate so we don't affect the state itself
	memcpy(gamestate, state->gamestate, (2*NUMROWS-1)*NUMCOLS*sizeof(bool));

	//apply the proposed move
	if (move.type == 'f'){
		if (move.horizontal){
			gamestate[move.row][move.col] = true;
			gamestate[move.row][move.col+1] = true;
		}else{
			gamestate[move.row][move.col] = true;
			//gamestate[move.row+1][move.col] = true;
			gamestate[move.row+2][move.col] = true;
		}
	}


	std::map<Pos, SearchMapItem> search_map;
	std::vector<SMII> found;

	MakeMap(gamestate, true, search_map); //fills search_map for player1
	int pathLength = FindGoalFrom(Pos(2*(state->p1.row), 2*(state->p1.col)), search_map, found);
	if (pathLength == -1){ //
		return -999;
	}

	//all that needs to be different for player 2 is the bottom row is the goal, then we can reuse the search_map
	for(int i = 0; i < 2*NUMCOLS-1; i++){
		search_map[Pos(0,i)].goal = false;
		search_map[Pos(2*NUMROWS-1,i)].goal = true;
	}
	found.clear();

	//MakeMap(gamestate, false, search_map); //for player2 
	int p2pathLength = FindGoalFrom(Pos(2*(state->p2.row), 2*(state->p2.col)), search_map, found);
	if (p2pathLength == -1){ 
		return -999;
	}

	int score = p2pathLength/2 - pathLength/2;

	return score;
}

int pathfinding2(StateNode* state, Move move){
	
	//heuristics to try and avoid needing to do pathfinding as much as possible //TAKEN OUT BECAUSE I NEED TO HAVE PATHLENGTH OF ALL VALID NODES, ONLY DISQUALIFYING HEURISTICS ALLOWED
	// int fencesUsed = 2*NUMFENCES - state->p1.numFences - state->p2.numFences;
	// if(fencesUsed < 3 || ((move.type == 'p') && fencesUsed < 4))
	// 	return true;

	AStarSearch<SearchNode> astarsearch;

	SearchNode startNode1 = SearchNode(state->p1.row, state->p1.col);
	SearchNode startNode2 = SearchNode(state->p2.row, state->p2.col);

	//copy over the state in question's gamestate so we don't affect the state itself
	memcpy(astarsearch.gamestate, state->gamestate, (2*NUMROWS-1)*NUMCOLS*sizeof(bool));

	//apply the proposed move
	if(move.type == 'f'){
		if (move.horizontal){
			astarsearch.gamestate[move.row][move.col] = true;
			astarsearch.gamestate[move.row][move.col+1] = true;
		}else{
			astarsearch.gamestate[move.row][move.col] = true;
			//gamestate[move.row+1][move.col] = true;
			astarsearch.gamestate[move.row+2][move.col] = true;
		}
	}
	else{
		if (state->turn){
			startNode1.row = move.row;
			startNode1.col = move.col;
		}else{
			startNode2.row = move.row;
			startNode2.col = move.col;
		}
	}

	SearchNode goalNode;
	//search player1's path, can't exit early though
	for(int i = 0; i < NUMCOLS; i++){
	}

	return -1;
}
SearchNode::SearchNode(){}
SearchNode::SearchNode(int row, int col){
	this->row = row;
	this->col = col;
}

float SearchNode::GoalDistanceEstimate( SearchNode &nodeGoal ){
	return std::abs(nodeGoal.row - this->row) + std::abs(nodeGoal.col - this->col);
}

bool SearchNode::IsGoal( SearchNode &nodeGoal ){
	return IsSameState( nodeGoal );
}
bool SearchNode::IsSameState(SearchNode &rhs){
	if (rhs.row == this->row && rhs.col == this->col)
		return true;
	return false;
}
bool GetSuccessors( AStarSearch<SearchNode> *astarsearch ){
	return false;
}
float GetCost( SearchNode *successor ){
	return -1.0;
}

const char FLOOR = '1' ;
const char WALL  = '0' ;
const char GOAL  = 'G' ;
//       Y pos      X pos
// char Map[MapHeight][MapWidth] = {
//     { '1', '1', 'G', '1', '1', '1', '1', '1', '1', '1' },
//     { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
//     { '1', '1', '0', '1', '1', '1', '1', '1', '0', 'G' },
//     { '1', '1', '0', '0', '1', '1', '1', '1', '0', '1' },
//     { '1', '1', '1', '1', '1', '1', '1', '1', '1', '1' },
//     { '1', '1', '1', '1', '1', '1', '1', '0', '0', '1' },
//     { '1', '1', '1', '1', '1', '1', '1', '0', '1', '1' },
//     { '1', '1', '1', '0', '0', '1', '1', '0', '1', '1' },
//     { '1', '1', '0', '0', '0', '1', '1', '0', '0', '0' },
//     { '1', '1', '1', 'G', '0', '1', '1', '1', '1', '1' }
// };

bool valid(Pos p) { int MapWidth = 2*NUMCOLS-1, MapHeight = 2*NUMROWS-1;
	return p.x>=0 && p.x<MapWidth && p.y>=0 && p.y<MapHeight; }
Pos deltas[d_end] = { {0,-1}, {+1,0}, {0,+1}, {-1,0} };

Dir& operator ++ ( Dir& d ) { d = (Dir) ( 1+(int)d ) ; return d; }

Dir other(Dir d)
{
    switch(d)
    {
    case d_up: return d_dn;
    case d_rg: return d_lf;
    case d_dn: return d_up;
    case d_lf: return d_rg;
    default: return d_end;
    }
}



void MakeMap(bool gamestate[][NUMCOLS], bool player1, std::map<Pos,SearchMapItem> &search_map)
{
	int MapWidth = 2*NUMCOLS-1, MapHeight = 2*NUMROWS-1;
	//construct Map out of gamestate by adding floor tiles between fence indices
	char Map[MapHeight][MapWidth];
	for (int i = 0; i < 2*NUMROWS -1; i++){
		for (int j = 0; j < 2*NUMCOLS -1; j++){
			if(i % 2 == 0 && j % 2 == 0)
				Map[i][j] = FLOOR;
			else if (i % 2 == 1 && j % 2 == 1)
				Map[i][j] = WALL;
		}
	}
	for(int i = 0; i < 2*NUMROWS-1; i++){
		for (int j = 0; j < NUMCOLS; j++){
			if(i % 2 == 0){
				if (j == 8)
					Map[i][2*j] = FLOOR;
				else if (gamestate[i][j])
					Map[i][2*j + 1] = WALL;
				else
					Map[i][2*j + 1] = FLOOR;
			}else{
				if(gamestate[i][j])
					Map[i][2*j] = WALL;
				else
					Map[i][2*j] = FLOOR;
			}
		}
	}

	//add the goal dependent on player we made map for
	if(player1){
		for (int j = 0; j < MapWidth; j+=2){
			Map[0][j] = GOAL;
		}
	}else{
		for (int j = 0; j < MapWidth; j+=2){
			Map[MapHeight-1][j] = GOAL;
		}
	}

	search_map.clear();
    Pos p;
    for(p.y=0;p.y<MapHeight;++p.y){
    	for(p.x=0;p.x<MapWidth;++p.x) 
    	{
	    	//cout << Map[p.y][p.x];
	        SearchMapItem smi;
	        smi.visited = false;
	        smi.cost_here = -1;
	        smi.came_from = d_end;
	        if( Map[p.y][p.x] == WALL )
	        {
	            smi.traversble = false;
	        }
	        else if( Map[p.y][p.x] == GOAL )
	        {
	            smi.traversble = true;
	            smi.goal = true;
	        }
	        else if( Map[p.y][p.x] == FLOOR )
	        {
	            smi.traversble = true;
	            smi.goal = false;
	            for( Dir d = d_beg; d != d_end; ++d )
	            {
	                Pos p2 = p + deltas[d];
	                smi.paths[d] = valid(p2) && (Map[p2.y][p2.x] != WALL) ;
	            }
	        }
	        search_map[p] = smi;
    	}
    	//cout << "\n";
    }


}

//places path in found
int FindGoalFrom( Pos start , std::map<Pos,SearchMapItem> &search_map, std::vector<SMII>& found)
{

    //std::vector<SMII> found;

    {
        SMII smii = search_map.find(start);

        if(smii==search_map.end()) { std::cout << "starting outside map\n"; 
        return false; }
        if(smii->second.goal) {return true; }
        if(!smii->second.traversble) { std::cout << "starting in a wall\n"; return false; }

        smii->second.visited = true;
        smii->second.cost_here = 0;
        found.push_back(smii);
    }

	int cost_so_far = 0;	
	bool did_find = false;

    while(!did_find)
    {

        std::vector<SMII> candidates;

        for( SMII smii : found )
        {
            for( Dir d = d_beg; d != d_end; ++d )
            {
                if( ! smii->second.paths[d] ) continue;
                Pos p = smii->first + deltas[d];
                if(!valid(p)) continue;
                SMII cand = search_map.find(p);
                if(cand==search_map.end()) continue;
                if(cand->second.visited) continue;
                cand->second.came_from=d;
                candidates.push_back(cand);
            }
        }

        ++cost_so_far;

        if( candidates.empty() ) break;

        for( SMII smii : candidates )
        {
            smii->second.visited = true;
            smii->second.cost_here = cost_so_far;
            found.push_back(smii);
            if( smii->second.goal ) { did_find = true; break; }
        }

    }

	if( ! did_find ) 
		return -1; 
	else 
		return cost_so_far;
}

void fill_int(char ch[], int integer, int num_digits){
	for (int i = num_digits-1; i >= 0; i--){
		
	}
}