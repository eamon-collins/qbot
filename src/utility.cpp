//Eamon Collins 
//A* pathfinding implementation and distance
#include "utility.h"
#include <cstdlib>
#include <vector>
#include <queue>
#include <iostream>
#include <algorithm>
#include <string>
#include <cstring>
#include <limits>


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

void print_map(std::map<Pos, SearchMapItem> map, Player p1, Player p2){
	Pos p;
	for (int r = 2*NUMROWS-2; r >= 0; r--) {
		for (int c = 0; c < 2*NUMCOLS-1; c++) {
			p.y = r;
			p.x = c;
			if (2*p1.row == r && 2*p1.col == c) {
				cout << "p";
			} else if (2*p2.row == r && 2*p2.col == c) {
				cout << "q";
			} else if (map[p].traversable) {
				cout << (map[p].goal ? "G" : "O");
			} else {
				cout << " ";
			}
		}
		cout << "\n";
	}
}

// Finds and returns the full path to the goal for just one player.
// Should be used more sparingly than other pathfinding
// TODO: account for other player blocking movement
int pathfinding(StateNode* state, vector<Move>& path, bool verbose) {
	Player p1 = state->turn ? state->p1 : state->p2;
	Player p2 = !state->turn ? state->p1 : state->p2;

	std::map<Pos, SearchMapItem> search_map;
	std::vector<SMII> found;
	std::vector<Pos> pos_path;

	MakeMap(state->gamestate, false, search_map); //fills search_map for player1
	if (verbose) {
		print_map(search_map, p1, p2); 
	}
	int pathLength = FindGoalFrom(Pos(2*(p1.col), 2*(p1.row)), search_map, pos_path, true, verbose);
	if (pathLength == -1){ //
		return -999;
	}

	if (!path.empty()) { path.clear(); }
	for ( Pos& p : pos_path) {
		if ( p.x % 2 == 0 && p.y % 2 == 0)
			path.push_back( Move('p', p.y / 2, p.x / 2, false) );
	}
	if (verbose) {
		for (Move& m : path) {
			std::cout << "(" << m.col << ","<<m.row<<") -> ";
		}
	}


	return pathLength;
}

// Checks distance to goal for each player and returns dist_p1 - dist_p2.
// returns -999 if either player has no path to goal.
// does not consider other player blocking at all
int pathfinding(StateNode* state, bool verbose){
	return pathfinding(state, state->move, verbose);
}

int pathfinding(StateNode* state, Move move, bool verbose){
	//copy over the state in question's gamestate so we don't affect the state itself
	static bool gamestate[2*NUMROWS - 1][NUMCOLS];
	memcpy(gamestate, state->gamestate, (2*NUMROWS-1)*NUMCOLS*sizeof(bool));

	
	// Player p1 = state->turn ? state->p1 : state->p2;
	// Player p2 = !state->turn ? state->p1 : state->p2;
	Player p1 = state->p1;
	Player p2 = state->p2;
	if (move != state->move) {
		//apply the proposed move
		if (move.type == 'f') {
			if (move.horizontal){
				gamestate[move.row][move.col] = true;
				gamestate[move.row][move.col+1] = true;
			}else{
				gamestate[move.row][move.col] = true;
				gamestate[move.row+2][move.col] = true;
			}
		} else if (move.type == 'p'){
			p1.row = move.row;
			p1.col = move.col;
		}
	}

	std::map<Pos, SearchMapItem> search_map;
	std::vector<SMII> found;
	// static because we aren't actually filling it for this pathfind
	static vector<Pos> path;

	MakeMap(gamestate, false, search_map); //fills search_map for player1
	if (verbose) {
		print_map(search_map, p1, p2); 
	}
	int pathLength = FindGoalFrom(Pos(2*(p1.col), 2*(p1.row)), search_map, path, false, verbose);
	if (pathLength == -1){ //
		return -999;
	}

	//all that needs to be different for player 2 is switching the goal, then we can reuse the search_map
	for(int i = 0; i < 2*NUMCOLS-1; i+=2){
		search_map[Pos(i,0)].goal = true;
		search_map[Pos(i,2*NUMROWS-2)].goal = false;
	}

	int p2pathLength = FindGoalFrom(Pos(2*(p2.col), 2*(p2.row)), search_map, path, false, verbose);
	if (p2pathLength == -1) { 
		return -999;
	}

	if (verbose) {
		print_map(search_map, p1, p2); 
		cout << "P1 distance: " << pathLength << std::endl;
		cout << "P2 distance: " << p2pathLength << std::endl;
	}

	return p2pathLength/2 - pathLength/2;
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
	            smi.traversable = false;
	        }
	        else if( Map[p.y][p.x] == GOAL )
	        {
	            smi.traversable = true;
	            smi.goal = true;
	        }
	        else if( Map[p.y][p.x] == FLOOR )
	        {
	            smi.traversable = true;
	            smi.goal = false;
	            for( Dir d = d_beg; d != d_end; ++d )
	            {
	                Pos p2 = p + deltas[d];
	                smi.paths[d] = valid(p2) && (Map[p2.y][p2.x] != WALL) ;
	            }
	        }
	        search_map[p] = smi;
    	}
    }


}

int heuristic(const Pos &current, const int goal_y) {
    return std::abs(current.y - goal_y);
}

int FindGoalFrom(const Pos &start, std::map<Pos, SearchMapItem> &search_map, std::vector<Pos>& path, bool fill_path, bool verbose) {
    // only really need a single y coord to inform heuristic
	int goal_y = search_map[Pos(0,0)].goal ? 0 : (2*NUMROWS-2);

    // Priority queue for A* search with f-score
    std::priority_queue<std::pair<int, Pos>, std::vector<std::pair<int, Pos>>, std::greater<>> pq;
    std::map<Pos, int> g_cost;
	std::map<Pos, Pos> came_from; // stores path

    // Initialize the search
    pq.push({heuristic(start, goal_y), start});
    g_cost[start] = 0;

    while (!pq.empty()) {
        auto [f_score, current] = pq.top();
        pq.pop();

        // Check if the current position is a goal
        if (search_map[current].goal) {
			if(verbose) {
				cout << "start: "<<start.x<<","<<start.y<<" end: "<<current.x<<","<<current.y<<std::endl;
			}
			if (fill_path) {
				for (Pos p = current; p != start; p = came_from[p]) {
					path.push_back(p);
				}
				path.push_back(start);
				std::reverse(path.begin(), path.end());
			}
            return g_cost[current];  // Return shortest path to the nearest goal
        }

        // Explore neighboring positions
        for (const auto &dir : deltas) {
            Pos neighbor = Pos(current.x + dir.x, current.y + dir.y);

            // Check if the position is in the map and is traversable
            if (search_map.find(neighbor) == search_map.end() || !search_map[neighbor].traversable) {
                continue;
            }

            int new_cost = g_cost[current] + 1;  // Each move costs 1

            if (g_cost.find(neighbor) == g_cost.end() || new_cost < g_cost[neighbor]) {
                g_cost[neighbor] = new_cost;
				if (fill_path) {
					came_from[neighbor] = current;
				}
                int f_score = new_cost + heuristic(neighbor, goal_y);
                pq.push({f_score, neighbor});
            }
        }
    }

    // No path to any goal found
	return -1;
}

//places path in found
int FindGoalFrom( Pos start , std::map<Pos,SearchMapItem> &search_map, std::vector<SMII>& found)
{

    //std::vector<SMII> found;

    {
        SMII smii = search_map.find(start);

        if(smii==search_map.end()) { std::cout << "starting outside map\n"; 
        return -1; }
        if(smii->second.goal) {return 0; }
        if(!smii->second.traversable) { std::cout << "starting in a wall\n"; return -1; }

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
