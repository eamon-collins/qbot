
#pragma once

#include "Tree.h"
#include <map>

using std::cout;
using std::endl;
using std::vector;


//returns thread_local randomness engine
std::mt19937& get_rng();


struct Pos
{
    short x,y;
    Pos operator + ( Pos p ) const { return Pos(x+p.x,y+p.y); }
    bool operator < ( Pos p ) const { return ( y==p.y ) ? (x<p.x) : (y<p.y) ; }
    bool operator != ( Pos p ) const { return ( y!=p.y ) || (x!=p.x) ; }
    Pos(short x=0,short y=0) : x(x), y(y) {}
};

enum Dir { d_beg, d_up=d_beg, d_rg, d_dn, d_lf, d_end };

Dir& operator ++ ( Dir& d );

Dir other(Dir d);

struct SearchMapItem
{
    bool traversable;
    bool goal;
    bool visited;
    int cost_here;
    Dir came_from;
    bool paths[d_end];
};

typedef std::map<Pos,SearchMapItem>::iterator SMII;

bool valid(Pos p);


int save_tree(StateNode* root);

int l1_norm(int i0, int j0, int i1, int j1);
int l1_f_p(Move move1,  Player p1);


//checks shortest path of each pawn to goal to make sure a proposed move is valid, but also returns p2pathlength-p1pathlength in move.row to help with scoring
int pathfinding(StateNode* state, Move move, vector<Move> p1Moves, vector<Move> p2Moves);
//Annoying how little redundancy b/w applying move and not, but perf was odd. Have to prio move passed in speed, as it's more common
int pathfinding(StateNode* state, Move move, bool verbose=false);
int pathfinding(StateNode* state, bool verbose=false);
int pathfinding(StateNode* state, vector<Move>& path, bool verbose=false);

//pathfinding helpers
void MakeMap(bool gamestate[][NUMCOLS], bool player1, std::map<Pos,SearchMapItem> &search_map);
//returns length of the shortest path or -1 if there is no path
int FindGoalFrom( Pos start, std::map<Pos,SearchMapItem>& search_map, std::vector<SMII>& found);
int FindGoalFrom( const Pos& start, std::map<Pos,SearchMapItem>& search_map, std::vector<Pos>& path, bool fill_path = false, bool verbose=false);

void fill_int(char ch[], int integer, int num_digits);
