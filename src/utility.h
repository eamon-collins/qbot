
#pragma once

#include "Tree.h"
#include "stlastar.h"
#include <map>

class SearchNode
{
public:
    int row;
    int col;

    SearchNode();
    SearchNode(int row, int col);

    float GoalDistanceEstimate( SearchNode &nodeGoal );
    bool IsGoal( SearchNode &nodeGoal );
    bool GetSuccessors( AStarSearch<SearchNode> *astarsearch );
    float GetCost( SearchNode *successor );
    bool IsSameState( SearchNode &rhs );

};

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
    bool traversble;
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
int pathfinding(StateNode* state, Move move);

//pathfinding helpers
void MakeMap(bool gamestate[][NUMCOLS], bool player1, std::map<Pos,SearchMapItem> &search_map);
//returns length of the shortest path or -1 if there is no path
int FindGoalFrom( Pos start, std::map<Pos,SearchMapItem>& search_map, std::vector<SMII>& found);

void fill_int(char ch[], int integer, int num_digits);