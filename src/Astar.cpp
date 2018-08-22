

#include "stlastar.h"
#include <cstdlib>



class SearchNode
{
public:
	int row;
	int col;


	float GoalDistanceEstimate( SearchNode &nodeGoal );
	bool IsGoal( SearchNode &nodeGoal );
	bool GetSuccessors( AStarSearch<SearchNode> *astarsearch );
	float GetCost( SearchNode *successor );
	bool IsSameState( SearchNode &rhs );

};


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