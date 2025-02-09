#pragma once

#include "Global.h"
#include <cstddef>
#include <vector>
#include <iostream>
#include <sstream>
#include <deque>
#include <exception>

//forward declare Game
class Game;

struct InvalidMoveException : public std::exception {
   const char * what () const throw () {
      return "Error: Invalid Move String";
   }
};

typedef struct Player{
	int row;
	int col;
	int numFences;
} Player;

//moves of type 'p' are indexed [9][9], moves of type 'f' are [17][8]
typedef struct Move{
	unsigned char type;
	int row;
	int col;
	bool horizontal; //only matters if type = 'f' ie this move places a fence

	Move(unsigned char type, int row, int col, bool horizontal){
		this->type = type;
		this->row = row;
		this->col = col;
		this->horizontal = horizontal;
	}
	Move(){
		this->type = 'u'; //default created, signifies unset, possibly root
	}
	//this constructor and unique string pair help hash/unhash
	Move(std::string s){
		sscanf(&(s.c_str()[2]), "%d %d", &row, &col);
		type = s.c_str()[0];
		if (s.c_str()[1] == '1')
			horizontal = true;
		else 
			horizontal = false;

		if (row < 0 || col < 0 )
			throw InvalidMoveException();
		if(type == 'p'){
			if (row > 8 || col > 8)
				throw InvalidMoveException();
		}else if (type == 'f'){
			if (row > 16 || col > 7)
				throw InvalidMoveException();
			if(horizontal && row % 2 == 0)
				throw InvalidMoveException();
		} else {
			throw InvalidMoveException();
		}
	}
	std::string unique_string(){
		std::ostringstream s;
		s << type << (horizontal ? "1" : "0") << row << " " << col;
		return s.str();
	}

	bool operator==(const Move& m){
		if (type != m.type || row != m.row || col != m.col || horizontal != m.horizontal)
			return false;
		return true;
	}
	bool operator!=(const Move& m){
		if (type != m.type || row != m.row || col != m.col || horizontal != m.horizontal)
			return true;
		return false;
	}

	friend std::ostream& operator<<(std::ostream& os, const Move& m);
} Move;

class StateNode
{
public:
	static Game* game;

	std::vector<StateNode> children;
	StateNode* parent = nullptr;
	
	Move move; //the move directly prior to this state
	Player p1; //player 1, at the bottom of the screen
	Player p2; //player 2, at the top of the screen
	bool turn; //true for it is currently player 1's turn, false for p2
	bool gamestate[2*NUMROWS - 1][NUMCOLS]; //stores every space in between squares where half a wall could be placed, even numbered rows have 8 0s with a -1 at the end
	double score;
	double vi; //average score of every node below this one
	int visits; //number of times we've visited this node
	int ply; //the total number of moves made up until this point.
	unsigned char serial_type; //bookkeeping for serialization/deserialization


	StateNode(bool turn); //should only be used to start a new game, all positions will be starting positions, p1 starts turn=true p2 starts turn=false
	StateNode(StateNode* parent, Move move, int score); //generate a child node, with a reference to the parent and a the move that takes gamestate from parent->child
	StateNode(unsigned char* node_buffer);

	StateNode(const StateNode &rhs) = default;
	StateNode& operator=(const StateNode &rhs) = default;
	// StateNode(const StateNode &s);

	bool operator==(const StateNode& node);


	void evaluate(); //attempts to score a specific gamestate
	int generate_valid_children(); //returns number of valid children generated
	int generate_valid_moves(std::vector<Move>& vmoves); //returns number of fence moves at end of vector
	int generate_random_child(); //generates a single child and places it in children[]. Can this be more permissive of "valid" states (ie consider all possible states?)
	bool is_valid_move(Move move); //checks whether the resulting gamestate from a certain move is legal or not
	int get_best_move();
	bool good_shrub(); //attempts to pick gamestates to be pruned as a heuristic
	int prune_children(); //uses me-defined heuristics to prune children of this node
	StateNode* play_out(); //simulates by random choice to determine a winner from this state
	double UCB() const; //simple 
	void update_vi(); //calculates average score for every node under this one.
	void fix_parent_references(); //changes all parent references in subtree below this node to correct pointers
	// void StateNode::set_game_pointer(const Game* game);
	std::string visualize_gamestate();
	int game_over() const; //tests whether the game is over at this state or not. 0 for no, 1 for p1 win, 2 for p2 win


	void print_node(); //printfs a representation of the gamestate at this node
};

std::ostream& operator<<(std::ostream &strm, const StateNode &sn); //print override for StateNode
bool test_and_add_move( StateNode* state, Move move); //helps testing validity and also beginning scoring before state fully initialized.
void best_move_worker(int id, StateNode* root);


 
#if PY_VERSION_HEX >= 0x030800f0
static inline void
py3__Py_DECREF(const char *filename, int lineno, PyObject *op)
{
    (void)filename; /* may be unused, shut up -Wunused-parameter */
    (void)lineno; /* may be unused, shut up -Wunused-parameter */
    _Py_DEC_REFTOTAL;
    if (--op->ob_refcnt != 0)
    {
#ifdef Py_REF_DEBUG
	if (op->ob_refcnt < 0)
	{
	    _Py_NegativeRefcount(filename, lineno, op);
	}
#endif
    }
    else
    {
	_Py_Dealloc(op);
    }
}

#undef Py_DECREF
#define Py_DECREF(op) py3__Py_DECREF(__FILE__, __LINE__, _PyObject_CAST(op))

    static inline void
py3__Py_XDECREF(PyObject *op)
{
    if (op != NULL)
    {
	Py_DECREF(op);
    }
}

#undef Py_XDECREF
#define Py_XDECREF(op) py3__Py_XDECREF(_PyObject_CAST(op))
#endif
