#include "Tree.h"


int save_tree(StateNode* root);

bool write_node(StateNode* node, FILE f );
int fill_player(char ch[], int offset, Player p);
int fill_move(char ch[], int offset, Move m);
int fill_gamestate(char ch[], int offset, bool gamestate[][NUMCOLS], bool turn);