#pragma once 

#include "Tree.h"
#include <cstring>

int save_tree(shared_ptr<StateNode> root, std::string database_name);
shared_ptr<StateNode> load_tree(std::string database_name);

bool write_node(shared_ptr<StateNode> node, unsigned char file_buffer[], int buffer_index);
int fill_player(unsigned char ch[], int offset, Player p);
int fill_move( unsigned char ch[], int offset, Move m);
int fill_gamestate( unsigned char ch[], int offset, bool gamestate[][NUMCOLS], bool turn);

void print_buffer(unsigned char* node_buffer);
void output_tree_stats(shared_ptr<StateNode> root);