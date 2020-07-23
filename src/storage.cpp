

#include "storage.h"
#include <bitset>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <stack>

//DONT MODIFY UNLESS YOU CHANGE write_node
int bytes_per_node = 56;


//saves the tree to disk
//~177 characters estimated for each statenode
int save_tree(StateNode* root, std::string database_name){
	//simple depth first iteration
	int nodes_per_write = int(4096/bytes_per_node);
	char file_buffer[nodes_per_write * bytes_per_node];

	FILE* save_file;
	save_file = fopen(database_name.c_str(), "w");


	std::stack<StateNode*> tree_stack;
	tree_stack.push(root);
	StateNode* curr;
	int nodes_written = 0;
	while (!tree_stack.empty()){
		curr = tree_stack.top();
		tree_stack.pop();
		write_node(curr, file_buffer, (nodes_written % nodes_per_write) * bytes_per_node);
		nodes_written++;
		if ( (nodes_written % nodes_per_write) == 0){
			fwrite(file_buffer, 1, nodes_per_write * bytes_per_node, save_file);
		}

		std::vector<StateNode>::iterator it;
		for (it = curr->children.begin(); it < curr->children.end(); it++) {
			tree_stack.push(&(*it));
		}
	}

	fclose(save_file);

	return nodes_written;
}

bool write_node(StateNode* node, char file_buffer[], int buffer_index){
	char ch[178];

	int offset = 0; 
	offset = fill_move(ch, offset, node->move);
	offset = fill_player(ch, offset, node->p1);
	offset = fill_player(ch, offset, node->p2);
	offset = fill_gamestate(ch, offset, node->gamestate, node->turn);

	//score assumed to be normalized to 0-1 value
	//could afford one extra byte to get to 56, could switch which one is more valuable 
	//Also potentially stream.str.cstr is area for performance improvement
	std::stringstream stream, stream2;
	stream << std::fixed << std::setprecision(7) << node->score;
	stream2 << std::fixed << std::setprecision(6) << node->vi;
	memcpy(&ch[offset], &stream.str().c_str()[2], 7);
	memcpy(&ch[offset + 6], &stream2.str().c_str()[2], 6);
	offset += 13;

	int visitsMagnitude = 7;  //number of digits of visits to save. at 7, 10million visits will rollover the visits counter
	for(int i = 0; i < visitsMagnitude; i++){
		ch[offset+i] = char(node->visits % (10^(visitsMagnitude - i))); 
	}
	offset += visitsMagnitude;

	for(int i = 0; i < 3; i++){
		ch[offset + i] = char( node->ply % (10^(3-i)));
	}
	offset += 3;

	memcpy(&file_buffer[buffer_index], ch, bytes_per_node );
}


int fill_player(char ch[], int offset, Player p){
	ch[offset] = char(p.row);
	ch[offset+1] = char(p.col);
	ch[offset+2] = char(p.row);
	ch[offset+3] = char(p.numFences);

	return offset+4;
}

int fill_move(char ch[], int offset, Move m){
	ch[offset] = m.type;
	if (m.row >= 10){
		ch[offset+1] = char(1);
		ch[offset+2] = char(m.row % 10);
	}else{
		ch[offset+1] = char(0);
		ch[offset+2] = char(m.row);
	}

	ch[offset+3] = char(m.col);
	ch[offset+4] = char(m.horizontal ? '1' : '0');

	return offset+5;

}

//also takes responsibility for turn as it fits in the extra space 
//might need to use unsigned chars
int fill_gamestate(char ch[], int offset, bool gamestate[][NUMCOLS], bool turn){
	char bitstring[160];
	int index = 0;
	for (int i = 0; i < 2*NUMROWS-1; i++){
		for(int j = 0; j < NUMCOLS; j++){
			bitstring[index] = gamestate[i][j] ? '1' : '0';
			index++;
		}
	}
	bitstring[index] = turn ? '1' : '0';
	//pad last 6 bits out as 0
	for (int i = 0; i < 6; i++){
		bitstring[index+1+i] = '0';
	}

	index = 0;
	char temp[8];
	for (int i = 0; i < 20; i++){
		std::memcpy(temp, &bitstring[i*8], 8);
		std::bitset<8> bin(temp);
		ch[offset+i] = char(bin.to_ulong());
	}

	return offset+20;
}


