	

#include "storage.h"
#include <bitset>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <stack>

//DONT MODIFY UNLESS YOU CHANGE write_node
#define BYTES_PER_NODE "56"
int bytes_per_node = 56;
int nodes_per_write = int(4096/bytes_per_node);

//saves the tree to disk
//~177 characters estimated for each statenode
int save_tree(StateNode* root, std::string database_name){
	char file_buffer[nodes_per_write * bytes_per_node];

	FILE* save_file = fopen(database_name.c_str(), "w");

	//simple, iterative, preorder depth first iteration
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

StateNode* iterative_read(std::string database_name){
	char file_buffer[nodes_per_write * bytes_per_node];
	FILE* load_file = fopen(database_name.c_str(), "r");

	//4096 / 56 = 73.1
	//this is to try to get as close to a page as possible each file read
	char node_buffer[bytes_per_node*73];
	char curr_node_buffer[bytes_per_node];
	fscanf(load_file, "%" BYTES_PER_NODE*73 "c", node_buffer);
	memcpy (curr_node_buffer, node_buffer, bytes_per_node);
	StateNode* root = new StateNode(curr_node_buffer);

	StateNode* curr = root;
	int nodes_read = 0;
	int buffer_offset = bytes_per_node; //because we just read root
	int nodes_left = 999; //placeholder, idk if we need tihs but if so this is arbitrarily high
	bool done = false;
	while (!done){

		//read the next node in the buffer
		memcpy(curr_node_buffer, &node_buffer[buffer_offset], bytes_per_node);
		StateNode* newNode = new StateNode(curr_node_buffer);
		buffer_offset += bytes_per_node;

		if(curr->serial_type == '0' || curr->serial_type == '1'){
			curr->children.push_back(newNode);
		} else if (curr->serial_type == '2') {
			curr->parent->children.push_back(newNode);
		} else if (curr->serial_type == '3'){
			//start upward iteration til find a 0
			while(curr->serial != '0'){
				curr = curr->parent;
			}
			//once more to get sibling of 0
			curr = curr->parent;
			curr->children.push_back(newNode);
		}
		//the student becomes the master
		curr = newNode;

		//if we need to read more 
		if(buffer_offset >= bytes_per_node*73){
			int eof = fscanf(load_file, "%" BYTES_PER_NODE*73 "c", node_buffer);
			buffer_offset = 0;
			if (eof != bytes_per_node*73){
				if (eof == EOF){
					nodes_left = 0;
					done = true;
				}
				else{
					nodes_left = eof / bytes_per_node;
					done = true;
				}
			}
		}
	}
	//then with the nodes left, duplicate of above code
	for (int i = 0; i < nodes_left; i++){
		memcpy(curr_node_buffer, &node_buffer[buffer_offset], bytes_per_node);
		StateNode* newNode = new StateNode(curr_node_buffer);
		buffer_offset += bytes_per_node;

		if(curr->serial_type == '0' || curr->serial_type == '1'){
			curr->children.push_back(newNode);
		} else if (curr->serial_type == '2') {
			curr->parent->children.push_back(newNode);
		} else if (curr->serial_type == '3'){
			//start upward iteration til find a 0
			while(curr->serial != '0'){
				curr = curr->parent;
			}
			//once more to get sibling of 0
			curr = curr->parent;
			curr->children.push_back(newNode);
		}
		//the student becomes the master
		curr = newNode;
	}
}

//loads tree from save file and returns pointer to root
StateNode* load_tree(std::string database_name){
	char file_buffer[nodes_per_write * bytes_per_node];
	FILE* load_file = fopen(database_name.c_str(), "r");

	char node_buffer[bytes_per_node];
	fscanf(load_file, "%c", node_buffer);
	StateNode* root = new StateNode(node_buffer);

	recursive_read(root, load_file);
}

int recursive_read(StateNode* node, FILE* load_file){
	char node_buffer[bytes_per_node];
	fscanf(load_file, "%c", node_buffer);

	StateNode newNode = StateNode(node_buffer);
	node->children.push_back(newNode);

	//if this node is marked as leaf or end of parent's children
	if (node_buffer[55] == '1'){
		return 1;
	}
	for(int i = 0; i < node->children.size(); i++){

	}
}


bool write_node(StateNode* node, char file_buffer[], int buffer_index){
	char ch[bytes_per_node];

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
	stream2 << std::fixed << std::setprecision(7) << node->vi;
	memcpy(&ch[offset], &stream.str().c_str()[2], 7);
	memcpy(&ch[offset + 6], &stream2.str().c_str()[2], 7);
	offset += 14;

	int visitsMagnitude = 7;  //number of digits of visits to save. at 7, 10million visits will rollover the visits counter
	for(int i = 0; i < visitsMagnitude; i++){
		ch[offset+i] = char(node->visits % (10^(visitsMagnitude - i))); 
	}
	offset += visitsMagnitude;

	for(int i = 0; i < 3; i++){
		ch[offset + i] = char( node->ply % (10^(3-i)));
	}
	offset += 3;

	//space left for one character denoting an end
	//of children marker, to help reconstruct tree
	if (node->parent == nullptr){
		//this is root node
		ch[offset] = '0';
	}
	//leaf AND last child in parent's vector
	else if (*(node->parent->children.begin()) == *node &&
			node->children.empty())
	{
		ch[offset] = '3';
	}
	//last child in parent's vector but not leaf
	else if (*(node->parent->children.begin()) == *node){
		ch[offset] = '1';
	}else {
		ch[offset] = '0';
	}

	offset++;

	memcpy(&file_buffer[buffer_index], ch, bytes_per_node );

	return true;
}

int fill_player(char ch[], int offset, Player p){
	ch[offset] = char(p.row);
	ch[offset+1] = char(p.col);
	ch[offset+2] = char(p.numFences);

	return offset+3;
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

	char temp[8];
	for (int i = 0; i < 20; i++){
		std::memcpy(temp, &bitstring[i*8], 8);
		std::bitset<8> bin(temp);
		ch[offset+i] = char(bin.to_ulong());
	}

	return offset+20;
}


