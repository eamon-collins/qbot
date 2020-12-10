

#include "storage.h"
#include <bitset>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <stack>

//DONT MODIFY UNLESS YOU CHANGE write_node
#define BYTES_PER_READ "4088"  //56*73
#define S_(x)
#define S(x) S_(x) 
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
			fwrite((const void*)file_buffer, bytes_per_node, (size_t)nodes_per_write, save_file);
		}

		std::list<StateNode>::iterator it;
		for (it = curr->children.begin(); it != curr->children.end(); it++) {
			tree_stack.push(&(*it));
		}
	}
	fwrite((const void*)file_buffer, bytes_per_node, (size_t)(nodes_written % nodes_per_write), save_file);


	fclose(save_file);

	return nodes_written;
}

StateNode* load_tree(std::string database_name){
	char file_buffer[nodes_per_write * bytes_per_node];
	FILE* load_file = fopen(database_name.c_str(), "r");

	//4096 / 56 = 73.1
	//this is to try to get as close to a page as possible each file read
	//WILL NEED TO PAD THIS TO EXACT PAGE
	// char node_buffer[bytes_per_node*nodes_per_write];
	// char curr_node_buffer[bytes_per_node];
	char* node_buffer = (char*)malloc((bytes_per_node+1)*nodes_per_write);
	char* curr_node_buffer = (char*)malloc(bytes_per_node+1);
	// memset(node_buffer, '\0', (bytes_per_node+1)*nodes_per_write);
	// memset(curr_node_buffer, '\0', bytes_per_node+1);
	//fscanf(load_file, "%" S(BYTES_PER_READ) "c", node_buffer);
	std::cout << fscanf(load_file, "%4088s", node_buffer) << "\n";
	std::cout <<"LOAD:\n" << node_buffer << "\n";
	memcpy(curr_node_buffer, node_buffer, bytes_per_node);
	StateNode* root = new StateNode(curr_node_buffer);
	std::cout << "WE IN HERE";
	// if(DEBUG){
	// 	std::cout << "Root node:\n";
	// 	std::cout << "turn " << root->turn << "\n";
	// 	std::cout << "score " << root->score << "\n";
	// 	std::cout << "vi " << root->vi << "\n";
	// 	std::cout << "serial_type " << root->serial_type << "\n";

	// 	std::cout << curr_node_buffer << "\n";
	// }

	StateNode* curr = root;
	int nodes_read = 0;
	int buffer_offset = bytes_per_node; //because we just read root
	int nodes_left = 999; //placeholder, idk if we need tihs but if so this is arbitrarily high
	bool done = false;
	while (!done){

		//read the next node in the buffer
		memcpy(curr_node_buffer, &node_buffer[buffer_offset], (size_t)bytes_per_node);
		
		StateNode* newNode = new StateNode(curr_node_buffer);
		buffer_offset += bytes_per_node;
		//std::cout << curr->serial_type;

		if(curr->serial_type == '0' || curr->serial_type == '1'){
						std::cout << "ZERO\n";
			newNode->parent = curr;
			curr->children.push_back(newNode);
		} else if (curr->serial_type == '2') {
			newNode->parent = curr->parent;
			curr->parent->children.push_back(newNode);
		} else if (curr->serial_type == '3'){
			//start upward iteration til find a 0
			std::cout << "THREE\n";
			while(curr->serial_type != '0'){
				curr = curr->parent;
			}
			//once more to get sibling of 0
			curr = curr->parent;
			newNode->parent = curr;
			curr->children.push_back(newNode);
		}else{
			std::cout << "UNEXPECTED SERIAL TYPE: " << curr->serial_type << "\n";
		}
		//the student becomes the master
		curr = newNode;

		//if we need to read more nodes
		//this is currently optimized to read 4096/56=73.1 -> 73 nodes at a time to get pages of 4096.
		//however, it may be best to align memory on each page, because after first it will be 2 pages per fscanf
		//unless it is cleanly on memory barrier
		if(buffer_offset >= bytes_per_node*73){
			int eof = fscanf(load_file, "%4088s", node_buffer);
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
			while(curr->serial_type != '0'){
				curr = curr->parent;
			}
			//once more to get sibling of 0
			curr = curr->parent;
			curr->children.push_back(newNode);
		}
		//the student becomes the master
		curr = newNode;
	}
	std::cout << "root children: " << root->children.size() << "\n";

	free (node_buffer);
	free(curr_node_buffer);
	return root;

}



bool write_node(StateNode* node, char file_buffer[], int buffer_index){
	char ch[bytes_per_node];

	int offset = 0; 
	offset = fill_move(ch, offset, node->move);
	offset = fill_player(ch, offset, node->p1);
	offset = fill_player(ch, offset, node->p2);
	if (node->parent == nullptr) std::cout << offset << " ";
	offset = fill_gamestate(ch, offset, node->gamestate, node->turn);

	//this is root
	if(node->parent == nullptr){
		std::cout << offset << "OFFSET AT GAMESTATE\n";
		for (int i = 0; i < 31; i++){
			if(ch[i] == '\0') std::cout <<'0';
			else std::cout << ch[i];
		}
		//std::cout << "\n" << ch[1];
	}

	//score assumed to be normalized to 0-.99 value
	//could afford one extra byte to get to 56, could switch which one is more valuable 
	//Also potentially stream.str.cstr is area for performance improvement
	std::stringstream stream, stream2, stream3, stream4;
	stream << std::fixed << std::setprecision(7) << node->score;
	stream2 << std::fixed << std::setprecision(7) << node->vi;
	stream3 << std::fixed << std::setprecision(7) << node->visits;
	//stream4 << std::fixed << std::setprecision(3) << node->ply;
	
	if (node->parent==nullptr) std::cout << stream.rdbuf() << " " << stream2.rdbuf() << " " << stream3.rdbuf() << "\n";
	memcpy(&ch[offset], &stream.str().c_str()[2], 8);
	memcpy(&ch[offset + 7], &stream2.str().c_str()[2], 7);
	//memcpy(&ch[offset + 14], &stream3.str().c_str()[0], 7);
	//memcpy(&ch[offset+21], &stream4.str().c_str()[0], 3);
	offset += 14;

	// int visitsMagnitude = 7;  //number of digits of visits to save. at 7, 10million visits will rollover the visits counter
	// snprintf(&ch[offset], visitsMagnitude+1, "%d", node->visits);
	//int tempvisits = node->visits;
	// for(int i = 0; i < visitsMagnitude; i++){
	// 	snprintf(&ch[offset+i], 2, "%d" tempvisits % (10^(visitsMagnitude - i)); 
	// 	tempvisits = tempvisits / 10;
	// }
	//offset += visitsMagnitude;

	//the snprintf+1 is 3+1 here
	snprintf(&ch[offset], 8, "%07d", node->visits);
	offset += 7;
	snprintf(&ch[offset], 4, "%03d", node->ply);
	// for(int i = 0; i < 3; i++){
	// 	ch[offset + i] = char( node->ply % (10^(3-i)));
	// }
	offset += 3;

	//ch[offset] = '\0';
	//std::cout << ch << '\n';
	//std::cout << &ch[31] << '\n';

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
		std::cout << "THREE\n";

	}
	//last child in parent's vector but not leaf
	else if (*(node->parent->children.begin()) == *node){
		ch[offset] = '1';
	}else {
		ch[offset] = '0';
	}

	offset++;

	memcpy(&file_buffer[buffer_index], ch, bytes_per_node );

	//std::cout << ch << "\n";
	return true;
}

//NOTE:::snprintf strings are maximum n-1 chars long because the last char is always \0
// so the second arg will always be 1 more than the chars I actually want from that 
int fill_player(char ch[], int offset, Player p){
	/*ch[offset] = char(p.row);
	ch[offset+1] = char(p.col);
	ch[offset+2] = char(p.numFences);*/

	//TODO maybe try std::to_string(p.row).c_str() if speed is a problem
	//see note about snprintf above
	snprintf(&ch[offset], 2, "%1d", p.row);
	snprintf(&ch[offset+1], 2, "%1d", p.col);
	//forgot each player started with 10 fences, need to allocate another byte perhaps? if speeds are slow
	if(p.numFences == 10){
		ch[offset+2] = 't';
	}else{
		snprintf(&ch[offset+2], 2, "%d", p.numFences);
	}
	return offset+3;
}

int fill_move(char ch[], int offset, Move m){
	ch[offset] = m.type;
	if (m.row >= 10){
		ch[offset+1] = '1';
		//ch[offset+2] = char(m.row % 10);
		snprintf(&ch[offset+2], 2, "%d", m.row % 10);
	}else{
		ch[offset+1] = '0';
		snprintf(&ch[offset+2], 2, "%d", m.row);
	}

	snprintf(&ch[offset+3], 2, "%d", m.col);
	ch[offset+4] = (m.horizontal ? '1' : '0');

	return offset+5;

}

//also takes responsibility for turn as it fits in the extra space 
//might need to use unsigned chars
int fill_gamestate(char ch[], int offset, bool gamestate[][NUMCOLS], bool turn){
	// char bitstring[160];
	// int index = 0;
	// int num_spaces = (2*NUMROWS-1)*NUMCOLS; //153 +1 for turn, then last 6 bits padded
	// for (int i = 0; i < 2*NUMROWS-1; i++){
	// 	for(int j = 0; j < NUMCOLS; j++){
	// 		bitstring[index] = (gamestate[i][j] ? '1' : '0');
	// 		index++;
	// 	}
	// }
	// bitstring[index] = turn ? '1' : '0';
	// //pad last 6 bits out as 0
	// for (int i = 0; i < 6; i++){
	// 	bitstring[index++] = '0';
	// }
	

	// char temp[8];
	// for (int i = 0; i < 20; i++){
	// 	std::memcpy(temp, &bitstring[i*8], 8);
	// 	std::bitset<8> bin(temp);
	// 	char binchar = char(bin.to_ulong());
	// 	ch[offset+i] = binchar == '\0' ? '0' : binchar;
	// 	// std::cout << temp << "\n";
	// 	// std::cout << ch[offset+i];
	// }

	// return offset+20;
	char bitstring[160];
	int index = 0;
	int num_spaces = (2*NUMROWS-1)*NUMCOLS; //153 +1 for turn, then last 6 bits padded
	bool temp_gamestate[(2*NUMROWS-1)*NUMCOLS];
	memcpy(temp_gamestate, &(gamestate[0][0]), 153);

	unsigned char output = 0;
	for (int i = 0; i < 19; i++){
		for (int j=0; j < 8; j++){
			output = output|(temp_gamestate[i*8 + j]<<j);
		}
		ch[offset+i] = (char)output;
		output = 0;		
	}
	//the last bit in gamestate and turn and padding
	output = 0;
	output = output|(temp_gamestate[152]<<0);
	output = output|(turn<<1);
	//pad last 6 bits out as 0
	for (int i = 2; i < 8; i++){
		output = output|(false<<i);
	}

	return offset+20;
}


