

#include "storage.h"
#include <bitset>
#include <fstream>
#include <cstdio>
#include <cstring>

//saves the tree to disk
//~177 characters estimated for each statenode
int save_tree(StateNode* root){
	// Archive<StateNode> ar = new Archive;
	// serialize(ar, *root, 1);


	
}

// namespace boost {
// namespace serialization {

// template<class Archive>
// void serialize(Archive & ar, StateNode & s, const unsigned int version)
// {
//     ar & s.children;
//     ar & s.parent;
//     ar & s.move;
//     ar & s.p1;
//     ar & s.p2;
// }

// } // namespace serialization
// } // namespace boost


bool write_node(StateNode* node, FILE f ){
	char ch[178];

	int offset = 0;
	offset = fill_move(ch, offset, node->move);
	offset = fill_player(ch, offset, node->p1);
	offset = fill_player(ch, offset, node->p2);



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

	index = 0;
	char temp[8];
	char output[20];
	for (int i = 0; i < 20; i++){
		std::memcpy(temp, *(bitstring[i*8]), 8);
		std::bitset<8> bin(temp);
		ch[offset+i] = char(bin.to_ulong());		
	}

	return offset+20;
}


