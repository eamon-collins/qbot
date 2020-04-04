

#include "Global.h"
#include "Tree.h"
#include "Game.h"
#include <unistd.h>
#include <cstdlib>
#include <cstddef>
#include <iostream>

int main(int argc, char *argv[]){
	int threadCount, player;


	int c;
	while ((c = getopt(argc, argv, "t:p:")) != -1) {
		switch (c) {
			case 't':
				threadCount = atoi(optarg);
				break;
			case 'p':
				player = atoi(optarg);
				break;
			}	
		break;
	}

	StateNode* root;
	if(player == 2)
		root = new StateNode(false); //inits starting gamestate
	else
		root = new StateNode(true);

	//keep an eye on this to make sure large trees don't take up prohibitive amounts of mem
	//std::cout << sizeof(*root) << "\n";

	//starts new game at gamestate specified by root
	Game game = Game(root);
	//uses generate_children and prune_children to generate state tree of moves from current root
	game.run_game();

	StateNode* arbitrary = &game.root->children[0];
	int numChildren = arbitrary->children.size();


	std::cout << numChildren <<"\n";

	int seclevel;
	for(int i = 0; i < numChildren; i++){
		//seclevel = root->children[i].generate_valid_children();
		seclevel = arbitrary->children[i].score;
		// std::cout << seclevel << "\t";
		// if (seclevel >4)
		// 	std::cout << arbitrary->children[i];
	}
	std::cout << arbitrary->children[0].children.size() << "\n";

	//std::cout << root->children[5].children.size() << "\n";
	

	// for(int i = 0; i < numChildren; i++){
	// 	std::cout << root->children[i];
	// }

	return 0;
}

