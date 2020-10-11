

#include "Global.h"
#include "Tree.h"
#include "Game.h"
#include "storage.h"
#include <unistd.h>
#include <cstdlib>
#include <cstddef>
#include <iostream>

int main(int argc, char *argv[]){
	int threadCount, player;
	std::string save_file = "database.txt";
	//
	int c;
	while ((c = getopt(argc, argv, "t:p:")) != -1) {
		switch (c) {
			case 't':
				threadCount = atoi(optarg);
				break;
			case 'p':
				player = atoi(optarg);
				break;
			case 's':
				save_file = optarg;
			}	
		break;
	}

	StateNode* root;
	if(player == 2)
		root = new StateNode(false); //inits starting gamestate
	else
		root = new StateNode(true);


	//TESTING PURPOSES ONLY< REMOVE THIS SEGMENT LATER
	// int num_no = save_tree(root, save_file);
	// StateNode* root3 = load_tree(save_file);
	// return 0;

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

	int num_nodes = save_tree(root, save_file);

	std::cout << "number of nodes saved: " << num_nodes << "\n";

	StateNode* root2 = load_tree(save_file);

	int second_time = save_tree(root2, "database2.txt");

	std::cout << "number of nodes after recreation: " << second_time << "\n";

	

	//std::cout << root->children[5].children.size() << "\n";
	

	// for(int i = 0; i < numChildren; i++){
	// 	std::cout << root->children[i];
	// }

	return 0;
}

