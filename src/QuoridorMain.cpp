#include "Global.h"
#include "Tree.h"
#include "Game.h"
#include "storage.h"
#include <unistd.h>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <ctime>


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

	auto arbitrary = game.root->children.begin();
	std::advance(arbitrary,3);
	int numChildren = arbitrary->children.size();


	std::cout << numChildren <<"\n";


	// int seclevel;
	// for(int i = 0; i < numChildren; i++){
	// 	seclevel = root->children[i].generate_valid_children();
	// 	seclevel = arbitrary->children[i].score;
	// 	std::cout << seclevel << "\t";
	// 	if (seclevel >4)
	// 		std::cout << arbitrary->children[i];
	// }
	std::cout << arbitrary->children.front().children.size() << "\n";

	std::cout << "ROOT before save\n";
	root->print_node();
	root->children.front().print_node();
	//print_node(&root->children[0].children[0]);
	// std::cout << "First child of root before save\n";
	// print_node(arbitrary);

	root->children.front().children.front().play_out();

	//test build_tree

	// build_tree(&(root->children.front().children.front().children.front()),
	// 	0, std::time(0));



	int num_nodes = save_tree(root, save_file);

	std::cout << "number of nodes saved: " << num_nodes << "\n";

	// StateNode* root2 = load_tree(save_file);

	// std::cout << "ROOT after save\n";
	// print_node(root2);
	// print_node(&(root2->children.front()));
	// //print_node(&root2->children[0].children[0]);
	// // std::cout << "First child of root after save\n";
	// // print_node(&root2->children[0]);
	// int second_time = save_tree(root2, "database2.txt");

	// std::cout << "number of nodes after recreation: " << second_time << "\n";

	
	//std::cout << root->children[5].children.size() << "\n";
	

	// for(int i = 0; i < numChildren; i++){
	// 	std::cout << root->children[i];
	// }

	return 0;
}

