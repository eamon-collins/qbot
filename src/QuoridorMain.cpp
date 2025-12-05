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
	int player = 1;
	std::string save_file = "database.txt";
	std::string load_file, load_model;
	bool train = false, verbose = false;
	int num_threads = 4;
	//
	int c;
	while ((c = getopt(argc, argv, "t:p:s:l:m:bv")) != -1) {
		switch (c) {
			case 't':
				num_threads = atoi(optarg); 
				break;
			case 'p':
				player = atoi(optarg);
				break;
			case 's':
				save_file = optarg;
				break;
			case 'b':
				train = true;
				break;
			case 'v':
				verbose = true;
				break;
			case 'l':
				load_file = optarg;
				break;
			case 'm':
				load_model = optarg;
				break;
			}
	}


	//if we have a tree-file given, load the tree. Otherwise, create a new one.
	StateNode* root;
	if (load_file.size() > 0){
		root = load_tree(load_file);
	} else {
		if(player == 2) {
			root = new StateNode(false); //inits starting gamestate
			root->print_node();
		} else {
			root = new StateNode(true);
		}
	}


	//TESTING PURPOSES ONLY< REMOVE THIS SEGMENT LATER
 //    root->generate_valid_children();
	// int num_no = save_tree(root, save_file);
	// StateNode* root3 = load_tree(save_file);
	// return 0;

	//keep an eye on this to make sure large trees don't take up prohibitive amounts of mem
	//std::cout << sizeof(*root) << "\n";
	//starts new game at gamestate specified by root
	Game game = Game(root, num_threads, load_model);
	StateNode::game = &game;
	if ( train ) {
		game.humanGame = false;
        // game.parallel_self_play(save_file, num_threads, 1);
		game.better_self_play(save_file, 100);
		// game.train_alpha(save_file, 10000);
	} else {
		//uses generate_children and prune_children to generate state tree of moves from current root
		game.humanGame = true;
		game.run_game();
	}

	// auto arbitrary = game.root->children.begin();
	// std::advance(arbitrary,3);
	// int numChildren = arbitrary->children.size();


	// std::cout << numChildren <<"\n";


	// int seclevel;
	// for(int i = 0; i < numChildren; i++){
	// 	seclevel = root->children[i].generate_valid_children();
	// 	seclevel = arbitrary->children[i].score;
	// 	std::cout << seclevel << "\t";
	// 	if (seclevel >4)
	// 		std::cout << arbitrary->children[i];
	// }
	//std::cout << arbitrary->children.front().children.size() << "\n";

	// std::cout << "ROOT before save\n";
	// root->print_node();
	// root->children.front().print_node();
	//print_node(&root->children[0].children[0]);
	// std::cout << "First child of root before save\n";
	// print_node(arbitrary);




	// int num_nodes = save_tree(root, save_file);

	// std::cout << "number of nodes saved: " << num_nodes << "\n";



	return 0;
}

