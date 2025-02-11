#pragma once
#include <random>

#define DEBUG 1

#define NUMROWS 9
#define NUMCOLS 9


//number of fences per player
#define NUMFENCES 10

#define MAXDEPTH 3
#define MAXTIME 1 //seconds it will take maximum building the state tree
#define NUM_THREADS 4

//exploration parameter, higher means trying less visited nodes
#define EXPLORATION_C 2.0

extern int num_threads;
//Set up rng
extern std::random_device rd; //obtains seed
extern std::mt19937 rng; 

/* ex: if MAXDEPTH = 3 you have:
	1		r
			/\
	2	   c  c
		  /	\	\
	3	 c   c	...
essentially, root node counts
*/
