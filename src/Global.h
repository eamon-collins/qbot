#pragma once
#include <random>

#define DEBUG 1

#define NUMROWS 9
#define NUMCOLS 9


//number of fences per player
#define NUMFENCES 10

#define MAXDEPTH 3
#define MAXTIME 2 //seconds it will take maximum building the state tree

//exploration parameter, higher means trying less visited nodes
#define EXPLORATION_C 2.0


/* ex: if MAXDEPTH = 3 you have:
	1		r
			/\
	2	   c  c
		  /	\	\
	3	 c   c	...
essentially, root node counts
*/
