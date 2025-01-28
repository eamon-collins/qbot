#pragma once

#define DEBUG 1

#define NUMROWS 9
#define NUMCOLS 9


//number of fences per player
#define NUMFENCES 3

#define MAXDEPTH 3
#define MAXTIME 2 //seconds it will take maximum building the state tree
#define NUM_THREADS 4

extern int num_threads;
/* ex: if MAXDEPTH = 3 you have:
	1		r
			/\
	2	   c  c
		  /	\	\
	3	 c   c	...
essentially, root node counts
*/
