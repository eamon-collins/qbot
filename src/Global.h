#pragma once

#define DEBUG 1

#define NUMROWS 9
#define NUMCOLS 9


//number of fences per player
#define NUMFENCES 10

#define MAXDEPTH 3
#define MAXTIME 5 //seconds it will take maximum building the state tree
#define NUM_THREADS 2
/* ex: if MAXDEPTH = 3 you have:
	1		r
			/\
	2	   c  c
		  /	\	\
	3	 c   c	...
essentially, root node counts
*/
