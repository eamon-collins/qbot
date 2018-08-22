#pragma once

#define NUMROWS 9
#define NUMCOLS 9


//number of fences per player
#define NUMFENCES 10


#define MAXTIME 10000 //milliseconds it will take maximum building the state tree
#define MAXDEPTH 4
/* ex: if MAXDEPTH = 3 you have:
	1		r
			/\
	2	   c  c
		  /	\	\
	3	 c   c	...
essentially, root node counts
*/
