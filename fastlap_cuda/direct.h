#ifndef _direct_h_included_
#define _direct_h_included_

#include "mulStruct.h"
#include "mulGlobal.h"

double **Q2P(snglrty **sngs, int numsngs, fieldpt **fpts, int numfpts, int swapOnly, double **mat);
double **Q2PAlloc(int nrows, int ncols);
double **ludecomp(double **matin, int size, int allocate);
void solve(double **mat, double *x, double *b, int size);
void invert(double **mat, int size, int *reorder);
void matcheck(double **mat, int rows, int size);
void matlabDump(double **mat, int size, char *name);
void coeffSwap(snglrty **sngs, int numsngs, int rows, double **mat);


#endif