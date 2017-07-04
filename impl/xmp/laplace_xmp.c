/*************************************************
 * Laplace XcalableMP C Version                                         
 *                                                               
 * T is initially 0.0                                            
 * Boundaries are as follows                                     
 *                                                               
 *                T                      4 sub-grids            
 *   0  +-------------------+  0    +-------------------+       
 *      |                   |       |                   |           
 *      |                   |       |-------------------|         
 *      |                   |       |                   |      
 *   T  |                   |  T    |-------------------|             
 *      |                   |       |                   |     
 *      |                   |       |-------------------|            
 *      |                   |       |                   |   
 *   0  +-------------------+ 100   +-------------------+         
 *      0         T       100                                    
 *                                                                 
 * Each PE only has a local subgrid.
 * Each PE works on a sub grid and then sends         
 * its boundaries to neighbors.
 *                                                                 
 *  Xinzhe WU, Maison de la Simulation 2017
 *
 *******************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <xmp.h>

#define COLUMNS 672
#define ROWS 672

#define NPES 4

#define DOWN 100
#define UP   101

#define MAX_TEMP_ERROR 0.01

#pragma xmp nodes p(NPES)
#pragma xmp template t(0:ROWS+1)
#pragma xmp distribute t(block) onto p

double Temperature[ROWS+2][COLUMNS+2];
double Temperature_last[ROWS+2][COLUMNS+2];

#pragma xmp align Temperature[i][*] with t(i)
#pragma xmp align Temperature_last[i][*] with t(i)
#pragma xmp shadow Temperature_last[1][0]

void initialize();
void track_progress(int iter, double dt);

int main(int argc, char *argv[]){

    int i, j;
    int max_iterations;
    int iteration = 1;
    double start_time, stop_time, elapsed_time;

    int npes;
    int my_PE_num;
    double dt = 100;

    my_PE_num = xmp_node_num();
    npes = xmp_num_nodes();

//    printf("My rank is %d of %d processors \n", my_PE_num, npes);
#pragma xmp task on p(1)
    printf("Running on %d MPI processes\n\n", npes);

    if(npes != NPES){
#pragma xmp task on p(1)
        printf("This code must be run with %d PEs\n", NPES);
	return 0;
    }

#pragma xmp task on p(1)
{
    max_iterations = 4000;
    printf("Maximum iterations = %d\n", max_iterations);
}

#pragma xmp bcast (max_iterations)

    if(my_PE_num == 1){
        start_time = xmp_wtime();
    }
    initialize(npes, my_PE_num);
    while (dt > MAX_TEMP_ERROR && iteration <= max_iterations){
#pragma xmp reflect (Temperature_last)
#pragma xmp loop on t(i)
    for(i = 1; i <= ROWS; i++)
	for(j = 1; j <= COLUMNS; j++)
		Temperature[i][j] = 0.25 * (Temperature_last[i+1][j]+Temperature_last[i-1][j] + Temperature_last[i][j+1] + Temperature_last[i][j-1]);

    dt = 0.0;
#pragma xmp loop on t(i) reduction(max:dt)
    for(i = 1; i <= ROWS; i++){
	for(j = 1; j <= COLUMNS; j++){
		dt = fmax(fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
		Temperature_last[i][j] = Temperature[i][j];
	}
    }
    if(iteration % 100 == 0){
	if(my_PE_num == 4){
        	track_progress(iteration, dt);
	}
   }
    iteration ++;
   }
#pragma xmp barrier
    if(my_PE_num == 1)
    {
        stop_time = xmp_wtime();
   	elapsed_time = stop_time - start_time;

   	printf("\nMax error at iteration %d was %20.15g\n", iteration - 1, dt);
    	printf("Total time was %f seconds.\n", elapsed_time);
    }

}

void initialize(int npes, int my_PE_num){
    double tMin, tMax;
    int i, j;

#pragma xmp loop on t(i)
    for(i=0; i <= ROWS+1; i++)
	for(j = 0; j <= COLUMNS+1; j++)
		Temperature_last[i][j] = 0.0;

tMin = (my_PE_num - 1)*100.0/npes;
tMax = (my_PE_num)*100.0/npes;

#pragma xmp loop on t(i)
    for(i=1; i <= ROWS; i++){
	Temperature_last[i][0] = 0.0;
	Temperature_last[i][COLUMNS+1] = tMin + ((tMax-tMin)/ROWS)*i;
    }

if(my_PE_num == 1)
    for(j = 0; j <= COLUMNS+1; j++)
	Temperature_last[0][j] = (100.0 / COLUMNS) * j;

if(my_PE_num == npes)
    for(j = 0; j <= COLUMNS+1; j++)
	Temperature_last[ROWS+1][j] = (100.0/COLUMNS) * j;
}

void track_progress(int iter, double dt){

    int i;
    printf("---- Iteration %d, dt = %f ----\n", iter, dt);
    for(i = 5; i >=3; i--){
	printf("[%d,%d]: %5.2f ", ROWS-i, COLUMNS-i, Temperature[ROWS-i][COLUMNS-i]);
    }
    printf("\n");

}

