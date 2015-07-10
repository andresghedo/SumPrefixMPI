/*
 * scan.c - somme prefisse di un array con MPI
 *
 *
 * Compile with:
 * mpicc -Wall scan.c -o scan
 *
 * Run with:
 * mpirun -n NP scan
 *
 * ES 4 process MPI:
 * mpirun -n 4 scan
 *
 * Il programma legge dal file di input in.txt, situato nella stessa 
 * directory dell'eseguibile, un array di n elementi di cui si richiede
 * l'implementazione parallela del calcolo delle somme prefisse.
 *
 * Author: Andrea Sghedoni <andrea.sghedoni4(at)studio.unibo.it>
 * Matricola: 0000736038
 */


#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* Struttura dati per la gestione di MPI_Wtime() */
typedef struct {
	double tstart;
    double tstop;
    } mpi_timer_t;

inline void timer_start( mpi_timer_t* t ){

    t->tstart = MPI_Wtime();

    }

inline void timer_stop (mpi_timer_t* t ){

    t->tstop = MPI_Wtime();

    }

inline double timer_elapsed (const mpi_timer_t* t ){

   return t->tstop - t->tstart;

    }

/* funzione per la lettura dell'input dal file */
void read_from_file (float *a, int size, FILE *f){

	int i;
	for(i = 0; i<size; i++)
		fscanf(f, "%f", &a[i]);
	fclose(f);

	}

/* funzione per la scrittura dei risultati nel file di output */
void write_result_outfile (float *p, int n){

	const char *path_out_file = "out.txt";
	FILE *fp = fopen(path_out_file, "w");

	if (fp == NULL) {
		printf("ERRORE APERTURA FILE!\n");
		exit(0);
		}

	fprintf(fp, "%d\n", n);
	int i;
	for (i = 0; i<n; ++i)
		fprintf(fp, "%f\n", p[i]);

	}


int main(int argc, char *argv[]) {

	int rank, comm_size;	/* id processo, numero di processi */
	const int master = 0;	/* processo master con ID 0 */
	int n, i;	/* dimensione dell'input, iteratore i */
	const char *path_inputfile = "in.txt";
	float *local_s = NULL;	/* puntatore all'array contenente le somme prefisse locali ad ogni processo */
	float *s = NULL;	/* puntatore all'array delle somme prefisse totali, subito contiene l'input iniziale */	
	int local_n;	/* numero di dati su cui ogni processo deve lavorare */
	mpi_timer_t t;	/* struttura dati per il calcolo dei tempi di esecuzione */

	MPI_Init( &argc, &argv );	/* no MPI calls before this line */
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

	int array_dim_scatterv[comm_size];	/* numero di elementi che ogni processo deve ricevere dalla scatterv */
	int array_disp_scatterv[comm_size];	/* displacement che indica da che punto dell'array reperire i dati per ogni processo */
	int array_dim_scatterv_blksum[comm_size];	/* come le var precedenti, relative alla scatterv dei float in blksum_master */
	int array_disp_scatterv_blksum[comm_size];
	float blksum_master[comm_size];	/* conterrà le somme prefisse degli ultimi elementi di ogni processore */

	FILE *f = fopen(path_inputfile, "r");
	fscanf(f, "%d", &n);

	int remainder = n % comm_size;	/* eventuale resto nel caso in cui il num di processi non sia multiplo di n */			

	if (rank == master)
		local_n =  (n/comm_size) + remainder;	/* resto preso in carico dal master */
	else
		local_n =  (n/comm_size);

	/* preparazione degli array local_s e s(master) */
	if (rank == master)
		s = (float *)malloc(sizeof(float) * n);

	local_s = (float *)malloc(sizeof(float) * local_n);

	if (rank == master){

		read_from_file(s, n, f);	/* lettura input dal file */

		/* preparazione array per la scatterv */
		array_dim_scatterv[0] = local_n;	/* solo il master prende in carico il resto (nel caso esso sia presente) */
		for (i = 1; i<comm_size; i++)
			array_dim_scatterv[i] = local_n - remainder;

		int temp = 0;
		for (i = 0; i<comm_size; i++){
			array_disp_scatterv[i] = temp;
			temp += array_dim_scatterv[i];
			}
		}

	if( rank==master )
		timer_start( &t );

	/* scatterv per lo split dei dati tra i processi */
	MPI_Scatterv(s,
			array_dim_scatterv,
			array_disp_scatterv,
			MPI_FLOAT,
			local_s,
			local_n,
			MPI_FLOAT,
			master,
			MPI_COMM_WORLD);

	/* ciclo for per le somme prefisse locali */
	for(i = 1; i<local_n; i++)
		local_s[i] += local_s[i-1];

	/* raccolta degli ultimi elementi delle somme prefisse locali nel master */
	float send_element = local_s[local_n-1];
	MPI_Gather(&send_element,
			1,
			MPI_FLOAT,
			blksum_master,
			1,
			MPI_FLOAT,
			master,
			MPI_COMM_WORLD);

    /* il master prepara i valori delle blksum prefisse da inoltrare */
	if (rank == master){

		for (i = 1; i<comm_size; i++)
			blksum_master[i] += blksum_master[i-1];

		array_dim_scatterv_blksum[0] = 0;
		for (i = 1; i<comm_size; i++)
			array_dim_scatterv_blksum[i] = 1;

		array_disp_scatterv_blksum[0] = 0;
		for (i = 1; i<comm_size; i++)
			array_disp_scatterv_blksum[i] = i-1;
		}

	float n_add;
	MPI_Scatterv(blksum_master,
			array_dim_scatterv_blksum,
			array_disp_scatterv_blksum,
			MPI_FLOAT,
			&n_add,
			1,
			MPI_FLOAT,
			master,
			MPI_COMM_WORLD);

	/* ogni processo, diverso dal master, somma il dato ricevuto alle proprie somme prefisse locali */
	if(rank != master)
		for(i = 0; i<local_n; i++)
	local_s[i] += n_add;

	/* infine si concatenano i risultati sul master, nell'array s */
	MPI_Gatherv(local_s,
			local_n,
			MPI_FLOAT,
			s,
			array_dim_scatterv,
			array_disp_scatterv,
			MPI_FLOAT,
			master,
			MPI_COMM_WORLD);

	MPI_Barrier( MPI_COMM_WORLD );

	/* stampa del tempo impiegato */
	if( rank == master ){
		timer_stop( &t );
		printf("Time sum prefix total(s): %lf\n", timer_elapsed( &t ) );
		printf("Number of process: %d\n", comm_size);
		}

	/* scrittura nel file di output out.txt*/
	if(rank == master )
		write_result_outfile (s, n);

	MPI_Finalize();	/* no MPI calls after this line */

	return EXIT_SUCCESS;
}
