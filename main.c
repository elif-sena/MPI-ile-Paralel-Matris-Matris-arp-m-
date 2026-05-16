#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void read_matrix(const char *filename, double **matrix, int *N) {
    FILE *f = fopen(filename, "r");
    if (!f) { 
        fprintf(stderr, "Dosya acilamadi: %s\n", filename); 
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }
    
    char line[8192]; // Uzun CSV satırları için geniş bellek
    // İlk satırı oku ve N değerini al
    if (fgets(line, sizeof(line), f)) {
        sscanf(line, "%d", N);
    }
    
    *matrix = (double *)malloc((*N) * (*N) * sizeof(double));
    int count = 0;
    
    // Matris değerlerini virgüllerden (,) ayırarak oku
    while (fgets(line, sizeof(line), f) && count < (*N) * (*N)) {
        char *token = strtok(line, ",\n\r"); //strtok (string -> token) fonksiyonu eklendi: belirli karakterlere göre token ayrıştırma
        while (token != NULL && count < (*N) * (*N)) {
            (*matrix)[count++] = atof(token);
            token = strtok(NULL, ",\n\r");
        }
    }
    fclose(f);
}

int main(int argc, char *argv[]) {
    int rank, size, N;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A, *local_C;
    double t_start, t_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Master matrisleri okur (CSV Formatı)
    if (rank == 0) {
        read_matrix("a.csv", &A, &N);
        int N_b;
        read_matrix("b.csv", &B, &N_b);
        if (N != N_b) {
            fprintf(stderr, "Matris boyutlari eslesmiyor!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        C = (double *)calloc(N * N, sizeof(double));
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) B = (double *)malloc(N * N * sizeof(double));
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rows_per_proc = N / size;
    int extra = N % size;
    int my_rows = rows_per_proc + (rank < extra ? 1 : 0);

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (rows_per_proc + (i < extra ? 1 : 0)) * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    local_A = (double *)malloc(my_rows * N * sizeof(double));
    local_C = (double *)calloc(my_rows * N, sizeof(double));

    // -----------------------------------------------------------------------------
    int row_start = displs[rank] / N;
    int row_end = row_start + my_rows - 1;
    printf("Rank %d: Satir araligi [%d, %d] (%d satir isleniyor)\n", rank, row_start, row_end, my_rows);
    //-----------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, my_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < my_rows; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];

    MPI_Gatherv(local_C, my_rows * N, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();

    if (rank == 0) {
        printf("N=%d, P=%d, T_P=%.6f saniye\n", N, size, t_end - t_start);

        // Sonucu CSV olarak kaydet
        FILE *out = fopen("c_result.csv", "w");
        fprintf(out, "%d\n", N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(out, "%.2f%s", C[i * N + j], (j == N - 1) ? "" : ",");
            }
            fprintf(out, "\n");
        }
        fclose(out);
        printf("Sonuc c_result.csv dosyasina yazildi.\n");

        free(A); free(C);
    }

    free(B); free(local_A); free(local_C);
    free(sendcounts); free(displs);
    MPI_Finalize();
    return 0;
}