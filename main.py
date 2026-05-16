from mpi4py import MPI
import numpy as np
import os

# NumPy'ın kendi thread'lerini kapat
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def read_matrix_csv(filename):
    with open(filename, 'r') as f:
        # İlk satırdaki N değerini virgüllerden ayırıp al
        first_line = f.readline().strip().split(',')
        N = int(first_line[0])
        data = []
        for line in f:
            # Satırı virgülden böl, boş alanları filtrele ve float yap
            tokens = line.strip().split(',')
            data.extend([float(x) for x in tokens if x])
    return N, np.array(data, dtype=np.float64).reshape(N, N)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

A = None
B = None
N = None

if rank == 0:
    # Artık a.csv ve b.csv'den okunuyor
    N_a, A = read_matrix_csv("a.csv")
    N_b, B = read_matrix_csv("b.csv")
    assert N_a == N_b, "Matris boyutlari eslesmiyor!"
    N = N_a

comm.Barrier()
t_start = MPI.Wtime()

N = comm.bcast(N, root=0)
B = comm.bcast(B, root=0)

rows_per_proc = N // size
extra = N % size
my_rows = rows_per_proc + (1 if rank < extra else 0)

sendcounts = [(rows_per_proc + (1 if i < extra else 0)) * N for i in range(size)]
displs     = [sum(sendcounts[:i]) for i in range(size)]

local_A = np.zeros((my_rows, N), dtype=np.float64)
comm.Scatterv([A, sendcounts, displs, MPI.DOUBLE], local_A, root=0)

# Manuel matris çarpımı
local_C = np.zeros((my_rows, N), dtype=np.float64)
for i in range(my_rows):
    for k in range(N):
        for j in range(N):
            local_C[i, j] += local_A[i, k] * B[k, j]

C = None
if rank == 0:
    C = np.zeros((N, N), dtype=np.float64)

comm.Gatherv(local_C, [C, sendcounts, displs, MPI.DOUBLE], root=0)

comm.Barrier()
t_end = MPI.Wtime()

if rank == 0:
    print(f"N={N}, P={size}, T_P={t_end - t_start:.6f} saniye")
    
    # Sonucu CSV dosyası olarak dışa aktar
    with open("py_result.csv", "w") as f:
        f.write(f"{N}\n")
        for row in C:
            f.write(",".join(f"{v:.2f}" for v in row) + "\n")
    print("Sonuc py_result.csv dosyasina yazildi.")