#!/usr/bin/env python3
"""
Tüm MPI deneylerini otomatik çalıştırır, süreleri toplar,
Speedup ve Efficiency hesaplar, tablo olarak gösterir.

Kullanım: python3 run_all.py
"""

import subprocess
import re
import sys

PROCESSORS = [1, 2, 4, 8, 16]
N = 512  # Matris boyutu

def run(cmd, oversubscribe=False):
    """Komutu çalıştır, T_P süresini döndür."""
    if oversubscribe:
        cmd = ["mpirun", "--oversubscribe"] + cmd[1:]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        output = result.stdout + result.stderr
        match = re.search(r"T_P=([\d.]+)", output)
        if match:
            return float(match.group(1))
        else:
            print(f"  ⚠️  Çıktı yakalanamadı:\n{output}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Zaman aşımı!")
        return None
    except Exception as e:
        print(f"  ⚠️  Hata: {e}")
        return None

def print_table(title, results):
    print(f"\n{'='*57}")
    print(f"  {title}")
    print(f"{'='*57}")
    print(f"{'P':>4} | {'T_P (s)':>10} | {'S (Speedup)':>12} | {'E (Efficiency)':>14}")
    print(f"{'-'*4}-+-{'-'*10}-+-{'-'*12}-+-{'-'*14}")
    T1 = results.get(1)
    for p in PROCESSORS:
        tp = results.get(p)
        if tp is None:
            print(f"{p:>4} | {'HATA':>10} | {'—':>12} | {'—':>14}")
        else:
            S = T1 / tp if T1 else 0
            E = S / p
            print(f"{p:>4} | {tp:>10.6f} | {S:>12.4f} | {E:>14.4f}")

def run_experiments(label, cmd_builder):
    print(f"\n🔄 {label} deneyleri başlatılıyor (N={N})...")
    results = {}
    for p in PROCESSORS:
        oversubscribe = p > 4
        cmd = cmd_builder(p)
        print(f"  ▶ P={p:>2} çalışıyor...", end=" ", flush=True)
        tp = run(cmd, oversubscribe=oversubscribe)
        if tp is not None:
            print(f"T_P = {tp:.6f} s")
        results[p] = tp
    return results

# ── C deneyleri ──────────────────────────────────────────
c_results = run_experiments(
    label="C (OpenMPI)",
    cmd_builder=lambda p: ["mpirun", "-np", str(p), "./matmul_mpi"]
)

# ── Python deneyleri ─────────────────────────────────────
py_results = run_experiments(
    label="Python (mpi4py)",
    cmd_builder=lambda p: ["mpirun", "-np", str(p), "python3", "matmul_mpi.py"]
)

# ── Tablolar ─────────────────────────────────────────────
print_table("C (OpenMPI) Sonuclari", c_results)
print_table("Python (mpi4py) Sonuclari", py_results)

# ── CSV olarak kaydet ─────────────────────────────────────
with open("results.csv", "w") as f:
    f.write("Dil,P,T_P,Speedup,Efficiency\n")
    for label, results in [("C", c_results), ("Python", py_results)]:
        T1 = results.get(1)
        for p in PROCESSORS:
            tp = results.get(p)
            if tp and T1:
                S = T1 / tp
                E = S / p
                f.write(f"{label},{p},{tp:.6f},{S:.4f},{E:.4f}\n")
            else:
                f.write(f"{label},{p},HATA,—,—\n")

print("\n✅ Sonuçlar results.csv dosyasına da kaydedildi.")