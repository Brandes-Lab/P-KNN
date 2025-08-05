#!/usr/bin/env python3

import argparse
import numpy as np

def bytes_str(b):
    GB = 1024**3
    MB = 1024**2
    if b >= GB:
        return f"{b/GB:.2f} GiB"
    if b >= MB:
        return f"{b/MB:.2f} MiB"
        
    return f"{b/1024:.2f} KiB"

def estimate_memory(n_tools, n_query, n_calibration, n_regularization, n_bootstrap, n_cpu_threads,
                    batch_size, dtype, index_dtype,
                    cdist_overhead, sort_overhead, imputer_overhead, safety_factor,
                    vram_gb=None, mode="gpu"):

    BYTES_PER_FLOAT = 8 if dtype == "float64" else 4
    BYTES_PER_INDEX = 8 if index_dtype == "int64" else 4
    GB = 1024 ** 3

    def size(*shape, item_size=BYTES_PER_FLOAT):
        return int(np.prod(shape)) * item_size

    print("=== P-KNN Memory Estimator ===")
    print(f"mode={mode}, dtype={dtype}, index_dtype={index_dtype}, n_threads={n_cpu_threads}")
    print(f"n_tools={n_tools}, n_query={n_query}, n_calibration={n_calibration}, n_regularization={n_regularization}, batch_size={batch_size}")
    print()

    # CPU shared base data
    cpu_raw = (size(n_query, n_tools) + size(n_calibration, n_tools) + size(n_regularization, n_tools)) * 1.3
    cpu_intermediate = cpu_raw * 2.0
    cpu_bootstrap = size(n_query, n_bootstrap)
    cpu_base = cpu_raw + cpu_intermediate + cpu_bootstrap

    if mode == "cpu":
        # per-thread computation buffer
        per_thread_compute = (
            size(batch_size, n_tools) +
            size(batch_size, n_calibration) +
            size(batch_size, n_regularization) +
            size(batch_size, n_calibration, item_size=BYTES_PER_INDEX) +
            size(batch_size, n_regularization,   item_size=BYTES_PER_INDEX) +
            size(batch_size, n_calibration) * imputer_overhead
        ) * safety_factor

        cpu_total = cpu_base + per_thread_compute * n_cpu_threads

        print("[CPU RAM Estimate - Parallel Mode]")
        print(f"  Base data (raw + intermediate): {bytes_str(cpu_base)}")
        print(f"  Per-thread compute buffer: {bytes_str(per_thread_compute)}")
        print(f"  Threads: {n_cpu_threads}")
        print(f"  ----------------------------------------------")
        print(f"  **Estimated total CPU RAM**: {bytes_str(cpu_total)}")
        print()
        return

    # GPU estimate
    calib_tensor = size(n_calibration, n_tools)
    reg_tensor   = size(n_regularization, n_tools)
    test_batch_tensor = size(batch_size, n_tools)
    calib_label = size(n_calibration, item_size=1)
    base_resident = calib_tensor + reg_tensor + test_batch_tensor + calib_label

    d_calib_vals = size(batch_size, n_calibration)
    d_reg_vals   = size(batch_size, n_regularization)
    d_calib_indices = size(batch_size, n_calibration, item_size=BYTES_PER_INDEX)
    d_reg_indices   = size(batch_size, n_regularization,   item_size=BYTES_PER_INDEX)

    cdist_block = (d_calib_vals + d_reg_vals) * cdist_overhead
    sort_block  = (d_calib_vals + d_reg_vals + d_calib_indices + d_reg_indices) * sort_overhead
    imputer_block = (size(batch_size, n_calibration) * imputer_overhead)

    per_batch_peak = (base_resident + cdist_block + sort_block + imputer_block)
    per_batch_peak *= safety_factor

    print("[CPU RAM Estimate - Base]")
    print(f"  Raw arrays (x1.3): {bytes_str(cpu_raw)}")
    print(f"  Intermediates (x2.0): {bytes_str(cpu_intermediate)}")
    print(f"  Bootstrap results: {bytes_str(cpu_bootstrap)}")
    print(f"  ----------------------------------------------")
    print(f"  Total (Base): {bytes_str(cpu_base)}")
    print()

    print("[GPU VRAM Estimate per batch]")
    print(f"  Base tensors (query/calib/reg): {bytes_str(base_resident)}")
    print(f"  cdist block * {cdist_overhead}: {bytes_str(cdist_block)}")
    print(f"  sort/topk block * {sort_overhead}: {bytes_str(sort_block)}")
    print(f"  imputer block * {imputer_overhead}: {bytes_str(imputer_block)}")
    print(f"  ----------------------------------------------")
    print(f"  **Estimated VRAM per batch**: {bytes_str(per_batch_peak)}")
    print()

    if vram_gb is not None:
        vram_bytes = vram_gb * GB
        if per_batch_peak > vram_bytes:
            print(f"[Warning] Estimated batch exceeds {vram_gb:.1f} GiB limit")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate memory usage for P-KNN run.")
    parser.add_argument('--n_tools', type=int, required=True,
                        help='Number of tools (features) in the dataset')
    parser.add_argument('--n_query', type=int, required=True,
                        help='Number of query samples')
    parser.add_argument('--n_calibration', type=int, required=True,
                        help='Number of calibration samples')
    parser.add_argument('--n_regularization', type=int, required=True,
                        help='Number of regularization samples')
    parser.add_argument('--n_bootstrap', type=int, default=100,
                        help='Number of bootstrap samples for calibration (default: 100)')
    parser.add_argument('--n_cpu_threads', type=int, default=1,
                        help='Number of CPU threads to use (default: 1)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for GPU processing (default: 512)')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64',
                        help='Data type for computations (default: float64)')
    parser.add_argument('--index_dtype', choices=['int32', 'int64'], default='int64',
                        help='Index data type for computations (default: int64)')
    parser.add_argument('--cdist_overhead', type=float, default=1.3,
                        help='Overhead multiplier for cdist (default: 1.3)')
    parser.add_argument('--sort_overhead', type=float, default=2.0,
                        help='Overhead multiplier for sort/topk (default: 2.0)')
    parser.add_argument('--imputer_overhead', type=float, default=1.5,
                        help='Overhead multiplier for imputer (default: 1.5)')
    parser.add_argument('--safety_factor', type=float, default=1.2,
                        help='Safety factor for memory estimates (default: 1.2)')
    parser.add_argument('--vram_gb', type=float, default=None,
                        help='Available GPU VRAM in GiB (default: None)')
    parser.add_argument('--mode', choices=['cpu', 'gpu'], default='gpu',
                        help='Estimate mode: cpu (multi-threaded) or gpu (batched)')

    args = parser.parse_args()
    estimate_memory(**vars(args))
