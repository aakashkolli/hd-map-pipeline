# Benchmarks

Hardware for current measurements:

- Machine: Apple Silicon (`arm64`)
- OS: `macOS-26.5-arm64-arm-64bit-Mach-O`
- Python: project virtualenv, Python 3.14.5

## Voxel Downsample

Command:

```bash
python scripts/benchmark.py --stage voxel --n_iterations 10
```

Input: 200,000 float32 points distributed across 20,000 occupied voxels.

| Implementation | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| C++ pybind11 | 32.98 ms | 3.92 ms | 31.29 ms | 44.68 ms |
| NumPy fallback | 70.63 ms | 1.03 ms | 68.36 ms | 72.78 ms |

Target: `< 80 ms`

Status: PASS

Measured speedup: `2.14x`
