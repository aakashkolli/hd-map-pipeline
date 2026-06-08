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
| C++ pybind11 | 49.61 ms | 20.44 ms | 32.20 ms | 87.81 ms |
| NumPy fallback | 70.37 ms | 0.91 ms | 68.66 ms | 71.55 ms |

Target: `< 80 ms`

Status: PASS

Measured speedup: `1.42x`

