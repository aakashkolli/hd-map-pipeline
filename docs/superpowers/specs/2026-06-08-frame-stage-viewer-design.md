# Frame + Stage Viewer Design

## Problem

Current point cloud is 5 frames of synthetic flat-ground data accumulated into a rectangular slab. It has no 3D structure and looks nothing like real LiDAR data.

## Solution

**1. Realistic synthetic Velodyne scans** — regenerate each of the 5 raw `.bin` frames as a proper HDL-64E scan: 64 elevation rings (−24° to +3°), 1800 azimuth steps, vehicle at LiDAR height 1.73 m. Geometry includes road surface, flanking building walls at ±10 m, and lane markings at ±1.5 m with high intensity. Vehicle advances 10 m per frame in world ENU. Result: the characteristic concentric-ring LiDAR pattern instead of a flat slab.

**2. Per-frame binary files** — pipeline writes:
- `data/outputs/frames/frame_{N}_raw.bin` — all points for frame N
- `data/outputs/frames/frame_{N}_ground.bin` — ground-separated points
- `data/outputs/frames/frame_{N}_obstacles.bin` — non-ground points
- `data/outputs/points.bin` — accumulated raw (unchanged, all 5 frames)

**3. Frame selector** — sidebar "Scene" section with buttons: Accumulated / Frame 0 / Frame 1 / Frame 2 / Frame 3 / Frame 4. Default on load: Frame 0.

**4. Stage selector** — sidebar "Stage" section with buttons: Raw / Ground / Obstacles. Active only when a specific frame is selected; accumulated always shows raw.

## Data Format

Same binary format as existing `points.bin`:
- bytes 0–3: uint32 N (point count, little-endian)
- bytes 4..4+N×12: float32 xyz triples, FRAME: world ENU
- bytes 4+N×12..: float32 intensities [0, 1]

## Frontend URL pattern

`/data/frames/frame_{N}_{stage}.bin` served by Vite's existing static file serving from `public/data/` → `data/outputs/`.

## Files changed

| File | Change |
|---|---|
| `scripts/run_pipeline.py` | Add `_generate_realistic_frames()`, call from `_run_full_kitti()` |
| `src/viz/src/io/DataLoader.ts` | Add `loadFrameStage(frame, stage)` |
| `src/viz/src/controls/Sidebar.ts` | Add Scene + Stage sections, `onSceneChange` callback |
| `src/viz/src/main.ts` | Wire `onSceneChange`, default to Frame 0 on initial load |
