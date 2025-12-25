# Oracle Integration Components

Complete Oracle integration stack for RBPF regime-switching system.

## Directory Structure

```
PGAS/
├── CMakeLists.txt              # Main PGAS build (add subdirectories here)
├── pgas_mkl.h / .c             # Core PGAS sampler
├── paris_mkl.h / .c            # PARIS smoother (scout sweeps)
│
├── hawkes_signal/              # Trigger Signal #1: Market activity
│   ├── CMakeLists.txt
│   ├── hawkes_integrator.h/c
│   └── test_hawkes_scenarios.c
│
├── kl_trigger/                 # Trigger Signal #2: Model confusion
│   ├── CMakeLists.txt
│   ├── kl_trigger.h/c
│   └── test_kl_trigger.c
│
├── pgas_confidence/            # PGAS output quality → adaptive γ
│   ├── CMakeLists.txt
│   ├── pgas_confidence.h/c
│   └── test_pgas_confidence.c
│
├── saem_blender/               # Safe parameter blending
│   ├── CMakeLists.txt
│   ├── saem_blender.h/c
│   ├── test_saem_blender.c
│   └── test_three_tier_reset.c
│
├── thompson_sampler/           # Explore/exploit handoff
│   ├── CMakeLists.txt
│   ├── thompson_sampler.h/c
│   └── test_thompson_sampler.c
│
├── rbpf_trajectory/            # Thread-safe trajectory buffer
│   ├── CMakeLists.txt
│   ├── rbpf_trajectory.h/c
│   └── test_rbpf_trajectory.c
│
└── oracle_bridge/              # Full integration pipeline
    ├── CMakeLists.txt
    ├── oracle_bridge.h/c
    ├── test_oracle_integration.c
    └── test_oracle_full_integration.c
```

## Add to PGAS/CMakeLists.txt

```cmake
#───────────────────────────────────────────────────────────────────────────────
# Oracle Integration Components
#───────────────────────────────────────────────────────────────────────────────

# Independent components (no inter-dependencies)
add_subdirectory(hawkes_signal)
add_subdirectory(kl_trigger)
add_subdirectory(saem_blender)
add_subdirectory(thompson_sampler)
add_subdirectory(rbpf_trajectory)

# Depends on pgas_mkl
add_subdirectory(pgas_confidence)

# Top-level integration (depends on all above)
add_subdirectory(oracle_bridge)
```

## Component Dependency Graph

```
                    ┌──────────────────┐
                    │  oracle_bridge   │  ← Top-level integration
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ hawkes_integrat │ │   kl_trigger    │ │ rbpf_trajectory │
│     (trigger)   │ │   (dual-gate)   │ │   (snapshot)    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         └───────────┬───────┴───────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────────┐
│  pgas_mkl   │ │  paris_mkl  │ │ pgas_confidence  │
│  (sampler)  │ │   (scout)   │ │    (γ adapt)     │
└─────────────┘ └─────────────┘ └──────────────────┘
         │           │                   │
         └───────────┼───────────────────┘
                     ▼
         ┌───────────────────┐
         │   saem_blender    │ ← Safe Π updates
         └─────────┬─────────┘
                   │
                   ▼
         ┌───────────────────┐
         │ thompson_sampler  │ ← Explore/exploit
         └───────────────────┘
```

## Data Flow

```
RBPF Thread                              Oracle Thread
───────────────────────────────────────────────────────────────────

1. TRIGGER DETECTION
   [returns] ──►  Hawkes Integrator ──►  surprise_σ > threshold?
   [innov]   ──►  KL Trigger        ──►  kl_σ > threshold?
                                              │
                      (dual-gate: both must fire)
                                              ▼
2. TRAJECTORY HANDOFF                   
   [regimes, h] ──► RBPFTrajectory ──snapshot──► [ref_path]
                                              │
3. SCOUT PRE-VALIDATION                       ▼
                                        PARIS scout sweep
                                        Check: acceptance > 0.10?
                                        Check: unique_frac > 0.25?
                                              │
4. PGAS SAMPLING                              ▼
                                        PGAS Gibbs sweeps
                                        Output: Π_new, counts
                                              │
5. CONFIDENCE → γ                             ▼
                                        PGASConfidence metrics
                                        γ = f(ESS, acceptance, divergence)
                                              │
6. SAFE BLENDING                              ▼
                                        SAEM blender
                                        Q_new = (1-γ)Q_old + γ·S_oracle
                                              │
7. THOMPSON HANDOFF                           ▼
                                        Thompson Sampler
                                        Explore: sample Π ~ Dir(Q)
                                        Exploit: Π = Q / sum(Q)
                                              │
   [Π_updated] ◄────────────────────────────◄─┘
```

## Build & Test

```bash
cd build
cmake ..
cmake --build . --config Release

# Run all Oracle tests
ctest -R hawkes
ctest -R kl_trigger
ctest -R pgas_confidence
ctest -R saem_blender
ctest -R three_tier
ctest -R thompson_sampler
ctest -R rbpf_trajectory
ctest -R oracle_integration
ctest -R oracle_full_integration

# Or run all at once
ctest -R "hawkes|kl_|saem|thompson|rbpf_traj|oracle"
```

## MSVC Compatibility Notes

All components have been tested and fixed for MSVC 2022:

| Issue | Fix Applied |
|-------|-------------|
| `__attribute__((aligned))` | Cross-platform `HAWKES_ALIGNED_ARRAY` macro |
| `__has_include` | Explicit `#ifdef USE_MKL` |
| VLAs (`const int N = 500; float arr[N]`) | `#define` constants |
| `clock_gettime` / `CLOCK_MONOTONIC` | `QueryPerformanceCounter` on Windows |
| `<stdatomic.h>` in C mode | Windows Interlocked functions |
| `struct PGASMKLState` vs `PGASMKLState` | Use typedef directly |

## External Dependencies

| Component | MKL Required? | Notes |
|-----------|---------------|-------|
| hawkes_signal | Optional | Falls back to scalar exp() |
| kl_trigger | No | Pure C |
| saem_blender | No | Pure C |
| thompson_sampler | No | Pure C |
| rbpf_trajectory | No | Pure C11 (atomics) |
| pgas_confidence | Yes | Links pgas_mkl |
| oracle_bridge | Yes | Links pgas_mkl, paris_mkl |
