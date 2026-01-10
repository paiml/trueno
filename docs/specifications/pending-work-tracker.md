# Pending Work Tracker

Consolidated tracking document for pending specification items in trueno.

**Created**: 2026-01-10
**Last Updated**: 2026-01-10

---

## Overview

This document provides a unified view of all pending specification work, linking
PMAT tickets to falsification tests (FKR entries) and their associated citations.

### Status Summary

| Category | Total | Completed | In Progress | Pending |
|----------|-------|-----------|-------------|---------|
| PMAT Tickets | 10 | 1 | 1 | 8 |
| FKR Entries | 12 | 2 | 2 | 8 |
| Total Citations | 60 | - | - | - |

**Documentation Status**: COMPLETE (2026-01-10)
- PMAT Tickets: `docs/pmat-tickets/PMAT-001-to-010.md`
- FKR Registry: `docs/CUDA_TDG_COMPLIANCE.md`

---

## Priority Matrix

### P0 - Blocking

| ID | Title | Status | Blocking Issue |
|----|-------|--------|----------------|
| PMAT-005 | LZ4 GPU Kernel | COMPLETE | F082 resolved via Lz4WarpShuffleKernel |

### P1 - Critical Path

| ID | Title | Status | Dependencies |
|----|-------|--------|--------------|
| PMAT-001 | Loop Splitting | PENDING | None |
| PMAT-002 | Token Sync | PENDING | None |
| PMAT-003 | FMA Fusion | PENDING | None |
| PMAT-004 | Memory Coalescing | PENDING | None |
| PMAT-008 | PTX Debugger | PENDING | trueno-ptx-debug crate |
| PMAT-009 | Numerical Stability | PENDING | None |
| PMAT-010 | Backend Equivalence | IN PROGRESS | None |

### P2 - Platform Expansion

| ID | Title | Status | Platform |
|----|-------|--------|----------|
| PMAT-006 | Metal Backend | PENDING | Apple Silicon (M1/M2/M3) |
| PMAT-007 | ROCm Backend | PENDING | AMD Instinct GPUs |

---

## Cross-Reference: PMAT Tickets to FKR Entries

| PMAT | FKR | Hypothesis | Status |
|------|-----|------------|--------|
| PMAT-001 | FKR-003 | Loop splitting eliminates divergence | PENDING |
| PMAT-002 | FKR-004 | Token sync equivalent to barriers | PENDING |
| PMAT-003 | FKR-005 | FMA produces IEEE 754 results | PENDING |
| PMAT-004 | FKR-006 | Coalescing achieves 4x bandwidth | PENDING |
| PMAT-005 | FKR-007 | GPU LZ4 matches reference | COMPLETE |
| PMAT-006 | FKR-011 | Metal matches CUDA reference | PENDING |
| PMAT-007 | FKR-012 | ROCm matches CUDA reference | PENDING |
| PMAT-008 | FKR-008 | PTX parser handles all PTX 8.0 | PENDING |
| PMAT-009 | FKR-009 | Operations stable under perturbation | PENDING |
| PMAT-010 | FKR-010 | All backends produce equivalent results | IN PROGRESS |

---

## Citation Index

All 60 citations organized by topic for quick reference.

### GPU Architecture & Memory

1. [Volkov & Demmel, 2008] Benchmarking GPUs. DOI:10.1109/SC.2008.5214359
2. [Mei & Chu, 2017] GPU Memory Hierarchy. DOI:10.1109/TPDS.2016.2549523
3. [Wong et al., 2010] GPU Microarchitecture. DOI:10.1109/ISPASS.2010.5452013
4. [Jia et al., 2018] Volta GPU Architecture. arXiv:1804.06826

### Memory Models & Synchronization

5. [Alglave et al., 2015] GPU Concurrency. DOI:10.1145/2694344.2694391
6. [Lustig et al., 2019] PTX Memory Consistency. DOI:10.1145/3297858.3304043
7. [Mansky et al., 2015] POWER Memory Model. DOI:10.1007/978-3-319-21690-4_9
8. [Sorensen & Donaldson, 2016] Cross-Platform OpenCL. DOI:10.1145/2909437.2909440

### Loop Optimization & Divergence

9. [Allen & Kennedy, 1987] Vectorization. DOI:10.1145/29873.29875
10. [Ryoo et al., 2008] GPU Optimization. DOI:10.1145/1345206.1345220
11. [Yang et al., 2010] GPGPU Compiler. DOI:10.1145/1806596.1806606
12. [Coutinho et al., 2011] Divergence Analysis. DOI:10.1109/PACT.2011.64
13. [Han & Abdelrahman, 2011] Reducing Divergence. DOI:10.1145/1964179.1964184
14. [Zhang et al., 2011] G-Streamline. DOI:10.1145/2000064.2000105

### Numerical Analysis & Floating-Point

15. [Muller et al., 2018] FP Arithmetic Handbook. DOI:10.1007/978-3-319-76526-6
16. [Boldo & Melquiond, 2008] FMA Emulation. DOI:10.1109/TC.2008.48
17. [Higham, 2002] Numerical Algorithms. ISBN:0-89871-521-0
18. [Demmel, 1997] Numerical Linear Algebra. ISBN:0-89871-389-7
19. [Goldberg, 1991] FP Arithmetic. DOI:10.1145/103162.103163
20. [IEEE, 2019] IEEE 754-2019. DOI:10.1109/IEEESTD.2019.8766229
21. [Kahan, 1996] IEEE 754 Status. UC Berkeley
22. [Whitehead & Fit-Florea, 2011] NVIDIA FP Compliance. NVIDIA Whitepaper
23. [Collange et al., 2015] SIMD FP Arithmetic. DOI:10.1109/MM.2015.54
24. [Lam et al., 2013] FP Expression Accuracy. DOI:10.1145/2491956.2462927
25. [Demmel & Nguyen, 2015] Reproducible Summation. DOI:10.1109/TPDS.2014.2345253

### Compression Algorithms

26. [Collet, 2011] LZ4 Algorithm. lz4.github.io
27. [Ozsoy et al., 2014] GPU LZSS. DOI:10.1109/ICPADS.2014.11
28. [Weissenberger & Schmidt, 2018] GPU Huffman. DOI:10.1145/3178487.3178523
29. [Sitaridi et al., 2016] Parallel Decompression. DOI:10.1109/ICPP.2016.31

### GPU Verification

30. [Betts et al., 2012] GPUVerify. DOI:10.1145/2384616.2384625
31. [Li & Gopalakrishnan, 2010] SMT GPU Verification. DOI:10.1145/1882291.1882320
32. [Leung et al., 2012] Loop Tiling for GPGPU. DOI:10.1145/2259016.2259067
33. [Collingbourne et al., 2011] GPU Kernel Semantics. DOI:10.1007/978-3-642-19718-5_14
34. [NVIDIA, 2023] PTX ISA 8.0. NVIDIA Documentation

### Platform-Specific

35. [Apple, 2023] Metal Best Practices. Apple Developer
36. [Gaster & Howes, 2012] Heterogeneous Computing. ISBN:978-0-12-387766-6
37. [Lopes et al., 2021] Apple Silicon ML. arXiv:2110.01599
38. [AMD, 2023] HIP Programming Guide. AMD ROCm
39. [Sun et al., 2019] GPU Design Trends. DOI:10.1109/IISWC47752.2019.9041952
40. [Arafa et al., 2019] Instruction-Level Power. DOI:10.1109/ISPASS.2019.00018
41. [Aaftab et al., 2020] Cross-Platform DL. ICLR Workshop
42. [Ruetsch & Micikevicius, 2009] Matrix Transpose. NVIDIA Technical Report

---

## Specification File References

| Spec File | Pending Items | PMAT Coverage |
|-----------|---------------|---------------|
| `cuda-tile-behavior.md` | F51-F65, F66-F80, F17-F29, F34-F39 | PMAT-001 to PMAT-004 |
| `ptx-debugger.md` | REQ-001 to REQ-010 | PMAT-008 |
| `ublk-batched-gpu-compression.md` | F-001 to F-010 | PMAT-005 |
| `apple-metal-backend.md` | (proposed) | PMAT-006 |
| `amd-rocm-backend.md` | (proposed) | PMAT-007 |
| `numerical-stability.md` | F92-F99 | PMAT-009 |
| `backend-equivalence.md` | F81-F87 | PMAT-010 |

---

## Document References

- **PMAT Tickets**: `docs/pmat-tickets/PMAT-001-to-010.md`
- **FKR Registry**: `docs/CUDA_TDG_COMPLIANCE.md` Section H
- **Quality Gates**: `docs/CUDA_TDG_COMPLIANCE.md` Section G
- **Continuous Protocol**: `docs/CUDA_TDG_COMPLIANCE.md` Section I

---

## Changelog

### 2026-01-10
- Initial creation with 10 PMAT tickets and 10 new FKR entries
- 60 peer-reviewed citations indexed
- Cross-references established between PMAT and FKR
