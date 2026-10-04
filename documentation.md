# BayesCNN Hardware Accelerator — Architecture Reference Manual

Reference paper: H. Fan, M. Ferianc, Z. Que, S. Liu, X. Niu, M. Rodrigues, W. Luk, *"FPGA-based Acceleration for Bayesian Convolutional Neural Networks"*, IEEE TCAD (`BCNN.pdf` in the repository root). Section, figure, table, algorithm and equation numbers below refer to that paper.

---

## Contents

1. [Executive Architecture Overview](#1-executive-architecture-overview)
2. [Global Configuration & Parameters](#2-global-configuration--parameters-rtlbcnn_pkgvh)
3. [Module-by-Module Walkthrough](#3-module-by-module-walkthrough)
   - [Stage 1 — Memory Subsystem](#stage-1--memory-subsystem-rtlstage1_buffers)
   - [Stage 2 — Bernoulli Sampler](#stage-2--bernoulli-sampler-rtlstage2_sampler)
   - [Stage 3 — Processing Engine](#stage-3--processing-engine-rtlstage3_pe_array)
   - [Stage 4 — Functional & Dropout Engine](#stage-4--functional--dropout-engine-rtlstage4_functional)
   - [Stage 5 — Cache & Reduction Engine](#stage-5--cache--reduction-engine-rtlstage5_cache_reduction)
   - [Master Top-Level](#master-top-level-rtlbcnn_topv)
4. [UAMH Innovations (`rtl/UAMH/`)](#4-uamh-innovations-rtluamh)
   - [Innovation 1 — Uncertainty-Modulated Precision Storage (UMPS)](#41-innovation-1--uncertainty-modulated-precision-storage-umps)
   - [Innovation 2 — Closed-Loop Early-Exit Sample Throttling](#42-innovation-2--closed-loop-early-exit-sample-throttling)
   - [Innovation 3 — Uncertainty-Tagged Cache Lines (U-Tagging)](#43-innovation-3--uncertainty-tagged-cache-lines-u-tagging)
5. [Verification Suite & Results](#5-verification-suite--results)
6. [Build, Simulation & Toolchain Guide](#6-build-simulation--toolchain-guide)
7. [Known Limitations & Deviations from the Paper](#7-known-limitations--deviations-from-the-paper)

---

## 1. Executive Architecture Overview

### 1.1 Summary

This repository implements, in synthesizable Verilog, the Monte Carlo Dropout (MCD) BayesCNN accelerator of Fan et al. The design follows the paper's overview (Fig. 4, Section III-A1): a **Neural Network Engine (NNE)** made of smart data/weight buffers, a processing engine (PE), a functional engine (FE) and a dropout engine (DE), plus a **Bernoulli sampler** that generates dropout masks in the background. On top of the paper's NNE, Stage 5 implements **Intermediate-layer Caching (IC)** (Section IV-B, Fig. 11(c)) and an on-chip **output reducer** that averages the S Monte Carlo samples (Eq. 1) and reports a per-channel variance as the uncertainty estimate.

The default configuration matches the paper's implemented design point (Section VI-A): $P_V = 1$, $P_C = 64$, $P_F = 64$, INT8 linear quantization. The paper's board design ran at 225 MHz on an Intel Arria 10 SX660; the RTL headers and testbenches here use a 220 MHz clock (4.54 ns period).

The network is executed layer by layer on a single NNE (Section III-A2). For a network with N layers of which the last B are Bayesian, the first sample runs all N layers; every subsequent sample skips layers 1..N−B and replays the cached output of layer N−B from on-chip memory with a fresh dropout mask.

Beyond the paper, `rtl/UAMH/` adds three optional innovations around the IC buffer and the Monte Carlo loop ([Section 4](#4-uamh-innovations-rtluamh)):

- **UMPS** stores low-activity channel pairs of the cached layer as two INT4 nibbles in one byte, so the same IC memory holds up to 2× more pixels, losslessly.
- **Early exit** stops the Monte Carlo loop as soon as the running per-channel variance has stopped changing, and finalizes the mean and variance over the samples actually run.
- **U-Tagging** tags every cached line by information content. Under memory pressure, flat lines are spilled to DRAM rather than filling on-chip memory, and spilled lines are brought back in order during replay.

All three are disabled by default (`umps_en = utag_en = early_exit_en = 0`). With them disabled, the accelerator behaves exactly as the paper baseline, cycle for cycle.

### 1.2 Block diagram

```
                         ┌──────────────────────────────────────────────┐
                         │          Off-chip DRAM / Host Bus            │
                         └───────┬──────────────────────────▲───────────┘
            dram_data_in (PC×8b) │ weight_din (PC×PF×8b)    │ layer_features_out (PF×PV×8b)
                                 ▼                          │
 ┌─────────────────────────────────────────────────────────────────────────────────────┐
 │ bcnn_top.v : Layer-level Controller  (ingress → compute → drain → advance)          │
 │                                                                                     │
 │  ┌──────────────────────── STAGE 1: Smart Buffers ───────────────────────┐          │
 │  │ data_ingress_engine ─► Ping/Pong RAM banks (PC×PV) ─► crossbar ─► tree │◄─┐       │
 │  │ read_addr_gen (Alg. 1) ──────┘                          fan-out (×PF)  │  │IC     │
 │  │ weight_fifo ×(PC×PF) ─► tree fan-out (×PV)   (weights recirculate)     │  │replay │
 │  └───────────────┬──────────────────────────────┬────────────────────────┘  │       │
 │      pe_data_out │ (PF×PV×PC×8b)  pe_weight_out │                           │       │
 │                  ▼                              ▼                           │       │
 │  ┌──────────────────── STAGE 3: Processing Engine (PF PUs) ─────────────┐   │       │
 │  │ PU: PC multipliers → log2(PC) adder tree → 32b accumulator →          │   │       │
 │  │     linear quantizer (INT8) → ReLU                                    │   │       │
 │  └───────────────────────────────┬───────────────────────────────────────┘   │       │
 │              pe_features_out     ▼ (PF×PV×8b)                                │       │
 │  ┌──────────── STAGE 4: Functional Engine ───────────┐   ┌──────────────────┐ │       │
 │  │ SC addition → 2D pooling ──┬──► Dropout Engine ───┼──►│ egress to DRAM   │ │       │
 │  │                 premask tap│      O = Y ⊙ M       │   └──────────────────┘ │       │
 │  └────────────────────────────┼──────────▲───────────┘                       │       │
 │                               │          │ mask (PF bits)                    │       │
 │                               │   ┌──────┴────────────────────────┐          │       │
 │                               │   │ STAGE 2: Bernoulli Sampler    │◄─ pop ───┤       │
 │                               │   │ LFSR(s) → AND → SIPO → FIFO   │ (arbit.) │       │
 │                               │   │ runs in the background        │          │       │
 │                               │   └──────┬────────────────────────┘          │       │
 │                               ▼          ▼ mask                              │       │
 │  ┌──────────────── STAGE 5: Cache & Reduction Engine ─────────────────────┐  │       │
 │  │ mc_sample_controller (layer 1..N, sample 1..S, routing flags)          │  │       │
 │  │ ic_buffer (layer N−B, pre-dropout) ─► replay dropout_engine ───────────┼──┘       │
 │  │ output_reducer (layer N of each sample) → mean_prediction,             │          │
 │  │                                            uncertainty_score           │          │
 │  └────────────────────────────────────────────────────────────────────────┘          │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Paper concept → hardware mapping

| Paper concept | Paper reference | Hardware implementation |
|---|---|---|
| Channel parallelism $P_C$ | Table I, Section III-A1/III-A2 | `PC` macro: RAM banks per group, multipliers per PU, adder-tree width (`multiplier_array.v`, `adder_tree.v`, `smart_data_buffer.v`) |
| Vector parallelism $P_V$ | Table I, Section III-A1, Fig. 6/8 | `PV` macro: RAM bank groups, weight fan-out (`smart_weight_buffer.v`), `W_tiles = W/PV` in the RAG |
| Filter parallelism $P_F$ | Table I, Section III-A1, Fig. 6 | `PF` macro: data tree fan-out, PF parallel PUs (`processing_engine.v`), PF-bit masks |
| Monte Carlo Dropout $O_i = Y_i \odot M_i$ | Section II-B1, Eq. 2 | `dropout_engine.v`: filter-wise mask, one bit per output filter |
| Bernoulli sampling with $N_{lfsr}$ LFSRs | Section III-A3, Fig. 5 | `lfsr_128bit.v`, `bernoulli_sampler.v` (AND of `N_LFSR` bits), `sipo_shift_reg.v`, `mask_fifo.v` |
| Overlapping sampling with computation | Section IV-A, Fig. 10 | Sampler always enabled in `bcnn_top.v`; mask FIFO filled ahead of use; pauses when full |
| Partial Bayesian inference ($B \le N$) | Section II-B2 | `mc_sample_controller.v`: `bayesian_layers_B`, MCD only on the inputs of the last B layers |
| Intermediate-layer Caching (IC) | Section IV-B, Fig. 11(c) | `ic_buffer.v`, replay path in `cache_reduction_engine.v`, source mux in `bcnn_top.v` |
| Predictive mean over S samples | Section II-B, Eq. 1 | `output_reducer.v`: `mean_prediction` |
| Uncertainty quantification | Section II-B, Section VI-A (aPE, ECE) | `output_reducer.v`: per-channel variance `uncertainty_score` (hardware proxy; the paper measures aPE/ECE in software) |
| Algorithm 1 loop nest | Section III-B1 | `read_addr_gen.v` (h, w, kl, kh, kw, c loops) |
| Data arrangement (channel-first, row-by-row, frame-by-frame) | Section III-A4, Fig. 7 | `data_ingress_engine.v` and `read_addr_gen.v` address map |
| Shortcut addition | Section III-B2, Fig. 9 | `sc_addition_unit.v` |
| 2D pooling | Section II-A Fig. 3, Section III-B3 | `pooling_unit_2d.v` |
| 8-bit linear quantization | Section III-A2, Section VI-A | `linear_quantizer.v` |
| Double buffering of inputs | Section III-A2, Section V-B | Ping/Pong RAM banks in `smart_data_buffer.v` |
| Weight reuse ("fetched weights flow back to FIFOs") | Section III-A4, Fig. 8 | Weight recirculation in `bcnn_top.v` |

---

## 2. Global Configuration & Parameters (`rtl/bcnn_pkg.vh`)

Every RTL file includes `bcnn_pkg.vh` (compiled with `-I rtl`) and exposes each macro as an overridable module `parameter`. The header is guarded by `` `ifndef BCN_PKG_VH ``.

### 2.1 Stage 1 — buffers and geometry

| Macro | Value | Meaning |
|---|---|---|
| `DATA_WIDTH` | 8 | INT8 activations and weights (Section VI-A) |
| `RAM_DEPTH` | 1024 | Words per RAM bank |
| `ADDR_WIDTH` | 10 | $\lceil\log_2(\text{RAM\_DEPTH})\rceil$ |
| `PC` | 64 | Channel parallelism |
| `PV` | 1 | Vector parallelism |
| `PF` | 64 | Filter parallelism |
| `DIM_WIDTH` | 16 | Width of H, W, L |
| `TILE_CNT_WIDTH` | 10 | Width of `C_tiles` (= C/PC) and `W_tiles` (= W/PV) |
| `KERNEL_DIM_WIDTH` | 4 | Width of KH, KW, KL |
| `STRIDE_WIDTH` | 3 | Width of stride |
| `FIFO_DEPTH` | 512 | Depth of each weight FIFO |

### 2.2 Stage 2 — Bernoulli sampler

| Macro | Value | Meaning |
|---|---|---|
| `LFSR_WIDTH` | 128 | LFSR length ($N_{reg}$ = 128, Section III-A3) |
| `LFSR_TAP1..4` | 127, 125, 100, 98 | Feedback taps → $x^{128}+x^{126}+x^{101}+x^{99}+1$ |
| `N_LFSR` | 1 | Number of LFSRs ANDed; keep probability $p = 1/2^{N_{lfsr}}$ (1 → 50 %) |
| `SIPO_CNT_WIDTH` | 6 | Counts PF = 64 serial bits |
| `MASK_FIFO_DEPTH` | 64 | Mask words buffered |
| `MASK_FIFO_ADDR` | 6 | $\log_2(\text{MASK\_FIFO\_DEPTH})$ |

### 2.3 Stage 3 — processing engine

| Macro | Value | Meaning |
|---|---|---|
| `LOG2_PC` | 6 | Adder-tree depth for PC = 64 |
| `MULT_OUT_WIDTH` | 16 | INT8 × INT8 product |
| `ADDR_TREE_WIDTH` | 22 | 16 + log2(64) bits of adder-tree growth |
| `ACCUM_WIDTH` | 32 | Window accumulator |
| `QUANT_SCALE_WIDTH` | 16 | Signed fixed-point scale |
| `QUANT_SHIFT_WIDTH` | 5 | Right-shift amount (0..31) |
| `QUANT_BIAS_WIDTH` | 32 | Signed bias |

### 2.4 Stage 4 — functional engine

| Macro | Value | Meaning |
|---|---|---|
| `POOL_MODE_WIDTH` | 2 | Pooling mode select |
| `POOL_MODE_BYPASS` / `MAX` / `AVG` | `2'b00` / `2'b01` / `2'b10` | Pooling modes |
| `POOL_WIN_SIZE` | 4 | 2×2 window |
| `POOL_CNT_WIDTH` | 2 | $\log_2(4)$; also the avg-pool divide shift |

### 2.5 Stage 5 — cache & reduction

| Macro | Value | Meaning |
|---|---|---|
| `MAX_SAMPLES` | 100 | Maximum MC samples S |
| `SAMPLE_CNT_WIDTH` | 7 | Holds 0..127 |
| `LAYER_CNT_WIDTH` | 6 | Up to 63 layers |
| `IC_RAM_DEPTH` | 1024 | Words (output pixels) cached for layer N−B |
| `IC_ADDR_WIDTH` | 10 | $\log_2(\text{IC\_RAM\_DEPTH})$ |
| `REDUCER_ACCUM_WIDTH` | 24 | Signed running sum of S INT8 samples |
| `VAR_ACCUM_WIDTH` | 32 | Running sum of squares; also the divider iteration count |

### 2.6 UAMH innovations

| Macro | Value | Meaning |
|---|---|---|
| `INT4_WIDTH` | 4 | Packed precision (UMPS); requires `DATA_WIDTH = 2 × INT4_WIDTH` |
| `UMPS_THRESH_WIDTH` | 8 | Width of the UMPS activity threshold τ |
| `UMPS_DEFAULT_THRESH` | `8'sd16` | Default τ (with τ ≥ 7 only the INT4 range test matters) |
| `PRECISION_MODE_INT8` / `INT4` | `1'b0` / `1'b1` | Per-channel precision metadata values |
| `EARLY_EXIT_THRESH_WIDTH` | 16 | Width of the convergence tolerance ε |
| `DEFAULT_EXIT_THRESH` | `16'd4` | Default ε |
| `MIN_SAMPLES_EXIT` | 4 | Warm-up $S_{min}$ before an early exit is allowed |
| `CONV_STABILITY_COUNT` | 2 | Consecutive stable passes K required to exit |
| `UTAG_WIDTH` | 2 | Uncertainty tag width |
| `UTAG_ZERO` / `LOW` / `HIGH` / `PINNED` | `2'b00` / `01` / `10` / `11` | Tag values |
| `UTAG_ZERO_THRESH` | `8'sd4` | Channels with \|value\| ≤ this count as inactive |
| `UTAG_HIGH_COUNT_TH` | `6'd8` | INT8-precision channels needed for `UTAG_HIGH` (the port is 7 bits wide, since counting 64 channels needs 7) |
| `UTAG_CAP_THRESH_PCT` | 80 | IC occupancy (%) at which admission filtering starts |

### 2.7 Parameterization rules and reconfiguration

- **No hard-coded widths.** Ports, buses and constants are written in terms of these macros or module parameters, e.g. `[(PF*PV*DATA_WIDTH)-1:0]`, `{DATA_WIDTH{1'b0}}`. A channel is always sliced as `bus[f*DATA_WIDTH +: DATA_WIDTH]`; its sign bit is `bus[(f+1)*DATA_WIDTH-1]`. An earlier bug used `bus[f*DATA_WIDTH-1 +: DATA_WIDTH]`, which reads bit −1 (X) for channel 0 and misaligns every other channel. That form is banned.
- **Two levels of reconfiguration**, as in the paper (Section V-A):
  - *Hardware parameters* (synthesis time): `PC`, `PF`, `PV`, `N_LFSR`, memory depths. The paper explores $P_C, P_F \in \{8,16,32,64,128\}$ and $P_V \in \{1,4,8,16\}$.
  - *Run-time configuration* (per layer, from the host): `H`, `W`, `C_tiles`, `W_tiles`, `L_frames`, `KH`, `KW`, `KL`, `stride`, `mode_3d`, `relu_en`, `sc_en`, `pool_mode` and the quantization triplet. Per network: `total_layers_N`, `bayesian_layers_B`, `total_samples_S`.
  - *UAMH configuration* (per inference, latched at `start_inference`): `umps_en`, `umps_thresh`, `utag_en`, `utag_zero_thresh`, `utag_high_count_th`, `early_exit_en`, `early_exit_thresh`, `early_exit_min_samples`.
- **Topologies.** LeNet-5, VGG-11, ResNet-18/34 and 3D CNNs are expressed as sequences of convolution layers with optional ReLU, pooling, shortcut addition and MCD. They differ only in the run-time configuration above and in the number of layers N, as long as each layer fits the current RTL limits in [Section 7](#7-known-limitations--deviations-from-the-paper). In particular the address generator has no filter-tile loop, padding or strided output sizing yet, and pooling assumes consecutive pixels. The verified system configuration is a 3-layer 1×1-convolution network.

---

## 3. Module-by-Module Walkthrough

Pipeline latencies quoted below are in clock cycles. "Issue" means the cycle in which a BRAM read address is presented.

### Stage 1 — Memory Subsystem (`rtl/stage1_buffers/`)

#### 3.1.1 `ram_bank.v`

- **Path:** `rtl/stage1_buffers/ram_bank.v`
- **Purpose:** One simple dual-port BRAM bank holding one channel (`DATA_WIDTH` bits) per word.
- **Paper reference:** Section III-A4 "Smart Data Buffer", Fig. 6 (the $P_C \times P_V$ RAM banks).
- **Interface:** Port A write (`we_a`, `addr_a`, `din_a`); Port B read (`re_b`, `addr_b`, `dout_b`); `clk`, `rst_n`.
- **Datapath:** `mem[RAM_DEPTH]` array. The write is synchronous on Port A. The read is synchronous and registered on Port B, so `dout_b` is valid one cycle after `re_b`. `dout_b` has a synchronous reset to 0.
- **Timing notes:** The registered read port is the single cycle of BRAM latency that `smart_data_buffer.v` compensates for (see 3.1.6). Without a reset on the array, synthesis tools infer block RAM.

#### 3.1.2 `data_ingress_engine.v`

- **Path:** `rtl/stage1_buffers/data_ingress_engine.v`
- **Purpose:** DRAM-to-BRAM DMA. It accepts an AXI-stream-style feature stream, one pixel of PC channels per beat, and writes each beat to its address in the RAM banks.
- **Paper reference:** Section III-A4 (input loaded from off-chip memory and cached in RAM banks before each layer); Section III-B1 and Fig. 7 (data arrangement).
- **Interface:**
  - Inputs: `start_ingress`, geometry (`H`, `W`, `C_tiles`, `W_tiles`, `L_frames`), and the `dram_valid` / `dram_data_in` stream.
  - Outputs: `dram_ready`, `ingress_busy`, `ingress_done`, and the BRAM write bus (`we_a`, `addr_a`, `din_a`).
- **Datapath:**
  - **States:** `STATE_IDLE` → `STATE_WRITE` → `STATE_DONE`.
  - **Counters:** nested counters `c_tile_cnt` (innermost), `w_tile_cnt`, `h_cnt` and `l_cnt` walk the tensor in channel-first, row-by-row, frame-by-frame order (Fig. 7).
  - **Address map:** each accepted beat is written to $\mathcal{M}(l,h,w,c) = ((l \cdot H + h)\cdot W_{tiles} + w_{tile})\cdot C_{tiles} + c_{tile}$.
  - **Write data:** `din_a = {PV{dram_data_in}}`, with all `we_a` bits set.
- **Timing notes:**
  - A beat is accepted when `dram_valid && dram_ready`. `dram_ready` is registered, rises the cycle after `start_ingress`, and drops at the accept edge of the last beat.
  - **Fix:** `ingress_done` used to be set and never cleared, so every later layer would immediately look finished. It is now cleared in `STATE_IDLE`, making it a short strobe (high from the last accept through `STATE_DONE`). The final RAM write lands on the `STATE_DONE` edge, so the data is in BRAM by the time the controller acts on `ingress_done`.

#### 3.1.3 `read_addr_gen.v`

- **Path:** `rtl/stage1_buffers/read_addr_gen.v`
- **Purpose:** Read Address Generator (RAG). It walks the convolution sliding windows and issues one BRAM row read per cycle.
- **Paper reference:** Section III-B1, **Algorithm 1** (loops 3–8: h, w/PV, kl, kh, kw, c/PC); Section III-A4 (the RAG receives kernel size and stride from the controller).
- **Interface:**
  - Inputs: `start_layer`, `mode_3d`, `H`, `W`, `C_tiles`, `W_tiles`, `KH`, `KW`, `KL`, `stride`.
  - Outputs: `read_addr`, `re_b`, `window_done`, `last_window`, `layer_done`.
- **Datapath:**
  - **Loop order:** cascaded counters in Algorithm 1 order, from innermost to outermost: `c_tile_cnt` → `kw_cnt` → `kh_cnt` → `kl_cnt` (3D only) → `w_tile_cnt` → `h_cnt`.
  - **Input coordinates:** `h_in = h_cnt·stride + kh_cnt`, `w_in = w_tile_cnt·PV·stride + kw_cnt` and `l_in = kl_cnt`.
  - **Address:** these are mapped through the same $\mathcal{M}$ as the ingress engine.
  - **States:** `STATE_IDLE` → `STATE_READ` → `STATE_DONE`.
- **Timing notes (fixes):**
  - `read_addr`, `re_b`, `window_done` and `last_window` are registered on the same edge and describe **the same read**.
  - Previously `re_b` rose one cycle before the first address, so the first read returned stale data. It also fell in the same cycle the last address appeared, so the last read was never performed. Both are fixed: `re_b` stays high for exactly the issued reads.
  - `window_done` used to be set unconditionally in `STATE_READ`. It now pulses only on the last read of each window (last c, kw, kh and kl), so the accumulator in Stage 3 closes its windows correctly.
  - The new `last_window` output is high while the reads of the final output window are issued (`w_tile_cnt == W_tiles−1 && h_cnt == H−1`). The top level uses it to drain the weight FIFO at the end of a layer.
  - `layer_done` pulses with the last read issue. The data is still in the pipeline at that point; the top level drains it before advancing the layer.

#### 3.1.4 `crossbar_switch.v`

- **Path:** `rtl/stage1_buffers/crossbar_switch.v`
- **Purpose:** Aligns the outputs of the RAM banks into the order the PE expects.
- **Paper reference:** Section III-A4, Fig. 6 ("together with the crossbar and RAG, the RAM banks are able to output $P_C \times P_V$ data in parallel in a sliding window manner").
- **Interface:** `enable`, `raw_bank_data` → `aligned_data` (`PC×PV×DATA_WIDTH` bits).
- **Datapath:** Combinational. Passes `raw_bank_data` when `enable` is high and drives zero otherwise.
- **Timing notes:** With $P_V = 1$ no lane rotation is needed, so the crossbar is an identity router. `enable` is now the **BRAM-latency-delayed** read strobe (`rag_re_b_d1`), so the gate opens in the cycle the data actually leaves the banks. A full $P_V > 1$ sliding-window rotation is not yet implemented (see Section 6).

#### 3.1.5 `tree_fanout.v`

- **Path:** `rtl/stage1_buffers/tree_fanout.v`
- **Purpose:** Replicates the $P_C \times P_V$ data vector $P_F$ times, so all PF filters see the same input window.
- **Paper reference:** Section III-A4, Fig. 6 ("we use the tree-like fan-out … to simply duplicate the outputs by $P_F$ times").
- **Interface:** `data_in` (`PV×PC×DATA_WIDTH`) → `data_out` (`PF×PC×PV×DATA_WIDTH`).
- **Datapath:** A `generate` loop of PF continuous assignments; slice `f` of `data_out` equals `data_in`. It is purely wiring. In a physical build, register stages can be inserted here for timing, as the paper notes.

#### 3.1.6 `smart_data_buffer.v`

- **Path:** `rtl/stage1_buffers/smart_data_buffer.v`
- **Purpose:** Top-level feature-memory subsystem. It integrates the ingress DMA, the RAG, two Ping/Pong sets of `PC×PV` RAM banks, the crossbar and the tree fan-out.
- **Paper reference:** Section III-A4, Fig. 6; double buffering in Section III-A2 and the memory model in Section V-B ($MEM = 2\times(MEM_{in}+MEM_{weight})+MEM_{FIFO}$).
- **Interface:**
  - Configuration and select: `ping_pong_sel` (0 = write Ping / read Pong, 1 = write Pong / read Ping) and the layer geometry.
  - Ingress side: `start_ingress`, `dram_valid`, `dram_data_in`, `dram_ready`, `ingress_done`.
  - Compute side: `start_compute`, `re_b_valid`, `window_done`, `layer_done`, `read_issue`, `last_window`, `pe_data_out`.
- **Datapath:**
  - The ingress engine drives Port A of the bank set selected for writing; the RAG drives Port B of the other set.
  - `raw_ram_out` is muxed from the set being read, gated by the crossbar, then fanned out PF times.
- **Timing notes (fixes):**
  - `re_b_valid` and `window_done` are now registered one cycle after the RAG (`rag_re_b_d1`, `rag_window_done_d1`), so they line up with `dout_b`. Before this, the PE received each strobe one cycle before its data.
  - `read_issue` (the undelayed `re_b`) and `last_window` are exported aligned with the read *request*. The weight FIFO has a registered pop, so popping on `read_issue` puts the weight word on `pe_weight_out` in the same cycle the data word reaches `pe_data_out`.
  - `layer_done` is passed through undelayed.

#### 3.1.7 `weight_fifo.v`

- **Path:** `rtl/stage1_buffers/weight_fifo.v`
- **Purpose:** One weight FIFO (one INT8 weight per word).
- **Paper reference:** Section III-A4 "Smart Weight Buffer", Fig. 8 (the $P_C \times P_F$ FIFOs).
- **Interface:** `push`, `din`, `full`; `pop`, `dout`, `empty`.
- **Datapath:** Circular buffer with `wr_ptr`, `rd_ptr` and an occupancy `count` of `ADDR_WIDTH+1` bits. Push and pop can happen in the same cycle.
- **Timing notes:** `dout` is **registered on pop**, not first-word-fall-through. The popped word appears the cycle after `pop` and is **held** until the next pop. The top level relies on both properties: it pops one cycle ahead (on `read_issue`), and for a one-step window it pops once and lets `dout` hold the word for the whole layer.

#### 3.1.8 `smart_weight_buffer.v`

- **Path:** `rtl/stage1_buffers/smart_weight_buffer.v`
- **Purpose:** $P_C \times P_F$ parallel weight FIFOs plus a $P_V$ fan-out.
- **Paper reference:** Section III-A4, Fig. 8 ("the smart weight buffer only contains $P_C \times P_F$ FIFOs and a tree-like fan-out … The fetched weights will flow back to FIFOs for data reuse").
- **Interface:** `weight_push`, `weight_din` (`PC×PF×DATA_WIDTH`), `weight_full`; `weight_pop`, `pe_weight_out` (`PV×PC×PF×DATA_WIDTH`), `weight_empty`.
- **Datapath:** FIFO `idx` stores byte `idx` of `weight_din`, where `idx = f·PC + c` (filter f, channel c). This matches the per-filter slicing in `processing_engine.v`. All FIFOs share push and pop. `weight_full` and `weight_empty` are ORs across the FIFOs. The raw bus is replicated PV times.
- **Reuse:** The "flow back" weight reuse of Fig. 8 is implemented around this module in `bcnn_top.v` (see 3.6).

---

### Stage 2 — Bernoulli Sampler (`rtl/stage2_sampler/`)

#### 3.2.1 `lfsr_128bit.v`

- **Path:** `rtl/stage2_sampler/lfsr_128bit.v`
- **Purpose:** 128-bit, 4-tap Fibonacci LFSR producing one pseudo-random bit per enabled cycle.
- **Paper reference:** Section III-A3, Fig. 5 ("4-tap LFSR … $S_{max} = 2^{N_{reg}}-1$ … $N_{reg}$ = 128").
- **Interface:** `en`, `load_seed`, `seed_in` → `lfsr_bit_out`, `lfsr_state_out`.
- **Datapath:**
  - **Feedback:** `feedback = r_state[TAP1] ^ r_state[TAP2] ^ r_state[TAP3] ^ r_state[TAP4]`, using taps 127/125/100/98, i.e. $x^{128}+x^{126}+x^{101}+x^{99}+1$.
  - **Shift:** the state shifts left with the feedback entering bit 0. The output bit is the MSB, `r_state[127]`.
  - **Reset:** the state resets to `DEFAULT_SEED = 128'hACE1_BEEF_CAFE_1234_5678_9ABC_DEF0_1357`.
  - **Seed load:** `load_seed` loads `seed_in`. An all-zero seed is replaced by `DEFAULT_SEED`, because a zero state would lock up the LFSR.
- **Notes:** The tap positions in Fig. 5 (R120–R127) differ from the polynomial used here; the comment in `bcnn_pkg.vh` records the paper's alternative. Because the output is the MSB of a left-shifting register, the first 128 output bits are the seed itself. The first two 64-bit masks after reset are therefore `0xACE1BEEFCAFE1234` and `0x56789ABCDEF01357`, as the verification logs show. For production Monte Carlo runs, load a seed and discard the first 128 bits.

#### 3.2.2 `sipo_shift_reg.v`

- **Path:** `rtl/stage2_sampler/sipo_shift_reg.v`
- **Purpose:** Serial-in, parallel-out converter that turns the 1-bit Bernoulli stream into PF-bit mask words.
- **Paper reference:** Section III-A3 ("we design a serial-in-parallel-out (SIPO) module … to expand the output bitwidth to $P_F$-bits").
- **Interface:** `shift_en`, `bit_in` → `parallel_mask` (PF bits), `word_valid`.
- **Datapath:** `shift_reg <= {shift_reg[PF-2:0], bit_in}`. A `bit_cnt` counts PF shifts; on the PF-th shift, `parallel_mask` captures the full word (the first bit ends up at the MSB) and `word_valid` pulses for one cycle. One mask word is produced every PF = 64 enabled cycles.

#### 3.2.3 `mask_fifo.v`

- **Path:** `rtl/stage2_sampler/mask_fifo.v`
- **Purpose:** Buffers mask words between the sampler and the dropout engines.
- **Paper reference:** Section III-A3 ("a FIFO buffer is placed at the end of the Bernoulli sampler to cache generated Bernoulli random variables and pop out the mask when required"); Section IV-A.
- **Interface:** `push`, `din`, `full`; `pop`, `dout`, `empty`, `occupancy`.
- **Datapath:** Circular buffer with an `ADDR_WIDTH+1`-bit count.
- **Timing notes:** **First-word fall-through (FWFT):** `dout = empty ? 0 : mem[rd_ptr]` is combinational, so the head word is visible *before* it is popped. A consumer must read `dout` and pop in the same cycle; a pop advances to the next word on the following edge. When the FIFO is empty, `dout` reads as all zeros. That is why consumers must check `!empty` (exported as `mask_valid`) before using `dout`: an unchecked read would look like "drop every filter".

#### 3.2.4 `bernoulli_sampler.v`

- **Path:** `rtl/stage2_sampler/bernoulli_sampler.v`
- **Purpose:** Top level of Stage 2 (Fig. 5): $N_{lfsr}$ LFSRs → bit-wise AND → SIPO → FIFO.
- **Paper reference:** Section III-A3, Fig. 5, Table I ($N_{lfsr}$, $p$); Section IV-A (overlapping).
- **Interface:**
  - Inputs: `sampler_en`, `load_seed`, `seed_in`, `mask_pop`.
  - Outputs: `mask_out` (PF bits), `mask_valid`, `mask_empty`, `mask_full`, `mask_count`.
- **Datapath:**
  - **LFSRs:** `N_LFSR` instances, each seeded with `seed_in ^ i` so they decorrelate.
  - **Keep probability:** `bernoulli_bit = &lfsr_bits`, so the keep probability is $p = 1/2^{N_{lfsr}}$. The paper allows $N_{lfsr}$ from 1 to 5, giving a minimum $p = 1/2^5$.
  - **Back-pressure:** the LFSRs and the SIPO advance only when `sampler_en && !mask_full`, so sampling stalls instead of overflowing.
  - **Valid:** `mask_valid = !mask_empty`.
- **Notes:** A mask bit of 1 means **keep** the filter; 0 means drop it. In `bcnn_top.v` the sampler is permanently enabled, which keeps the FIFO ahead of the computation as in Fig. 10.

---

### Stage 3 — Processing Engine (`rtl/stage3_pe_array/`)

#### 3.3.1 `multiplier_array.v`

- **Path:** `rtl/stage3_pe_array/multiplier_array.v`
- **Purpose:** PC parallel signed INT8 × INT8 multipliers.
- **Paper reference:** Section III-A2 ("a MAC unit with $P_C$ multipliers"); Section V-B (DSP model $DSP = P_C P_F P_V / 2$).
- **Interface:** `data_vec`, `weight_vec` (`PC×DATA_WIDTH`) → `prod_vec` (`PC×MULT_OUT_WIDTH`).
- **Datapath:** Combinational `generate` loop. Each lane sign-interprets its bytes and produces a 16-bit signed product.
- **Notes:** The paper packs two 8-bit multipliers into one DSP block, which is where the "/2" in its model comes from. The RTL leaves DSP mapping to the synthesis tool and does not hand-pack multipliers. Explicit packing is a possible later optimization.

#### 3.3.2 `adder_tree.v`

- **Path:** `rtl/stage3_pe_array/adder_tree.v`
- **Purpose:** Balanced binary reduction of the PC products to a single partial sum.
- **Paper reference:** Section III-A2 ("followed by a $\log_2 P_C$-level adder tree").
- **Interface:** `prod_vec` → `sum_out` (`ADDR_TREE_WIDTH` = 22 bits, signed).
- **Datapath:** Six combinational levels (PC → PC/2 → … → 1), each one bit wider than the last (17 … 22 bits). That is exactly enough to hold a sum of 64 signed 16-bit products without overflow.
- **Notes:** The number of levels is written out explicitly for six levels, so the module currently requires $P_C = 64$; the `LOG2_PC` parameter is not yet used to generate the levels. A recursive or generate-by-level tree would be needed for other $P_C$ values (see Section 6).

#### 3.3.3 `mac_unit.v`

- **Path:** `rtl/stage3_pe_array/mac_unit.v`
- **Purpose:** Multiplier array + adder tree with a registered output.
- **Paper reference:** Section III-A2, Fig. 4 ("MAC (PC)").
- **Interface:** `valid_in`, `data_vec`, `weight_vec` → `sum_out`, `valid_out`.
- **Datapath:** The combinational multiply-reduce is followed by one register stage. **Latency: 1 cycle.**

#### 3.3.4 `accumulator_32bit.v`

- **Path:** `rtl/stage3_pe_array/accumulator_32bit.v`
- **Purpose:** Accumulates partial sums across the receptive field (the c/PC, kw, kh and kl loops of Algorithm 1).
- **Paper reference:** Section III-A2 ("an adder tree and an accumulator"); Section III-B1 (channel and kernel loops computed right after the innermost loop).
- **Interface:** `valid_in`, `window_done`, `sum_in` → `accum_out` (32-bit), `accum_valid`.
- **Datapath:** While `valid_in && !window_done`, `running_sum += sum_in`. When `valid_in && window_done`, it outputs `running_sum + sum_in`, pulses `accum_valid`, and clears `running_sum` for the next window. **Latency: 1 cycle** after the closing partial sum.

#### 3.3.5 `linear_quantizer.v`

- **Path:** `rtl/stage3_pe_array/linear_quantizer.v`
- **Purpose:** Maps the 32-bit accumulator back to INT8 for the next layer.
- **Paper reference:** Section III-A2 ("a quantization (Quant) module … to map the accumulated 32-bit results back to 8-bit integers"); Section VI-A (8-bit linear quantization [54]).
- **Interface:** `valid_in`, `accum_in`, `quant_scale` (signed 16), `quant_shift` (5), `quant_bias` (signed 32) → `quant_out`, `valid_out`.
- **Datapath:** In `ACCUM_WIDTH + QUANT_SCALE_WIDTH` = 48-bit precision it computes $y = (acc \cdot scale + (bias \ll shift)) \ggg shift$, then saturates to $[-128, +127]$. **Latency: 1 cycle.**

#### 3.3.6 `relu_unit.v`

- **Path:** `rtl/stage3_pe_array/relu_unit.v`
- **Purpose:** ReLU that can be bypassed.
- **Paper reference:** Section III-A2 ("each PU followed by a ReLU unit which can be optionally bypassed").
- **Interface:** `relu_en`, `data_in` → `data_out`.
- **Datapath:** Registered. Outputs 0 if `relu_en` is set and the sign bit is 1; otherwise passes the value through. **Latency: 1 cycle.** The unit has no valid output of its own (see 3.3.7).

#### 3.3.7 `processing_unit.v`

- **Path:** `rtl/stage3_pe_array/processing_unit.v`
- **Purpose:** Pipeline for one output filter: MAC → accumulator → quantizer → ReLU.
- **Paper reference:** Section III-A2, Fig. 4 (PU).
- **Interface:** `valid_in`, `window_done`, `relu_en`, `data_vec`, `weight_vec`, quantization parameters → `feature_out`, `feature_valid`.
- **Datapath / timing:**
  - `window_done` is delayed one cycle (`window_done_d1`) to match the MAC output register.
  - Total latency from input valid to `feature_valid` is **4 cycles** (MAC 1 + ACC 1 + QUANT 1 + RELU 1).
- **Fix:** `feature_valid` used to be the quantizer's valid, while `feature_out` comes out of the extra ReLU register. Every output pixel was therefore labelled valid one cycle early, i.e. paired with the previous pixel's value. A `relu_valid` register now delays the valid to match the ReLU output.

#### 3.3.8 `processing_engine.v`

- **Path:** `rtl/stage3_pe_array/processing_engine.v`
- **Purpose:** PF parallel PUs, one per output filter.
- **Paper reference:** Section III-A2, Fig. 4 ("In PE, there are … processing units (PUs)"); filter parallelism $P_F$.
- **Interface:** `valid_in`, `window_done`, `relu_en`, `pe_data_in` and `pe_weight_in` (`PF×PV×PC×DATA_WIDTH`), quantization parameters → `pe_features_out` (`PF×PV×DATA_WIDTH`), `features_valid`.
- **Datapath:** PU `f` takes data slice `f` (the identical fanned-out copy) and weight slice `f` (filter f's PC weights). Its output byte becomes channel `f` of `pe_features_out`. All PUs run in lock-step, so `features_valid` is taken from PU 0.

---

### Stage 4 — Functional & Dropout Engine (`rtl/stage4_functional/`)

#### 3.4.1 `sc_addition_unit.v`

- **Path:** `rtl/stage4_functional/sc_addition_unit.v`
- **Purpose:** ResNet shortcut addition, $Out = Conv(X) + X$.
- **Paper reference:** Section III-B2, Fig. 9 (2D/3D SC addition using the SC buffer); Section III-A2 (functional engine).
- **Interface:** `valid_in`, `sc_en`, `conv_features_in`, `sc_features_in` → `sc_features_out`, `valid_out`.
- **Datapath:** Combinational, per channel. A (DATA_WIDTH+1)-bit signed sum is clamped to [−128, +127]. If `sc_en = 0`, the convolution value passes through. **Latency: 0 cycles** (`valid_out = valid_in`).
- **Notes:** The cached input X comes from the `sc_features_in` port. The SC/Pool buffer of Fig. 9 that would produce it lives outside the current top level.

#### 3.4.2 `pooling_unit_2d.v`

- **Path:** `rtl/stage4_functional/pooling_unit_2d.v`
- **Purpose:** 2×2 max pooling, average pooling, or bypass across all PF×PV channels.
- **Paper reference:** Section II-A, Fig. 3(a) (2D max pooling); Section III-B3 (pooling in the functional engine).
- **Interface:** `valid_in`, `pool_mode`, `pool_win_done`, `pool_step`, `features_in` → `pooled_features`, `valid_out`.
- **Datapath:**
  - **Bypass:** register and forward the input.
  - **Max:** `max_regs[i]` is loaded at `pool_step == 0` and updated when a larger value arrives. On `pool_win_done`, it outputs `max(max_regs[i], current)`.
  - **Avg:** `sum_regs[i]` is a (DATA_WIDTH+POOL_CNT_WIDTH)-bit signed accumulator. On `pool_win_done`, it outputs `(sum + current) >>> POOL_CNT_WIDTH`, i.e. the floor of the sum divided by 4.
  - `valid_out` follows `pool_win_done` in the pooling modes. **Latency: 1 cycle.**
- **Fixes:**
  - The average path sliced channels as `features_in[i*DATA_WIDTH-1 +: DATA_WIDTH]`. For channel 0 that starts at bit −1, so the output was **X**, and every other channel was misaligned by one bit. Both the sum and output slices now use `[i*DATA_WIDTH +: DATA_WIDTH]`.
  - The sign-extended operand is now explicitly `$signed`, so the arithmetic shift preserves negative averages.

#### 3.4.3 `dropout_engine.v`

- **Path:** `rtl/stage4_functional/dropout_engine.v`
- **Purpose:** Applies the MCD mask: $O = Y \odot M$.
- **Paper reference:** Section II-B1, **Eq. 2** (a *filter-wise* mask $M_i \in \mathbb{R}^{F_i}$ applied to the output feature map $Y_i$); Section III-A2 ("the DE … creates a mask to randomly drop intermediate output filters"); Section III-A3 ("up to $P_F$ dropout masks in each dropout engine").
- **Interface:**
  - Inputs: `valid_in`, `mcd_en`, `features_in`, `mask_in`, `mask_valid`, `mask_load`.
  - Outputs: `mask_pop`, `masked_features`, `valid_out`.
- **Datapath:**
  - **Pop:** `mask_pop = mask_load && mcd_en && mask_valid` is combinational and fires in the same cycle as `mask_load`. Because the Stage 2 FIFO is FWFT, `mask_in` is the word being consumed.
  - **Latch:** `mask_reg` latches that word and holds it until the next `mask_load`. It resets to all ones (keep everything).
  - **Same-cycle load:** `active_mask = mask_pop ? mask_in : mask_reg`, so a mask loaded in the same cycle as the first pixel applies to that pixel.
  - **Apply:** when `mcd_en` is set, channel `f` is zeroed if `active_mask[f % PF] == 0`. When `mcd_en` is clear, features pass through and no mask is consumed. **Latency: 1 cycle.**
- **Fix (filter-wise latching):** The original engine popped a new mask for **every output pixel**, through a registered `mask_pop` that fired one cycle late. That gave every spatial position its own mask, which contradicts the filter-wise $M_i$ of Eq. 2, and back-to-back pixels could reuse a stale head word. The engine now consumes exactly one mask per `mask_load`: one per PF-filter tile per Monte Carlo sample.

#### 3.4.4 `functional_engine.v`

- **Path:** `rtl/stage4_functional/functional_engine.v`
- **Purpose:** Top level of Stage 4: SC addition → pooling → dropout.
- **Paper reference:** Section III-A2, Fig. 4 (functional engine and dropout engine).
- **Interface:**
  - Inputs: `valid_in`, `sc_en`, `pool_mode`, `pool_win_done`, `pool_step`, `mcd_en`, `conv_features_in`, `sc_features_in`, `mask_in`, `mask_valid`, `mask_load`.
  - Outputs: `mask_pop`, `stage4_features_out`, `stage4_valid_out`, **`premask_features_out`, `premask_valid_out`**.
- **Datapath:** `sc_addition_unit` (0 cycles) → `pooling_unit_2d` (1 cycle) → `dropout_engine` (1 cycle). **Total latency: 2 cycles.**
- **Pre-dropout tap:** `premask_features_out` / `premask_valid_out` expose the pooling output *before* the mask. IC must cache the unmasked layer N−B output so that every later sample can apply its own fresh mask (Section IV-B: "the final result of the second MC sample can be obtained by applying MCD on the cached data"). Caching the masked output would freeze sample 1's mask into every sample.

---

### Stage 5 — Cache & Reduction Engine (`rtl/stage5_cache_reduction/`)

#### 3.5.1 `ic_buffer.v`

- **Path:** `rtl/stage5_cache_reduction/ic_buffer.v`
- **Purpose:** On-chip cache holding the **unmasked** output of layer N−B, captured during sample 1. Lines can have variable length, so that UMPS-compressed pixels really do take less memory.
- **Paper reference:** Section IV-B, Fig. 11(c) (intermediate results cached directly in on-chip memory, unlike the off-chip caching of earlier work).
- **Interface:**
  - Control: `clear` (inference start) and `rd_rewind` (layer boundary).
  - Write side: `wr_en`, `wr_data`, `wr_len` (bytes), `wr_mask` (UMPS precision metadata), `wr_u_tag`.
  - Read side: `rd_en`, `rd_data`, `rd_mask`, `rd_u_tag`, `rd_valid`.
  - Occupancy: `line_count`, `byte_count`, `full`.
- **Datapath:**
  - **Storage:** `PF×PV` byte-wide RAM lanes (`GEN_LANE[k].bank`) of `IC_RAM_DEPTH` rows, addressed as one byte stream. Byte address A lives in lane A mod (PF×PV), row A / (PF×PV).
  - **Back-to-back lines:** lines are appended at a byte write pointer, so a line can span two rows. Every lane still sees one row per cycle, so each lane maps onto its own block RAM. A rotator moves line byte j to lane (start + j) mod PF×PV on write, and back on read.
  - **Per-line side arrays**, sized 2 × `IC_RAM_DEPTH` (the most half-size lines that fit):
    - the length, read asynchronously because it is needed to find the next line's start address;
    - one precision bit per channel pair, read synchronously with the data;
    - the 2-bit U-tag, read synchronously with the data.
  - **Order:** lines are read back in the order they were written. The read is registered, so data appears 1 cycle after `rd_en`.
  - **Full:** `full` asserts when another full-length line would not fit, or when the line arrays are full.
- **Baseline equivalence:** with full-length (64-byte) lines, every line starts on a row boundary and the buffer behaves exactly like the original one-pixel-per-row cache.
- **Simulation view:** a simulation-only array `mem[row]` (excluded under `` `ifdef SYNTHESIS ``) mirrors the lanes row by row, so testbenches can inspect the cache contents.

#### 3.5.2 `mc_sample_controller.v`

- **Path:** `rtl/stage5_cache_reduction/mc_sample_controller.v`
- **Purpose:** Tracks the current layer $l \in [1,N]$ and sample $s \in [1,S]$, and derives the routing flags. It also implements the early-exit hold-and-decide step (Innovation 2).
- **Paper reference:** Section II-B2 (partial Bayesian: the last B layers are Bayesian, the first N−B are a feature extractor); Section IV-B, Fig. 11; Section IV-A, Fig. 10 (layer-by-layer, sample-by-sample execution).
- **Interface:**
  - Inputs: `start_inference`, `layer_done`, and the configuration `total_layers_N`, `bayesian_layers_B`, `total_samples_S`.
  - Early-exit inputs: `early_exit_en`, `eval_done`, `early_exit_trigger`.
  - Progress outputs: `sample_idx`, `layer_idx`, `num_samples`, `busy`, `run_start`.
  - Routing outputs: `ic_write_en`, `ic_read_en`, `bypass_feature_extractor`, `mcd_en`, `is_final_layer`, `inference_done`.
  - Early-exit outputs: `early_exit_active`, `eval_pending`, `samples_executed`, `early_exit_triggered`.
- **Datapath:**
  - **Start:** on `run_start` (`start_inference && !busy && N≠0`) it latches N and B (B clamped to N), and S. S becomes 1 if S = 0 or B = 0, since a non-Bayesian network gives identical passes. It also latches `early_exit_en`.
  - **Layer advance:** on each `layer_done`, if the current layer is not N it increments the layer.
  - **Sample advance:** otherwise, if this was not the last sample, it increments the sample and **jumps to layer N−B+1**. With early exit active, it first raises `eval_pending` and holds the current sample until `eval_done`. It then either finishes early (`early_exit_trigger`) or advances.
  - **Finish:** otherwise it goes idle and pulses `inference_done`.
  - `samples_executed` is updated each time layer N completes, so it ends as $s_{actual}$.
  - Flag equations (`det = N−B`):
    - `ic_write_en = busy && layer == det && sample == 1`
    - `bypass_feature_extractor = busy && sample > 1 && det ≠ 0`
    - `ic_read_en = bypass_feature_extractor && layer == det + 1`
    - `mcd_en = busy && B ≠ 0 && det ≤ layer < N`, i.e. MCD on the outputs of layers N−B..N−1, which are the inputs of the B Bayesian layers
    - `is_final_layer = busy && layer == N`
  - **Savings:** total layer executions are $N + (S-1)\cdot B = (N-B) + B\cdot S$ instead of $N \cdot S$. With early exit, S is replaced by $s_{actual}$.

#### 3.5.3 `output_reducer.v`

- **Path:** `rtl/stage5_cache_reduction/output_reducer.v`
- **Purpose:** Reduces the final-layer output vectors of the Monte Carlo samples into a predictive mean and a per-channel variance. It hosts the `convergence_monitor` for early exit.
- **Paper reference:** Section II-B, **Eq. 1** ($p(D) \approx \frac{1}{S}\sum_{s=1}^{S} p(D\mid w_s)$); Section VI-A (uncertainty and confidence metrics).
- **Interface:**
  - Inputs: `clear`, `sample_valid`, `sample_in`, `sample_idx`, `total_samples_S`.
  - Early-exit inputs: `early_exit_en`, `early_exit_thresh`, `early_exit_min_samples`.
  - Outputs: `mean_prediction` (`PF×PV×DATA_WIDTH`), `uncertainty_score` (`PF×PV×VAR_OUT_WIDTH`, where `VAR_OUT_WIDTH = 2·DATA_WIDTH`), `reduction_done`.
  - Early-exit outputs: `sample_decided`, `early_exit_trigger` (both levels), `variance_delta`.
- **Datapath:**
  - **States:** `RED_ACCUM` → `RED_LOAD` → `RED_DIV` → `RED_FINAL` → `RED_DONE`. Early exit adds `RED_EVAL` → `RED_DECIDE`.
  - **Accumulate:** per channel, `sum += x` (24-bit signed) and `sum_sq += x²` (32-bit).
  - **When to divide:**
    - Baseline: only when the vector of sample S arrives.
    - Early exit: after **every** sample, dividing by the number of samples s seen so far, so the running statistics are always exact.
  - **Division:** each channel has two bit-serial restoring dividers sharing the divisor. They run for `VAR_ACCUM_WIDTH` = 32 cycles, which avoids PF×PV wide combinational dividers. The mean divides the magnitude and the sign is restored afterwards.
  - **Results:** $\text{Mean} = \operatorname{trunc}(\sum x / s)$ and $\text{Var} = \lfloor \sum x^2 / s \rfloor - \text{Mean}^2$.
  - **Early-exit decision:** for s < S, `RED_EVAL` hands the running variance to the monitor and `RED_DECIDE` reads its verdict. "Converged" finalizes the results with $s_{actual} = s$; otherwise accumulation continues. Sample S always finalizes directly.
  - `reduction_done` is set and held until the next `clear`.
- **Precision:** Truncating the mean toward zero guarantees $\text{Mean}^2 \le \lfloor\sum x^2/s\rfloor$, so the variance is never negative. The result is integer-precision; for example, samples {1, 2} give a variance of 1, not 0.25. The variance of INT8 data is at most $2^{14}$, so 16 bits are enough.
- **Uncertainty metric:** The paper evaluates uncertainty with predictive entropy (aPE) and ECE on softmax outputs computed in software. The variance here is a hardware-friendly proxy computed directly on the INT8 logits.
- **Assumption:** each sample contributes exactly one vector, i.e. the final layer outputs one logit vector per sample.

#### 3.5.4 `cache_reduction_engine.v`

- **Path:** `rtl/stage5_cache_reduction/cache_reduction_engine.v`
- **Purpose:** Top level of Stage 5. It integrates the controller, the IC buffer with its write and replay paths, the reducer, and the three UAMH innovations.
- **Paper reference:** Section IV-B, Fig. 11(c); Eq. 1.
- **Interface:**
  - Stage 4 inputs: `premask_features_in` / `premask_valid_in` (cached) and `stage4_features_in` / `stage4_valid_in` (reduced).
  - Replay interface: `ic_rd_req`, `replay_mask_load`, `mask_in`, `mask_valid`, `replay_mask_pop`, `replay_features`, `replay_valid`, `ic_word_count` (lines), `ic_byte_count`, `ic_full`.
  - UAMH control, spill path and telemetry ([Section 4](#4-uamh-innovations-rtluamh)).
  - Outputs: the controller flags and the reduction results.
- **Datapath:**
  - **Mode latch:** `umps_on` and `utag_on` are latched at `run_start`, so every line is decoded and replayed the way it was written.
  - **Write path:**
    - Baseline: `ic_write_en && premask_valid_in` writes the full 64-byte line directly.
    - When UMPS or U-Tagging is on: the line goes through `variance_analyzer` → `umps_packer` / `uncertainty_tagger` (2 cycles), and `u_tag_manager` decides whether to admit it.
  - **Pointers** now live inside `ic_buffer`. Write pointers reset on `run_start`; read pointers reset on `run_start` and on every `layer_done`, so every replay starts at the first line.
  - **Read path:**
    - Baseline: `ic_read_en && ic_rd_req` reads the next line.
    - UMPS: `umps_unpacker` adds 1 cycle.
    - U-Tagging: a sequencer walks the line directory and merges IC lines with lines returned from DRAM, in their original order.
  - **Replay dropout:** a second instance of `dropout_engine` (`u_replay_dropout`) sits on the replay stream, with `mcd_en = ic_read_en`. On `replay_mask_load` it pops a fresh mask, which applies MCD to the cached data for samples 2..S. Baseline replay latency is 2 cycles: BRAM 1 + dropout 1.
  - **Reducer feed:** the reducer receives Stage 4 output when `is_final_layer && stage4_valid_in`, and is cleared on `run_start`.

---

### Master Top-Level (`rtl/bcnn_top.v`)

- **Path:** `rtl/bcnn_top.v`
- **Purpose:** Chip-level integration of Stages 1–5 together with the layer-level **Controller** of Fig. 4.
- **Paper reference:**
  - Section III-A1, Fig. 4 (NNE + Bernoulli sampler + off-chip interface).
  - Section III-A2 (layer-by-layer execution on one NNE).
  - Section III-A4 (weight reuse).
  - Section IV-A (overlapped sampling).
  - Section IV-B (IC).
- **Interface:**
  - Control: `start_inference`, `busy`, `inference_done`.
  - Network configuration: `total_layers_N`, `bayesian_layers_B`, `total_samples_S`.
  - Per-layer geometry: `H`, `W`, `C_tiles`, `W_tiles`, `L_frames`, `KH`, `KW`, `KL`, `stride`, `mode_3d`.
  - Per-layer operators: `relu_en`, `sc_en`, `pool_mode`, `quant_scale`, `quant_shift`, `quant_bias`.
  - Data and weight streams: `dram_data_valid/ready/in`; `weight_push`, `weight_din`, `weight_full`.
  - Other inputs: `sc_features_in`; `load_seed`, `seed_in`.
  - Results: `mean_prediction`, `uncertainty_score`, `reduction_done`.
  - Egress: `layer_features_out`, `layer_features_valid`, `layer_done`.
  - `pool_win_done` and `pool_step` are generated internally rather than taken as ports (see below).
  - UAMH control:
    - UMPS: `umps_en`, `umps_thresh`.
    - U-Tagging: `utag_en`, `utag_zero_thresh`, `utag_high_count_th`.
    - Early exit: `early_exit_en`, `early_exit_thresh`, `early_exit_min_samples`.
  - UAMH outputs:
    - Cache usage: `ic_lines_cached`, `ic_bytes_used`.
    - Spill egress: `ic_spill_data`, `ic_spill_valid`.
    - U-Tagging telemetry: `high_u_cached_count`, `low_u_bypassed_count`, `current_line_u_tag`.
    - Early-exit telemetry: `early_exit_triggered`, `samples_executed`.
    - `eval_pending`: an early-exit decision is pending for the sample that just finished. The host pushes the next layer's weights only after it clears and `busy` is still high, otherwise a cancelled sample would leave a stale weight word behind.
  - Parameters `IC_RAM_DEPTH` / `IC_ADDR_WIDTH` size the IC (defaults: the package macros). A smaller IC models a BRAM-constrained device; `tb_uamh_top.v` uses 8 rows.

**Controller FSM (one pass per layer):**

| State | Action |
|---|---|
| `S_IDLE` | Wait for `start_inference` (Stage 5 latches the configuration on the same edge). |
| `S_LAYER` | If Stage 5 is no longer busy → `S_IDLE`. While Stage 5 has an early-exit decision pending (`eval_pending`), wait, so no further data or weights are fetched for a sample that may be cancelled. Otherwise select the input source: `src_replay = ic_read_en`. For a replay, first wait for a mask in the FIFO, then pulse `replay_mask_load`. Pulse `start_ingress`. |
| `S_INGRESS` | Wait for `ingress_done`. |
| `S_ARM` | For an MCD layer, wait for `mask_valid`. Toggle `ping_pong_sel` so the RAG reads the bank just written, then pulse `start_compute`. The same pulse is Stage 4's `mask_load`. |
| `S_COMPUTE` | Wait for the RAG `layer_done` (last read issued). |
| `S_DRAIN` | Wait `PIPE_DRAIN = PIPE_LATENCY + 1 = 8` cycles. `PIPE_LATENCY = 7` = BRAM 1 + MAC 1 + ACC 1 + QUANT 1 + RELU 1 + POOL 1 + DROPOUT 1. Then pulse `layer_done`. |
| `S_ADVANCE` | Stage 5 advances its layer/sample on this edge → `S_LAYER`. |

**Internal routing:**

1. **Ingress source multiplexer (IC replay).**
   - `src_replay = 0`: Stage 1 ingests `dram_data_in`.
   - `src_replay = 1`, i.e. layer N−B+1 of samples 2..S: Stage 1 ingests `replay_features`. The ingress engine's `dram_ready` becomes `ic_rd_req`. Stage 5 blocks reads beyond the cached word count, so requests in flight past the end are harmless.
   - During a replay layer, `dram_data_ready` reflects only Stage 5's `spill_req`. With U-Tagging off this is always 0, so the host sees no request. With U-Tagging on, the host answers each request with the next spilled line on `dram_data_in`.
2. **Mask-pop arbitration.** `sampler_mask_pop = (ic_read_en && state != S_COMPUTE) ? replay_mask_pop : stage4_mask_pop`. The replay pop happens during the replay ingress and Stage 4's pop happens at the start of compute, so the two never collide. Arbitrating on the replay *phase* rather than on `ic_read_en` alone keeps Stage 4's pop available during the compute of layer N−B+1, which matters when B ≥ 2.
3. **Weight reuse (Fig. 8 "flow back").**
   - `single_step_window = (KH==1 && KW==1 && C_tiles==1 && (!mode_3d || KL==1))`.
   - For a one-step window, the FIFO is popped once at the first read issue (`weight_first`), and the registered `dout` holds the word for the whole layer.
   - Otherwise every `read_issue` pops one word, and the word is pushed back one cycle later (`recirc_push = weight_pop_d1 && !last_window_d1`). The reads of the last window do not recirculate, so the FIFO is empty at the end of the layer, ready for the next layer's weights.
   - `wb_din` is the recirculated word when recirculating and the host's `weight_din` otherwise. The host must not push weights while a layer is computing.
4. **Pooling window sequencer.** `pool_cnt` counts the PE output pixels of the current layer, modulo `POOL_WIN_SIZE`, and resets on `start_compute`. `pool_step = pool_cnt` and `pool_win_done = (pool_cnt == POOL_WIN_SIZE−1)`. These are cycle-level strobes the host cannot drive, so the Controller generates them.
5. **Stage 2** runs with `sampler_en = 1`. Its FIFO fills ahead of use and pauses when full (Fig. 10).
6. **Egress.** `layer_features_out` / `layer_features_valid` are Stage 4's (masked) output and go to DRAM. The host uses them as the next layer's input.

**Width constraint:** a replayed word (`PF×PV×DATA_WIDTH`) is fed into the ingress port (`PC×DATA_WIDTH`), so the design requires $P_F \cdot P_V = P_C$. This holds for the default 64/1/64.

---

## 4. UAMH Innovations (`rtl/UAMH/`)

The baseline caches layer N−B at full INT8 precision, treats every cached pixel the same, and always runs all S Monte Carlo samples. The three UAMH innovations make the IC buffer and the sample loop react to the data, building on the paper's IC (Section IV-B), its memory model (Section V-B: on-chip memory is the limiting resource) and Eq. 1:

| # | Innovation | Saves | Enable | Files |
|---|---|---|---|---|
| 1 | Uncertainty-Modulated Precision Storage (UMPS) | IC memory (up to 2×) | `umps_en` | `variance_analyzer.v`, `umps_packer.v`, `umps_unpacker.v`, `ic_buffer.v` |
| 2 | Closed-Loop Early-Exit Sample Throttling | Monte Carlo passes (latency, DRAM traffic, energy) | `early_exit_en` | `convergence_monitor.v`, `output_reducer.v`, `mc_sample_controller.v` |
| 3 | Uncertainty-Tagged Cache Lines (U-Tagging) | On-chip capacity for informative pixels on small-BRAM devices | `utag_en` | `uncertainty_tagger.v`, `u_tag_manager.v`, `ic_buffer.v`, `cache_reduction_engine.v` |

Common rules:
- **Default off.** Every enable defaults to 0 in the existing testbenches. When off, each innovation's logic is bypassed and timing is identical to the baseline.
- **Latched per inference.** Each enable is latched at `start_inference`, so a cached line is always decoded and replayed the way it was written.
- **Combinable.** UMPS and U-Tagging share the analyzer stage. Early exit is independent of both.

### 4.1 Innovation 1 — Uncertainty-Modulated Precision Storage (UMPS)

**Idea.** Many cached activations are small (after ReLU, most are near zero). A channel that fits the signed INT4 range [−8, +7] loses nothing when stored in 4 bits. When both channels of a pair (2k, 2k+1) fit, they share one byte, so the line shrinks from 64 bytes to as few as 32.

**Why a magnitude test, not a variance test.** Layer N−B is deterministic: its output is identical in every Monte Carlo sample, and it is cached during sample 1, before any sample-to-sample variance exists. The per-pixel channel magnitude is therefore used as the activity proxy. A channel is marked low only if it also fits INT4, so the truncation is **always lossless**; τ (`umps_thresh`) can only make the test stricter.

#### `rtl/UAMH/variance_analyzer.v`
- **Interface:** `valid_in`, `features_in`, `thresh_in` (τ) → `is_low_var` (1 bit per channel), `features_out` (registered copy), `valid_out`.
- **Logic:** `is_low_var[f] = (−8 ≤ val ≤ 7) && (−τ ≤ val ≤ τ)`, evaluated with one guard bit so −128 and −τ never overflow. **Latency:** 1 cycle.

#### `rtl/UAMH/umps_packer.v`
- **Interface:** `valid_in`, `umps_en`, `is_low_var`, `features_in` → `packed_features_out`, `pack_mask_out`, `packed_len_out`, `valid_out`.
- **Logic:** walks the pairs in order k = 0 … 31, appending each at a running byte position:
  - a packed pair emits one byte, `{ch[2k+1][3:0], ch[2k][3:0]}`;
  - any other pair emits `ch[2k]` and `ch[2k+1]` as two bytes.
- `pack_mask_out[2k] = pack_mask_out[2k+1] = 1` for packed pairs, and the line length is 32–64 bytes. With `umps_en = 0` the line passes through at 64 bytes with an all-zero mask. **Latency:** 1 cycle.

#### `rtl/UAMH/umps_unpacker.v`
- **Interface:** `valid_in`, `pack_mask_in`, `packed_features_in` → `unpacked_features_out`, `valid_out`.
- **Logic:** recomputes each pair's byte position from the mask exactly as the packer did. Each nibble is sign-extended (`{{4{n[3]}}, n}`); full bytes are copied. Decoding depends only on the metadata stored with each line, so the unpacker has no enable. **Latency:** 1 cycle.

#### Storage and integration
- **Variable-length storage:** a fixed 512-bit word would save nothing, so `ic_buffer.v` was rebuilt to store variable-length lines back to back (3.5.1). The IC memory size is unchanged; the number of lines it can hold rises up to 2×.
- **Precision metadata:** one bit per pair (32 bits per line) is kept alongside the data.
- **Path:** write `variance_analyzer` → `umps_packer` → `ic_buffer`; read `ic_buffer` → `umps_unpacker` → replay dropout. Both are bypassed when UMPS is off.
- **Telemetry:** `ic_lines_cached` and `ic_bytes_used`; the compression ratio is `ic_lines_cached × 64 / ic_bytes_used`.

### 4.2 Innovation 2 — Closed-Loop Early-Exit Sample Throttling

**Idea.** For an unambiguous input, the predictive distribution settles after a few passes, and running the remaining samples is wasted work. The hardware measures how much the running per-channel variance changes per pass. Once it stays within ε for K consecutive passes (after a warm-up of $S_{min}$), the loop stops and the results are finalized over the samples actually run.

$$\Delta\sigma^2_s = \max_f \left|\sigma^2_s[f] - \sigma^2_{s-1}[f]\right|, \qquad \text{exit when } \Delta\sigma^2 \le \varepsilon \text{ for } K \text{ consecutive passes with } s \ge S_{min}$$

#### `rtl/UAMH/convergence_monitor.v`
- **Interface:**
  - Inputs: `clear`, `sample_valid`, `sample_idx`, `current_variance`, `early_exit_en`, `early_exit_thresh` (ε), `min_samples` ($S_{min}$).
  - Outputs: `early_exit_trigger` (1-cycle pulse), `variance_delta_out` (saturated $\Delta\sigma^2$).
- **Logic:**
  - Keeps the previous pass's variance vector, computes the largest per-channel absolute change, and counts consecutive stable passes.
  - The first pass after `clear` has no predecessor and is never stable.
  - The trigger fires on the pass that brings the count to `CONV_STABILITY_COUNT` (K).

#### Changes in Stage 5
- **`output_reducer.v`:** with early exit on, the exact running mean and variance are recomputed after **every** sample by dividing by s, so a stop at $s_{actual}$ needs no further correction. It then runs the monitor (`RED_EVAL` / `RED_DECIDE`) and reports `sample_decided` and `early_exit_trigger`.
- **`mc_sample_controller.v`:** when layer N of a sample s < S finishes, it raises `eval_pending` and holds the sample until the decision. "Converged" ends the inference at once (`inference_done`, `early_exit_triggered`); otherwise sample s+1 starts.
- **`bcnn_top.v`:** the layer controller waits in `S_LAYER` while `eval_pending` is high, so no weights or replay data are fetched for a sample that may be cancelled.

#### Why the controller waits
The reducer's divider takes about 35 cycles, but the next sample's layer would otherwise start a few cycles after `layer_done`. Waiting costs about 40 cycles per sample and only when early exit is enabled. That is small compared with a layer's compute time, and it is what makes the exit take effect immediately.

**Telemetry:** `samples_executed` ($s_{actual}$) and `early_exit_triggered`.

### 4.3 Innovation 3 — Uncertainty-Tagged Cache Lines (U-Tagging)

**Idea.** On a device whose IC buffer is smaller than the layer N−B map, the baseline silently drops whatever does not fit. U-Tagging grades each line by information content. Under memory pressure, flat background lines are spilled to DRAM so on-chip capacity stays available for informative lines. Every spilled line is brought back during replay, so results are unchanged.

#### `rtl/UAMH/uncertainty_tagger.v`
- **Interface:** `valid_in`, `features_in`, `is_low_var`, `zero_thresh`, `high_count_thresh` → `u_tag_out`, `valid_out`.
- **Logic:** `active` counts channels with \|val\| > `zero_thresh`; `high` counts channels that need INT8 precision.
  - `active == 0` → `UTAG_ZERO` (flat background).
  - otherwise `high ≥ high_count_thresh` → `UTAG_HIGH` (wide dynamic range).
  - otherwise → `UTAG_LOW` (confident, well bounded).
- `UTAG_PINNED` is reserved: it is never generated here and is never filtered. The tagger runs alongside the packer, so its tag is aligned with the packed line.

#### `rtl/UAMH/u_tag_manager.v`
- **Interface:**
  - Inputs: `clear`, `utag_en`, `line_valid_in`, `u_tag_in`, `ic_occupancy`, `ic_capacity`, `ic_full`.
  - Outputs: `admit_to_bram`, `spill_to_dram`, `high_u_cached_count`, `low_u_bypassed_count`.
- **Policy** (combinational; pressure = occupancy ≥ capacity × `UTAG_CAP_THRESH_PCT` / 100):

| Condition | Decision |
|---|---|
| `utag_en = 0` | admit while not full; never spill (baseline) |
| pressure and tag = `UTAG_ZERO` | spill |
| not full | admit |
| full | spill, `UTAG_HIGH` included (resident lines are never evicted) |

- **Units:** occupancy is measured in **bytes**, so UMPS compression is taken into account.
- **Telemetry:** the counters saturate and clear at inference start. `high_u_cached_count` counts admitted HIGH lines; `low_u_bypassed_count` counts spilled ZERO and LOW lines.

#### Spill round trip (in `cache_reduction_engine.v`)
A spilled line has to come back, otherwise replay would give layer N−B+1 fewer pixels than it needs. The DRAM egress of layer N−B cannot be reused, because it already carries sample 1's dropout mask.
1. **Spill:** a line that is not admitted leaves on `ic_spill_data` / `ic_spill_valid` as **unmasked INT8**, in order.
2. **Directory:** a 1-bit-per-line directory (`RAM_DEPTH` entries, since a replayed map must fit Stage 1's data buffer) records whether each line is resident or spilled.
3. **Replay:** a sequencer walks the lines in their original order.
   - Resident lines are read from the IC.
   - For a spilled line, `spill_req` raises `dram_data_ready`, and the host returns the next spilled line on `dram_data_in`.
   - A spilled line is only requested once all earlier IC reads have landed, so order is preserved. All-resident runs keep full speed.
   - Both kinds pass through the replay dropout, so every sample is re-masked as usual.

**Telemetry:** `high_u_cached_count`, `low_u_bypassed_count`, `current_line_u_tag` (tag of the last line replayed from the IC).

**When it matters:** with the default sizes (`IC_RAM_DEPTH` = `RAM_DEPTH` = 1024 rows), the IC can always hold the largest replayable map, so admission filtering never activates. U-Tagging targets builds where `IC_RAM_DEPTH` is set smaller to save BRAM.

---

## 5. Verification Suite & Results

All testbenches are self-checking and print a final pass/fail banner. The Stage 5 and system testbenches tie every UAMH enable to 0, so they double as the "UAMH off = baseline" regression. Stimulus is driven on the **falling** clock edge; driving on the rising edge raced the DUT in early versions and hid timing bugs. The clock is `always #2.27 clk = ~clk` (4.54 ns, about 220 MHz). Every result below comes from a fresh `iverilog` build of the current sources.

| # | Testbench | Test cases | Status | Key empirical result |
|---|---|---|---|---|
| 1 | `tb_stage1_top.v` | (1) DRAM ingress of a 4×4×64 map into Ping; (2) RAG traversal (3×3 kernel) and first read check; (3) weight FIFO push/pop and fan-out | **PASS** (0 errors) | First valid read = pixel (0,0): Ch0 = 0, Ch63 = 63; weight Filter 0/Ch 0 = 5 |
| 2 | `tb_bernoulli_sampler.v` | (1) reset; (2&3) background generation and FIFO fill with back-pressure (`mask_full`); (4) FWFT pop of 5 words; (5) statistics over 4,096 bits; (6) seed reload | **PASS** (0 errors) | Pops in order `0xACE1BEEFCAFE1234`, `0x56789ABCDEF01357`, …, count 63 → 59; **p = 0.5007** (2051 / 4096 ones) |
| 3 | `tb_processing_engine.v` | (1) reset; (2) 9-step window, 64 filters in parallel; (3) negative result with ReLU; (4) ReLU bypass | **PASS** (0 errors) | 576 / 8 = **72** on Filter 0 and Filter 63; ReLU → 0; bypass → **−72** |
| 4 | `tb_functional_engine.v` | (1) reset; (2) SC addition and saturation; (3) 2×2 max pool; (4) 2×2 avg pool; (5) Stage 2 + Stage 4 filter-wise dropout | **PASS** (0 errors) | 20+15 = 35, 100+50 → **127**; max(10,45,30,22) = **45**; avg(12,24,36,48) = **30**; the same mask holds across 2 pixels (50, −30) with exactly 1 pop |
| 5 | `tb_cache_reduction_engine.v` | (1) reset; (2) IC caching at layer N−B, sample 1 (N=3, B=1); (3) IC replay, raw and re-masked; (4) reduction over S=4; (5) second inference N=4, B=2, S=3 | **PASS** (0 errors) | Mean(10,20,30,40) = **25**, Var = **125**; Ch1 = −25/125; all 64 channels bit-exact; trace 1-2-3-4 \| 3-4 \| 3-4 → 8 layers vs 12 |
| 6 | `tb_bcnn_top.v` | (1) reset; (2) layer 1 from DRAM; (3) layer 2 with IC caching and mask 1; (4) samples 2–3 replayed from IC; (5) reduction, layer count, DRAM traffic | **PASS** (0 errors) | **5 layers instead of 9**; **12 DRAM beats instead of 36**; mean and variance **bit-exact on all 64 channels**, all non-zero |
| 7 | `tb_uamh_top.v` | (1) baseline parity, all innovations off; (2) UMPS; (3) U-Tagging with spill and replay; (4) early exit; (5) all three together | **PASS** (0 errors) | UMPS **2.00×**; U-Tagging spills 8 of 16 lines (6 with UMPS) and replays all of them in order; early exit after **7 of 10** and **5 of 10** samples; everything bit-exact against the golden models ([5.4](#54-unified-uamh-system-test-tb_uamh_topv)) |

### 5.1 End-to-end system test (`tb_bcnn_top.v`) in detail

- **Network:** N = 3, B = 1, S = 3. The input is 2×2×64, and each layer is a 1×1 convolution with PF = 64 filters.
  - Layer 1: conv + ReLU.
  - Layer 2: conv + ReLU. This is layer N−B: it is cached in the IC buffer, and MCD is applied to its output.
  - Layer 3: conv + 2×2 average pool, giving one 64-wide logit vector per sample.
- **Host model:** the testbench is the host and DRAM. It streams each layer's input, which is the captured egress of the previous layer, and pushes one weight window per layer execution.
- **Golden model:** a bit-exact reference of conv → quantize → saturate → ReLU → pool → mask runs alongside. It uses the masks actually popped from Stage 2, captured from the DUT hierarchy.
- **Execution trace:** (s1, L1) → (s1, L2) → (s1, L3) → (s2, L3) → (s3, L3). This is $(N-B) + B\cdot S = 2 + 3 = 5$ layer executions instead of $N \cdot S = 9$. IC removed $(N-B)\cdot(S-1) = 4$ layer executions.
- **IC contents:** after layer 2 the IC buffer holds the **unmasked** layer-2 output (4 words, checked word by word), while the DRAM egress carries mask 1 (`0xACE1BEEFCAFE1234`) from Stage 4.
- **Replay samples:** samples 2 and 3 read **0 DRAM beats**. They use fresh masks from the replay dropout engine (`0x56789ABCDEF01357` and `0x986F7787C97A8D67`), with exactly 1 Stage 4 pop and 2 replay pops in total.
- **DRAM traffic:** 12 input beats instead of 36 (3 of 9 layer executions read off-chip).
- **Reduction:** `reduction_done = 1`. The mean and variance match the golden model **bit for bit on all 64 channels**, and every channel has a non-zero mean and variance (e.g. Ch0: mean 6, variance 40; Ch1: mean 9, variance 15). The monitors also check that `ic_write_en` is only ever high at (sample 1, layer N−B) and that `bypass_feature_extractor == (sample > 1)` throughout.

### 5.2 Bugs found and fixed through verification

| Module | Bug | Fix |
|---|---|---|
| `pooling_unit_2d.v` | `[i*DW-1 +: DW]` slicing → X on channel 0 and misalignment elsewhere | `[i*DW +: DW]` plus explicit `$signed` |
| `dropout_engine.v` | New mask per pixel, popped one cycle late | Filter-wise `mask_load` latch, combinational FWFT pop |
| `read_addr_gen.v` | `window_done` always high; `re_b` misaligned with the address; last address never read | Read-aligned strobes; `window_done` only at window end; new `last_window` |
| `smart_data_buffer.v` | Valid and `window_done` one cycle ahead of BRAM data | One-cycle delayed strobes; `read_issue` exported for the weight FIFO |
| `data_ingress_engine.v` | `ingress_done` never cleared | Cleared in IDLE |
| `processing_unit.v` | `feature_valid` one cycle ahead of the ReLU output | `relu_valid` register |
| `functional_engine.v` | No pre-dropout tap for IC | `premask_features_out` / `premask_valid_out` |
| Stage 1/2/4 testbenches | Rising-edge stimulus races; Stage 2 read `mask_out` after the pop (counted empty-FIFO zeros, giving p = 0.23) | Falling-edge stimulus; read the head word, then pop for one cycle and wait for `mask_valid` |

### 5.3 UAMH innovations — regression with the innovations off

`tb_cache_reduction_engine.v` and `tb_bcnn_top.v` tie every UAMH enable to 0, and pass with **0 errors** after each innovation was added. Test 1 of `tb_uamh_top.v` repeats this check on a DUT built with the constrained IC. Together these confirm the bypass paths.

During development, each innovation was also checked on its own at Stage 5 level with standalone simulations kept outside the repository: UMPS line packing (1.28× on mixed data, lines spanning two rows), U-Tagging admission and spill-replay order, and the early-exit decision for ε = 4 and ε = 40. `tb_uamh_top.v` now covers all of these at system level.

### 5.4 Unified UAMH system test (`tb_uamh_top.v`)

**Purpose:** validate the three innovations on the complete chip (`bcnn_top.v`), first individually and then together, against bit-exact golden models.

**DUT:** `bcnn_top #(.IC_RAM_DEPTH(8), .IC_ADDR_WIDTH(3))`, a **512-byte IC** (8 full-length lines) that makes memory pressure reachable.

**Networks:**

| Network | Shape | Layers | Used by |
|---|---|---|---|
| A | N = 3, B = 1, 2×2×64 input | the three 1×1 conv layers of `tb_bcnn_top.v`; layer 3 ends in a global 2×2 average pool. The 4 cached lines always fit the IC | Tests 1, 2, 4 |
| B | N = 4, B = 2, 4×4×64 input | layers 1–2 are identity 1×1 convolutions, so the input image directly sets each of the **16 cached lines**: 6 high-activity (H), 5 low-activity (L), 5 flat (Z), in the order H L Z H L H Z L Z H Z L H Z L H. Layer 3 pools 16 → 4 pixels, layer 4 pools 4 → 1 | Tests 3, 5 |

**Host and DRAM model:**
- Streams each layer's input (the captured egress of the previous layer) and pushes one weight window per layer execution.
- **Spill return:** stores every `ic_spill_valid` beat in `dram_spill_mem` in order. During replay layers, whenever `dram_data_ready` requests a line, it returns the next stored one on `dram_data_in` with `dram_data_valid`.
- **Weight pacing:** after each `layer_done` it waits for `eval_pending` to clear, and pushes the next layer's weights only if `busy` shows another layer will run. A monitor flags any layer whose compute starts before its weights are present.

**Checks** (every test, on top of the per-test criteria below):
1. **Every executed (sample, layer) step** matches a golden conv → quantize → ReLU → pool → MCD model, using the dropout masks actually popped from Stage 2. Masks are captured per (sample, layer) from Stage 4 pops and replay pops.
2. **Every replayed line** equals the unmasked layer N−B output under that sample's replay mask. This covers UMPS unpacking and spilled lines returned from DRAM.
3. **Off-chip traffic per layer:** DRAM input beats for normal layers; spill returns (and no normal DRAM beats) for replay layers.
4. **Telemetry:** `ic_lines_cached`, `ic_bytes_used`, `high_u_cached_count` and `low_u_bypassed_count` match a golden UMPS / U-Tagging admission model.
5. **Early exit:** the exit point, `samples_executed`, `early_exit_triggered`, and the final mean and variance (divided by $s_{actual}$) match a golden convergence model. Exactly one `inference_done` pulse.
6. **Non-degenerate prediction:** at least half of the 64 channels have non-zero variance, so the masks demonstrably shape the output.

**Results** (all PASS, 0 errors):

| Test | Settings | Criteria | Measured |
|---|---|---|---|
| 1 Baseline parity | all off; network A; S = 3 | (N−B) + B·S layers, no spills, no compression, bit-exact | 5 layers (naive 9); 4 lines in 256 bytes; Ch0 mean 6 / var 40; 64/64 channels non-zero |
| 2 UMPS | `umps_en = 1`, τ = 16; network A with small layer-2 activations; S = 3 | ≥ 50 % of cached channels fit INT4; `ic_bytes_used < ic_lines_cached × 64` and ≥ 1.25×; replay bit-exact | 256/256 channels fit INT4; 4 lines in **128 bytes** (**2.00×**); 8 replayed lines unpacked bit-exact |
| 3 U-Tagging | `utag_en = 1`; network B; S = 3 | `low_u_bypassed_count > 0`, `high_u_cached_count > 0`, spills > 0, all lines replayed in order with distinct per-sample masks | 8 lines kept (512 bytes), **8 spilled**; high_u_cached = 3, low_u_bypassed = 5; 32 replayed lines checked (16 per replay sample, 8 of them returned from DRAM); 57/64 channels non-zero |
| 4 Early exit | `early_exit_en = 1`, ε = 40, $S_{min}$ = 4; network A; S = 10 | `early_exit_triggered = 1`, `samples_executed < 10`, `reduction_done` already high at `inference_done`, results exact over $s_{actual}$ | exit after **7 of 10** samples (9 layers instead of 12); Ch0 mean 8 / var 38; 64/64 channels non-zero |
| 5 All three | all on; network B; S = 10 | compression, fewer spills than Test 3, U-Tagging counters active, early exit taken | 10 lines in **454 bytes** (640 uncompressed); **6 spills vs 8** in Test 3; high_u_cached = 4, low_u_bypassed = 4; exit after **5 of 10** samples (12 layers instead of 22) |

**How the data was chosen:** the dropout masks are consecutive 64-bit words of the LFSR stream, consumed in a fixed order. The layer shifts were therefore tuned with a Python model of that exact mask sequence, so that the predictions are non-trivial and Tests 4 and 5 converge before S. The testbench itself verifies against its own Verilog golden models, not against those Python numbers.

**Run time:** about 10–15 minutes on a desktop PC. The full chip has 4,096 weight FIFOs and 4,096 multipliers, which dominate simulation time. The waveform file `sim/uamh_top_simulation.vcd` is about 156 MB; run with `+nodump` to skip it.

---

## 6. Build, Simulation & Toolchain Guide

### 6.1 Prerequisites (Windows)

1. **Icarus Verilog** 11 or later (`iverilog`, `vvp` on `PATH`). The Windows installer from bleyer.org/icarus adds both.
2. **GTKWave** (bundled with the Icarus Windows installer) or **Surfer** (`surfer.exe`) to view waveforms.
3. Run every command from the **repository root** (`BCNN_Accelerator\`). `-I rtl` resolves `` `include "bcnn_pkg.vh" `` and `-g2005-sv` selects the language level used throughout.
4. The `sim\stage1` … `sim\stage5` folders must exist; they are in the repository. Each testbench writes its VCD there.

### 6.2 Compile and run (Windows `cmd`)

**Stage 1 — Smart buffers**
```bat
iverilog -I rtl -g2005-sv -o sim/stage1/stage1_sim.out rtl/stage1_buffers/ram_bank.v rtl/stage1_buffers/tree_fanout.v rtl/stage1_buffers/crossbar_switch.v rtl/stage1_buffers/data_ingress_engine.v rtl/stage1_buffers/read_addr_gen.v rtl/stage1_buffers/weight_fifo.v rtl/stage1_buffers/smart_data_buffer.v rtl/stage1_buffers/smart_weight_buffer.v test_benches/tb_stage1_top.v
vvp sim/stage1/stage1_sim.out
```

**Stage 2 — Bernoulli sampler**
```bat
iverilog -I rtl -g2005-sv -o sim/stage2/stage2_sim.out rtl/stage2_sampler/lfsr_128bit.v rtl/stage2_sampler/sipo_shift_reg.v rtl/stage2_sampler/mask_fifo.v rtl/stage2_sampler/bernoulli_sampler.v test_benches/tb_bernoulli_sampler.v
vvp sim/stage2/stage2_sim.out
```

**Stage 3 — Processing engine**
```bat
iverilog -I rtl -g2005-sv -o sim/stage3/stage3_sim.out rtl/stage3_pe_array/multiplier_array.v rtl/stage3_pe_array/adder_tree.v rtl/stage3_pe_array/mac_unit.v rtl/stage3_pe_array/accumulator_32bit.v rtl/stage3_pe_array/linear_quantizer.v rtl/stage3_pe_array/relu_unit.v rtl/stage3_pe_array/processing_unit.v rtl/stage3_pe_array/processing_engine.v test_benches/tb_processing_engine.v
vvp sim/stage3/stage3_sim.out
```

**Stage 4 — Functional & dropout engine** (uses Stage 2 for real masks)
```bat
iverilog -I rtl -g2005-sv -o sim/stage4/stage4_sim.out rtl/stage2_sampler/lfsr_128bit.v rtl/stage2_sampler/sipo_shift_reg.v rtl/stage2_sampler/mask_fifo.v rtl/stage2_sampler/bernoulli_sampler.v rtl/stage4_functional/sc_addition_unit.v rtl/stage4_functional/pooling_unit_2d.v rtl/stage4_functional/dropout_engine.v rtl/stage4_functional/functional_engine.v test_benches/tb_functional_engine.v
vvp sim/stage4/stage4_sim.out
```

**Stage 5 — Cache & reduction engine** (reuses `dropout_engine.v` for replay; needs the six UAMH modules)
```bat
iverilog -I rtl -g2005-sv -o sim/stage5/stage5_sim.out rtl/stage4_functional/dropout_engine.v rtl/UAMH/variance_analyzer.v rtl/UAMH/umps_packer.v rtl/UAMH/umps_unpacker.v rtl/UAMH/uncertainty_tagger.v rtl/UAMH/u_tag_manager.v rtl/UAMH/convergence_monitor.v rtl/stage5_cache_reduction/ic_buffer.v rtl/stage5_cache_reduction/mc_sample_controller.v rtl/stage5_cache_reduction/output_reducer.v rtl/stage5_cache_reduction/cache_reduction_engine.v test_benches/tb_cache_reduction_engine.v
vvp sim/stage5/stage5_sim.out
```

**Master top — full system** (about 1 minute; the 4,096 weight FIFOs dominate run time)
```bat
iverilog -I rtl -g2005-sv -o sim/bcnn_top_sim.out rtl/stage1_buffers/ram_bank.v rtl/stage1_buffers/tree_fanout.v rtl/stage1_buffers/crossbar_switch.v rtl/stage1_buffers/data_ingress_engine.v rtl/stage1_buffers/read_addr_gen.v rtl/stage1_buffers/weight_fifo.v rtl/stage1_buffers/smart_data_buffer.v rtl/stage1_buffers/smart_weight_buffer.v rtl/stage2_sampler/lfsr_128bit.v rtl/stage2_sampler/sipo_shift_reg.v rtl/stage2_sampler/mask_fifo.v rtl/stage2_sampler/bernoulli_sampler.v rtl/stage3_pe_array/multiplier_array.v rtl/stage3_pe_array/adder_tree.v rtl/stage3_pe_array/mac_unit.v rtl/stage3_pe_array/accumulator_32bit.v rtl/stage3_pe_array/linear_quantizer.v rtl/stage3_pe_array/relu_unit.v rtl/stage3_pe_array/processing_unit.v rtl/stage3_pe_array/processing_engine.v rtl/stage4_functional/sc_addition_unit.v rtl/stage4_functional/pooling_unit_2d.v rtl/stage4_functional/dropout_engine.v rtl/stage4_functional/functional_engine.v rtl/UAMH/variance_analyzer.v rtl/UAMH/umps_packer.v rtl/UAMH/umps_unpacker.v rtl/UAMH/uncertainty_tagger.v rtl/UAMH/u_tag_manager.v rtl/UAMH/convergence_monitor.v rtl/stage5_cache_reduction/ic_buffer.v rtl/stage5_cache_reduction/mc_sample_controller.v rtl/stage5_cache_reduction/output_reducer.v rtl/stage5_cache_reduction/cache_reduction_engine.v rtl/bcnn_top.v test_benches/tb_bcnn_top.v
vvp sim/bcnn_top_sim.out
```

**Unified UAMH testbench** (about 10–15 minutes; `+nodump` skips the 156 MB waveform file)
```bat
iverilog -I rtl -g2005-sv -o sim/uamh_top_sim.out rtl/stage1_buffers/*.v rtl/stage2_sampler/*.v rtl/stage3_pe_array/*.v rtl/stage4_functional/*.v rtl/stage5_cache_reduction/*.v rtl/UAMH/*.v rtl/bcnn_top.v test_benches/tb_uamh_top.v
vvp sim/uamh_top_sim.out
```
If your shell does not expand the `*.v` wildcards, use the explicit file list of the master-top command above with `test_benches/tb_uamh_top.v` in place of `tb_bcnn_top.v`.

Shorter equivalent for the system build: `iverilog -I rtl -g2005-sv -o sim/bcnn_top_sim.out rtl/stage1_buffers/*.v rtl/stage2_sampler/*.v rtl/stage3_pe_array/*.v rtl/stage4_functional/*.v rtl/UAMH/*.v rtl/stage5_cache_reduction/*.v rtl/bcnn_top.v test_benches/tb_bcnn_top.v`. The wildcards are expanded by Git Bash or PowerShell; plain `cmd` needs the explicit list above.

Each run ends with a banner such as `ALL BCNN_TOP SYSTEM TEST CASES PASSED PERFECTLY! (0 ERRORS)`. On failure, it ends with `TESTBENCH COMPLETED WITH n ERRORS.` and prints `[ERROR]` lines naming the channel, word and expected value.

### 6.3 Waveforms

| Testbench | VCD file |
|---|---|
| Stage 1 | `sim/stage1/stage1_simulation.vcd` |
| Stage 2 | `sim/stage2/stage2_simulation.vcd` |
| Stage 3 | `sim/stage3/stage3_simulation.vcd` |
| Stage 4 | `sim/stage4/stage4_simulation.vcd` |
| Stage 5 | `sim/stage5/stage5_simulation.vcd` |
| Top | `sim/bcnn_top_simulation.vcd` |
| UAMH unified | `sim/uamh_top_simulation.vcd` |

```bat
gtkwave sim/stage5/stage5_simulation.vcd
surfer sim/bcnn_top_simulation.vcd
```

Useful signals in the system waveform:
- **Controller:** `dut.state`, `dut.src_replay`, `dut.ping_pong_sel`.
- **Stage 5 indices:** `dut.u_stage5.u_mc_ctrl.sample_idx`, `dut.u_stage5.u_mc_ctrl.layer_idx`.
- **Stage 1 strobes:** `dut.s1_read_issue`, `dut.s1_re_b_valid`, `dut.s1_window_done`.
- **Stage 3 / 4 outputs:** `dut.pe_features_valid`, `dut.premask_valid`, `layer_features_valid`.
- **Masks and IC:** `dut.sampler_mask_pop`, `dut.u_stage5.ic_wr`, `dut.u_stage5.ic_rd`.
- **Results:** `reduction_done`.
- **UAMH:** `dut.u_stage5.umps_on`, `dut.u_stage5.utag_on`, `dut.u_stage5.mgr_admit`, `dut.u_stage5.mgr_spill`, `dut.u_stage5.u_reducer.state`, `dut.s5_eval_pending`, `samples_executed`.

Icarus does not dump memory arrays by default. Read BRAM contents through hierarchical references in a testbench, as `tb_bcnn_top.v` does with `dut.u_stage5.u_ic_buffer.mem[...]`.

### 6.4 Writing new tests

- Drive inputs on `@(negedge clk)` and sample DUT outputs at the falling edge, or inside `always @(posedge clk)` monitors, which see pre-update values.
- The Stage 2 FIFO is FWFT: read `mask_out` first, then pop for exactly one cycle, and only when `mask_valid` is high.
- At the top level, push a layer's weights between `layer_done` and the start of that layer's compute, never during compute.
- Tie unused UAMH inputs to sized constants (e.g. a `localparam [SAMPLE_CNT_WIDTH-1:0]` for `MIN_SAMPLES_EXIT`) to avoid port-width warnings.
- With U-Tagging on, the host must store `ic_spill_data` beats in order and, during replay layers, answer each `dram_data_ready` with the next stored line.

---

## 7. Known Limitations & Deviations from the Paper

| Area | Current behaviour | Paper / general requirement |
|---|---|---|
| Filter tiling | No F/PF loop in the RAG; one layer computes at most PF filters | Algorithm 1 line 2 tiles F by PF |
| Padding and stride | No zero padding; output loop bounds use the input H × W_tiles, so the verified case is 1×1, stride 1 | General convolution geometry |
| Vector parallelism | Ingress writes the same beat to all PV groups; the crossbar is an identity route | Fig. 6/7 distribute pixels across PV bank groups and realign them |
| Adder tree | Six explicit levels, so it requires PC = 64 | $\log_2 P_C$ levels for any $P_C$ |
| DSP packing | Not hand-packed; left to synthesis | Two INT8 multipliers per DSP (Section V-B) |
| Pooling | 4 *consecutive* pixels form one window, which is a true 2×2 window only when the map is 2 pixels wide; 3D pooling and the SC/Pool buffer are not integrated | Section III-B3, Fig. 9 buffer |
| Shortcut source | `sc_features_in` is a top-level port | SC buffer caching the layer input (Fig. 9) |
| Layer drain | Fixed `PIPE_DRAIN` of 8 cycles, matching the current pipeline depth | Must be updated if any pipeline stage is added |
| Reducer | One output vector per sample; integer mean and variance | The paper reports aPE / ECE on softmax probabilities |
| Width coupling | Replay requires PF × PV = PC | — |
| LFSR start-up | The first 128 bits after reset or seed load equal the seed | Load a seed and discard 128 bits for production MC runs |
| Weight push | The host must not push weights while a layer is computing (recirculation shares the push port) | — |
| Clock | 220 MHz in the RTL headers and testbenches | 225 MHz achieved on Arria 10 SX660 (Section VI-A) |
| UMPS geometry | Requires `DATA_WIDTH = 2 × INT4_WIDTH` and an even, power-of-two PF × PV | — |
| UMPS criterion | Per-pixel magnitude, lossless INT4 only (no lossy rescaling) | — |
| U-Tagging eviction | Resident lines are never evicted; a HIGH line arriving at a full IC is spilled (and still replayed correctly) | — |
| U-Tagging spill protocol | The host must accept every spill beat (no back-pressure) and return spilled lines in order | — |
| Early-exit overhead | About 40 cycles per sample spent waiting for the decision, only when enabled | — |
| UAMH verification | `tb_uamh_top.v` covers 1×1 convolutions with up to 4 layers and 16-pixel maps, and a single host spill / return protocol | Larger kernels and maps inherit the RAG and pooling limits above |
