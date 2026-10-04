//==============================================================================
// Package: bcnn_pkg.vh
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Target Architecture: Intel Arria 10 / Xilinx UltraScale FPGA
//------------------------------------------------------------------------------
// Purpose:
//   Master header file containing global macros, parallelism parameters, 
//   memory depths, and parameterizable bit-width specifications.
//
// Architectural Scope:
//   - Configures Channel (PC), Vector (PV), and Filter (PF) parallelism.
//   - Defines dimension widths to ensure zero hardcoded bit-widths in RTL.
//==============================================================================

`ifndef BCN_PKG_VH
`define BCN_PKG_VH

// Stage 1
`define DATA_WIDTH 8
`define RAM_DEPTH 1024
`define ADDR_WIDTH 10 // $clog2(RAM_DEPTH=1024)

`define PC 64 
`define PV 1
`define PF 64  

`define DIM_WIDTH 16  // Width for H, W
`define TILE_CNT_WIDTH  10   // Width for C_tiles, W_tiles, F_tiles - actually derived C/Pc and W/Pw
`define KERNEL_DIM_WIDTH  4  // Width for KH, KW, KL
`define STRIDE_WIDTH  3      // Width for Stride

`define FIFO_DEPTH 512

// Stage 2
`define LFSR_WIDTH 128
`define LFSR_TAP1 127 // 4-Tap Polynomial: x^128 + x^126 + x^125 + x^120 + 1 - in paper
`define LFSR_TAP2 125 
`define LFSR_TAP3 100 // Polynomial used here: // Primitive Polynomial: x^128 + x^126 + x^101 + x^99 + 1
`define LFSR_TAP4 98
`define N_LFSR 1  // 50 PERCENT PROB
`define SIPO_CNT_WIDTH 6  // count till 64
`define MASK_FIFO_DEPTH 64 
`define MASK_FIFO_ADDR 6  // 2^6 addresses in MASK_FIFO

//stage 3
`define LOG2_PC 6  // $clog2(PC) -> log2(64) = 6 levels
`define MULT_OUT_WIDTH 16
`define ADDR_TREE_WIDTH 22  // 16 bit + log2(64)
`define ACCUM_WIDTH 32
`define QUANT_SCALE_WIDTH 16   // 16-bit fixed-point quantization scale multiplier
`define QUANT_SHIFT_WIDTH 5
`define QUANT_BIAS_WIDTH 32  

//Stage 4
`define POOL_MODE_WIDTH 2    // 2-bit mode: 00=Bypass, 01=Max Pool, 10=Avg Pool
`define POOL_MODE_BYPASS 2'b00
`define POOL_MODE_MAX 2'b01
`define POOL_MODE_AVG 2'b10
`define POOL_WIN_SIZE 4    // 2x2 spatial pooling window (4 pixels)
`define POOL_CNT_WIDTH 2   // $clog2(POOL_WIN_SIZE) = 2 bits (0 to 3)

// Stage 5: Cache Reduction & Uncertainty Parameters

`define MAX_SAMPLES 100         // Maximum Monte Carlo samples S
`define SAMPLE_CNT_WIDTH 7           // $clog2(MAX_SAMPLES) (0 to 100)
`define LAYER_CNT_WIDTH 6           // Supports up to 64 layers N
`define IC_RAM_DEPTH 1024        // Depth of Intermediate-Layer Cache BRAM
`define IC_ADDR_WIDTH 10          // $clog2(IC_RAM_DEPTH)
`define REDUCER_ACCUM_WIDTH 24          // Accumulator width for S samples: DATA_WIDTH + SAMPLE_CNT_WIDTH + margin
`define VAR_ACCUM_WIDTH 32          // Variance sum-of-squares accumulator width

// -------------------------------------------------------------
// UAMH Innovation 1: UMPS Parameters
// -------------------------------------------------------------
`define INT4_WIDTH 4 // Truncated precision width
`define UMPS_THRESH_WIDTH 8 // Threshold bit-width
`define UMPS_DEFAULT_THRESH 8'sd16 // Default variance/activity threshold (INT8)
`define PRECISION_MODE_INT8 1'b0 // Full precision mode
`define PRECISION_MODE_INT4 1'b1 // Packed 4-bit precision mode

// -------------------------------------------------------------
// UAMH Innovation 3: U-Tagging Parameters
// -------------------------------------------------------------
`define UTAG_WIDTH 2 // 2-bit Uncertainty Tag
`define UTAG_ZERO 2'b00 // Inactive / Zero-activity line
`define UTAG_LOW 2'b01 // Confident / Low-uncertainty line
`define UTAG_HIGH 2'b10 // Ambiguous / High-uncertainty line
`define UTAG_PINNED 2'b11 // Safety-critical / Pinned line
`define UTAG_ZERO_THRESH 8'sd4 // Absolute magnitude below which a channel is considered zero
`define UTAG_HIGH_COUNT_TH 6'd8 // Number of active/high channels required to classify line as UTAG_HIGH
`define UTAG_CAP_THRESH_PCT 80 // Percentage of IC buffer occupancy to trigger admission filtering (80%)

// -------------------------------------------------------------
// UAMH Innovation 2: Early-Exit Convergence Parameters
// -------------------------------------------------------------
`define EARLY_EXIT_THRESH_WIDTH 16 // Variance delta threshold bit-width
`define DEFAULT_EXIT_THRESH 16'd4 // Default convergence variance tolerance epsilon
`define MIN_SAMPLES_EXIT 4 // Minimum mandatory samples before early exit (S_min >= 2 for variance)
`define CONV_STABILITY_COUNT 2 // Number of consecutive stable passes required to exit (K)

`endif