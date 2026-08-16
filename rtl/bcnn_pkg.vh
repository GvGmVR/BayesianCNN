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

`define DATA_WIDTH 8
`define RAM_DEPTH 1024
`define ADDR_WIDTH = 10 // $clog2(RAM_DEPTH=1024)

`define PC 64 
`define PV 1
`define PF 64  

`define DIM_WIDTH 16  // Width for H, W
`define TILE_CNT_WIDTH  10   // Width for C_tiles, W_tiles, F_tiles - actually derived C/Pc and W/Pw
`define KERNEL_DIM_WIDTH  4  // Width for KH, KW, KL
`define STRIDE_WIDTH  3      // Width for Stride

`define FIFO_DEPTH 512

`endif