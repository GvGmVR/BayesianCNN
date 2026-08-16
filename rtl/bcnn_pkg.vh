//bcnn_pkg.vh

`ifndef BCN_PKG_VH
`define BCN_PKG_VH

`define DATA_WIDTH 8
`define RAM_DEPTH 1024
parameter ADDR_WIDTH = 10 // $clog2(RAM_DEPTH=1024)

`define PC 64 
`define PV 1
`define PF 64  

`define DIM_WIDTH 16  // Width for H, W
`define TILE_CNT_WIDTH  10   // Width for C_tiles, W_tiles, F_tiles - actually derived C/Pc and W/Pw
`define KERNEL_DIM_WIDTH  4  // Width for KH, KW, KL
`define STRIDE_WIDTH  3      // Width for Stride

`endif