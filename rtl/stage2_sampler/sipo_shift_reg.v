//==============================================================================
// Module: sipo_shift_reg.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 2 - Serial-to-Parallel Conversion
//------------------------------------------------------------------------------
// Purpose:
//   Serial-In Parallel-Out (SIPO) shift register that converts a 1-bit/cycle 
//   random stream into PF-bit (64-bit) parallel mask vectors.
//
// Architectural Inputs:
//   - clk, rst_n    : Clock (220 MHz) and active-low reset.
//   - shift_en      : Enable signal indicating a valid bit is arriving.
//   - bit_in        : 1-bit random input from LFSR / Bit-wise logic.
//
// Architectural Outputs:
//   - parallel_mask : PF-bit (64-bit) assembled dropout mask word.
//   - word_valid    : 1-cycle strobe pulse indicating a complete 64-bit word is ready.
//
// Description:
//   Collects 64 sequential random bits over 64 clock cycles, then latches the 
//   assembled word into parallel_mask and fires word_valid to push into FIFO.
//==============================================================================

`include "bcnn_pkg.vh"

module sipo_shift_reg #(
    parameter PF            = `PF,
    parameter CNT_WIDTH     = `SIPO_CNT_WIDTH
)(
    input clk,rst_n,shift_en,bit_in,

    output re[PF-1:0] parallel_mask,
    
);



endmodule