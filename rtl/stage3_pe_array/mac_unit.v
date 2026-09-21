//==============================================================================
// Module: mac_unit.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 3 - Parallel Convolution Dot-Product
//------------------------------------------------------------------------------
// Purpose:
//   Executes PC parallel signed 8-bit multiplications followed by a 6-level
//   balanced binary adder tree to compute a 64-channel dot product in 1 cycle.
//
// Architectural Inputs:
//   - clk, rst_n    : Clock (220 MHz) and active-low reset.
//   - valid_in      : Valid strobe from memory buffer.
//   - data_vec      : PC parallel 8-bit signed input activations (512 bits).
//   - weight_vec    : PC parallel 8-bit signed filter weights (512 bits).
//
// Architectural Outputs:
//   - sum_out       : ADDR_TREE_WIDTH-bit (22-bit) signed accumulated channel sum.
//   - valid_out     : 1-cycle strobe indicating valid sum_out.
//
// Description:
//   Implements PC signed multipliers that map to FPGA DSP blocks via DSP packing,
//   summing products through a log2(PC)-stage tree with proper bit growth.
//==============================================================================

`include "bcnn_pkg.vh"

module mac_unit #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC = `PC
    parameter MULT_OUT_WIDTH = `MULT_OUT_WIDTH
    parameter ADDR_TREE_WIDTH = `ADDR_TREE_WIDTH
)(
    input wire clk,rst_n,valid_in,
    input wire [(PC*DATA_WIDTH-1):0] data_vec,
    input wire [(PC*DATA_WIDTH-1):0] weight_vec,

    output reg signed [ADDR_TREE_WIDTH-1:0] sum_out,
    output reg valid_out
);

    // Step 1: PC Parallel Signed Multipliers (8-bit x 8-bit = 16-bit)
    wire signed [DATA_WIDTH-1:0] in_signed [PC-1:0];
    wire signed [DATA_WIDTH-1:0] w_signed [PC-1:0];
    wire signed [MULT_OUT_WIDTH-1:0] prod_lvl0 [PC-1:0];

    genvar m;
    generate
        for(m=0; m<PC; m=m+1) begin: GEN_MULTIPLIERS
            assign in_signed[m] = data_vec[(m+1)*DATA_WIDTH-1:m*DATA_WIDTH];
            assign in_signed[m] = weight_vec[(m+1)*DATA_WIDTH-1:m*DATA_WIDTH];
            assign prod_lvl0[m] = in_signed[m]*w_signed[m];
        end
    endgenerate

    // Step 2: 6-Level Balanced Binary Adder Tree (64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)
    
    // Level 1: 64 terms (16-bit) -> 32 sums (17-bit)
    wire signed [MULT_OUT_WIDTH:0] sum_lvl1 [31:0];
    genvar l1;
    generate
        for(l1=0;l1<32;l1=l1+1) begin : GEN_ADDR_LVL1
            assign sum_lvl1[l1] = prod_lvl0[2*l1] + prod_lvl0[2*l1 + 1];
        end
    endgenerate

     // Level 2: 32 terms (17-bit) -> 16 sums (18-bit)
    wire signed [MULT_OUT_WIDTH+1:0] sum_lvl2 [15:0];
    genvar l2;
    generate
        for (l2 = 0; l2 < 16; l2 = l2 + 1) begin : GEN_ADDER_LVL2
            assign sum_lvl2[l2] = sum_lvl1[2*l2] + sum_lvl1[2*l2 + 1];
        end
    endgenerate

    // Level 3: 16 terms (18-bit) -> 8 sums (19-bit)
    wire signed [MULT_OUT_WIDTH+2:0] sum_lvl3 [7:0];
    genvar l3;
    generate
        for (l3 = 0; l3 < 8; l3 = l3 + 1) begin : GEN_ADDER_LVL3
            assign sum_lvl3[l3] = sum_lvl2[2*l3] + sum_lvl2[2*l3 + 1];
        end
    endgenerate

    // Level 4: 8 terms (19-bit) -> 4 sums (20-bit)
    wire signed [MULT_OUT_WIDTH+3:0] sum_lvl4 [3:0];
    genvar l4;
    generate
        for (l4 = 0; l4 < 4; l4 = l4 + 1) begin : GEN_ADDER_LVL4
            assign sum_lvl4[l4] = sum_lvl3[2*l4] + sum_lvl3[2*l4 + 1];
        end
    endgenerate

    // Level 5: 4 terms (20-bit) -> 2 sums (21-bit)
    wire signed [MULT_OUT_WIDTH+4:0] sum_lvl5 [1:0];
    genvar l5;
    generate
        for (l5 = 0; l5 < 2; l5 = l5 + 1) begin : GEN_ADDER_LVL5
            assign sum_lvl5[l5] = sum_lvl4[2*l5] + sum_lvl4[2*l5 + 1];
        end
    endgenerate

    // Level 6: 2 terms (21-bit) -> 1 final sum (22-bit)
    wire signed [ADDR_TREE_WIDTH-1:0] tree_final_sum;
    assign tree_final_sum = sum_lvl5[0] + sum_lvl5[1];

    // Step 3: Registered Output Latch
endmodule