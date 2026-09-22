//==============================================================================
// Module: mac_unit.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 3 - Top MAC Subsystem (Multiplier Array + Adder Tree)
//------------------------------------------------------------------------------
// Purpose:
//   Top wrapper integrating multiplier_array and adder_tree into a single MAC block.
//
// Architectural Inputs:
//   - clk, rst_n    : Clock (220 MHz) and active-low reset.
//   - valid_in      : Valid strobe from Stage 1 buffers.
//   - data_vec      : PC parallel input feature bytes (PC x DATA_WIDTH bits).
//   - weight_vec    : PC parallel filter weight bytes (PC x DATA_WIDTH bits).
//
// Architectural Outputs:
//   - sum_out       : Registered ADDR_TREE_WIDTH-bit (22-bit) signed sum.
//   - valid_out     : 1-cycle strobe indicating sum_out is valid.
//
// Description:
//   Instantiates multiplier_array and adder_tree, registering the tree's final
//   sum to ensure high clock timing closure.
//==============================================================================

`include "bcnn_pkg.vh"

module mac_unit #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC = `PC,
    parameter LOG2_PC = `LOG2_PC,
    parameter MULT_OUT_WIDTH = `MULT_OUT_WIDTH,
    parameter ADDR_TREE_WIDTH = `ADDR_TREE_WIDTH
)(
    input wire clk,rst_n,valid_in,
    input wire [(PC*DATA_WIDTH-1):0] data_vec,
    input wire [(PC*DATA_WIDTH-1):0] weight_vec,

    output reg signed [ADDR_TREE_WIDTH-1:0] sum_out,
    output reg valid_out
);

    //INternal bus between Multiplier array abd Adder tree
    wire [(PC*MULT_OUT_WIDTH-1):0] raw_prod_vec;
    wire signed [ADDR_TREE_WIDTH-1:0] tree_sum_comb;

    // 1. Parallel Multiplier Array (PC Multipliers)
    multiplier_array #(
        .DATA_WIDTH(DATA_WIDTH),
        .PC(PC),
        .MULT_OUT_WIDTH(MULT_OUT_WIDTH)
    ) u_mult_array (
        .data_vec(data_vec),
        .weight_vec(weight_vec),
        .prod_vec(raw_prod_vec)
    );

    // 2. Parameterized Log2-Level Binary Adder Tree
    adder_tree #(
        .PC(PC),
        .LOG2_PC(LOG2_PC),
        .IN_WIDTH(MULT_OUT_WIDTH),
        .OUT_WIDTH(ADDR_TREE_WIDTH)
    ) u_adder_tree (
        .prod_vec(raw_prod_vec),
        .sum_out(tree_sum_comb)
    );

    // 3. Registered Output Latch
    always @(posedge clk or negedge rst_n) begin 
        if(!rst_n) begin 
            sum_out <= {ADDR_TREE_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else begin 
            sum_out <= tree_sum_comb;
            valid_out <= valid_in;
        end
    end
    
endmodule