//==============================================================================
// Module: accumulator_32bit.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 3 - Receptive Field Spatial Accumulation
//------------------------------------------------------------------------------
// Purpose:
//   32-bit signed accumulator that integrates partial sums across the 
//   KH x KW x KL convolution sliding window and resets on window_done.
//
// Architectural Inputs:
//   - clk, rst_n    : Clock (220 MHz) and active-low reset.
//   - valid_in      : Valid strobe from MAC adder tree.
//   - sum_in        : ADDR_TREE_WIDTH-bit signed dot-product sum.
//   - window_done   : 1-cycle strobe from RAG signaling window completion.
//
// Architectural Outputs:
//   - accum_out     : ACCUM_WIDTH-bit final accumulated output pixel.
//   - accum_valid   : 1-cycle strobe indicating accum_out is ready for Quant.
//
// Description:
//   Continuously sums incoming channel dot products. When window_done asserts,
//   it latches the complete receptive field sum to accum_out and clears internal
//   state to 0 for the next spatial window.
//==============================================================================

`include "bcnn_pkg.vh"

module accumulator_32bit #(
    parameter ACCUM_WIDTH = `ACCUM_WIDTH,
    parameter ADDR_TREE_WIDTH = `ADDR_TREE_WIDTH
)(
    input wire clk, rst_n, valid_in, window_done,
    input wire signed [ADDR_TREE_WIDTH-1:0] sum_in,

    output reg signed [ACCUM_WIDTH-1:0] accum_out,
    output reg accum_valid
);

    reg signed [ACCUM_WIDTH-1:0] running_sum;

    always @(posedge clk or negedge rst_n) begin 
        if(!rst_n) begin
            running_sum <= {ACCUM_WIDTH{1'b0}};
            accum_out <= {ACCUM_WIDTH{1'b0}};
            accum_valid <= 1'b0;
        end else if(valid_in) begin 
            if(window_done) begin 
                accum_out <= running_sum + sum_in;
                accum_valid <= 1'b1;
                running_sum <= {ACCUM_WIDTH{1'b0}};
            end else begin 
                running_sum <= running_sum + sum_in;
                accum_valid <= 1'b0;
            end
        end else begin 
            accum_valid <= 1'b0;
        end
    end
endmodule