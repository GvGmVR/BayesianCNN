//==============================================================================
// Module: linear_quantizer.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 3 - Fixed-Point Linear Quantization
//------------------------------------------------------------------------------
// Purpose:
//   Scales, biases, shifts, and symmetrically clamps 32-bit accumulated sums
//   back down to signed DATA_WIDTH-bit integers.
//
// Architectural Inputs:
//   - clk, rst_n      : Clock (220 MHz) and active-low reset.
//   - valid_in        : Valid strobe from Accumulator.
//   - accum_in        : ACCUM_WIDTH-bit (32-bit) signed unquantized input.
//   - quant_scale     : QUANT_SCALE_WIDTH-bit (16-bit) fixed-point scale factor.
//   - quant_shift     : QUANT_SHIFT_WIDTH-bit (5-bit) right-shift amount.
//   - quant_bias      : QUANT_BIAS_WIDTH-bit (32-bit) layer bias value.
//
// Architectural Outputs:
//   - quant_out       : DATA_WIDTH-bit signed clamped output (INT8: [-128, 127]).
//   - valid_out       : 1-cycle strobe indicating valid quant_out.
//
// Description:
//   Implements parameterized dynamic saturation bounds (INT_MAX / INT_MIN)
//   derived directly from DATA_WIDTH with zero hardcoded bit constants.
//==============================================================================

`include "bcnn_pkg.vh"

module linear_quantizer #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter ACCUM_WIDTH = `ACCUM_WIDTH,
    parameter QUANT_SCALE_WIDTH = `QUANT_SCALE_WIDTH,
    parameter QUANT_SHIFT_WIDTH = `QUANT_SHIFT_WIDTH,
    parameter QUANT_BIAS_WIDTH = `QUANT_BIAS_WIDTH
)(
    input wire clk,rst_n,valid_in,
    input wire signed [ACCUM_WIDTH-1:0] accum_in,
    input wire signed [QUANT_SCALE_WIDTH-1:0] quant_scale,
    input wire [QUANT_SHIFT_WIDTH-1:0] quant_shift,
    input wire signed [QUANT_BIAS_WIDTH-1:0] quant_bias,

    output reg signed [DATA_WIDTH-1:0] quant_out,
    output reg valid_out
);

    localparam PROD_WIDTH =ACCUM_WIDTH + QUANT_SCALE_WIDTH;

    localparam signed [DATA_WIDTH-1:0] INT_MAX = {1'b0,{(DATA_WIDTH-1){1'b1}}}; //+127
    localparam signed [DATA_WIDTH-1:0] INT_MIN = {1'b1,{(DATA_WIDTH-1){1'b0}}}; //-128

    // Step 1: Scale multiplication
    wire signed [PROD_WIDTH-1:0] scaled_val;
    assign scaled_val = accum_in * quant_scale;

    // Step 2: Add bias
    wire signed [PROD_WIDTH-1:0] biased_val;
    assign biased_val = scaled_val + ($signed(quant_bias)<<<quant_shift);

    // Step 3: Arithmetic right shift
    wire signed [PROD_WIDTH-1:0] shifted_val;
    assign shifted_val = biased_val >>> quant_shift;

    always @(posedge clk or negedge rst_n) begin 
        if(!rst_n) begin 
            quant_out <= {DATA_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if(valid_in) begin
            valid_out <= 1'b1;

            //Dynamic stauration
            if(shifted_val > $signed({{PROD_WIDTH-DATA_WIDTH{1'b0}}, INT_MAX})) begin 
                quant_out <= INT_MAX;
            end else if (shifted_val < $signed({{PROD_WIDTH-DATA_WIDTH{1'b1}}, INT_MIN})) begin 
                quant_out <= INT_MIN;
            end else begin 
                quant_out <= shifted_val[DATA_WIDTH-1:0];
            end
        end else begin 
            valid_out <= 1'b0;
        end
    end
endmodule