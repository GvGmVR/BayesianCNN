//==============================================================================
// Module: processing_unit.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 3 - Single Filter Compute Pipeline (PU)
//------------------------------------------------------------------------------
// Purpose:
//   Computes 2D/3D convolution for 1 output filter across PC channels, 
//   accumulates across the sliding window, and applies Quantization and ReLU.
//
// Architectural Inputs:
//   - clk, rst_n      : Clock (220 MHz) and active-low reset.
//   - valid_in        : Input valid strobe from Stage 1 buffers.
//   - window_done     : Strobe from RAG indicating end of sliding window.
//   - relu_en         : 1-bit flag to enable or bypass ReLU activation.
//   - data_vec        : PC parallel input feature bytes (PC x DATA_WIDTH bits).
//   - weight_vec      : PC parallel filter weight bytes (PC x DATA_WIDTH bits).
//   - quant_scale     : 16-bit fixed-point scale factor.
//   - quant_shift     : 5-bit right shift amount.
//   - quant_bias      : 32-bit layer bias value.
//
// Architectural Outputs:
//   - feature_out     : 8-bit signed quantized and activated output feature pixel.
//   - feature_valid   : 1-cycle strobe indicating feature_out is valid.
//
// Description:
//   Integrates mac_unit -> accumulator_32bit -> linear_quantizer -> relu_unit 
//   into a unified single-filter processing element (Section III-A2, Fig. 4).
//==============================================================================

`include "bcnn_pkg.vh"

module processing_unit #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC = `PC,
    parameter LOG2_PC = `LOG2_PC,
    parameter MULT_OUT_WIDTH = `MULT_OUT_WIDTH,
    parameter ADDR_TREE_WIDTH = `ADDR_TREE_WIDTH,
    parameter ACCUM_WIDTH = `ACCUM_WIDTH,
    parameter QUANT_SCALE_WIDTH = `QUANT_SCALE_WIDTH,
    parameter QUANT_SHIFT_WIDTH = `QUANT_SHIFT_WIDTH,
    parameter QUANT_BIAS_WIDTH  = `QUANT_BIAS_WIDTH
)(
    input wire clk,rst_n,valid_in,window_done,relu_en,

    input wire [PC*DATA_WIDTH-1:0] data_vec,
    input wire [PC*DATA_WIDTH-1:0] weight_vec,

    //quantization parameters
    input wire signed [QUANT_SCALE_WIDTH-1:0] quant_scale,
    input wire [QUANT_SHIFT_WIDTH-1:0] quant_shift,
    input wire signed [QUANT_BIAS_WIDTH-1:0] quant_bias,

    // Final Processed Output (1 Filter Pixel)
    output wire signed [DATA_WIDTH-1:0] feature_out,
    output wire feature_valid
);
    wire signed [ADDR_TREE_WIDTH-1:0] mac_sum;
    wire mac_valid;

    reg window_done_d1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_done_d1 <= 1'b0;
        end else begin
            window_done_d1 <= window_done;
        end
    end

    wire signed [ACCUM_WIDTH-1:0] accum_val;
    wire accum_valid;

    wire signed [DATA_WIDTH-1:0] quant_val;
    wire quant_valid;

    // 1. MAC Unit: 64 Multipliers + 6-level Adder Tree
    mac_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .PC(PC),
        .LOG2_PC(LOG2_PC),
        .MULT_OUT_WIDTH(MULT_OUT_WIDTH),
        .ADDR_TREE_WIDTH(ADDR_TREE_WIDTH)
    ) u_mac (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .data_vec(data_vec),
        .weight_vec(weight_vec),
        .sum_out(mac_sum),
        .valid_out(mac_valid)
    );

    // 2. 32-bit Accumulator across sliding window
    accumulator_32bit #(
        .ADDR_TREE_WIDTH(ADDR_TREE_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH)
    ) u_accum (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(mac_valid),
        .sum_in(mac_sum),
        .window_done(window_done_d1),
        .accum_out(accum_val),
        .accum_valid(accum_valid)
    );

    // 3. Linear Quantizer: 32-bit to INT8 scaling and clamping
    linear_quantizer #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .QUANT_SCALE_WIDTH(QUANT_SCALE_WIDTH),
        .QUANT_SHIFT_WIDTH(QUANT_SHIFT_WIDTH),
        .QUANT_BIAS_WIDTH(QUANT_BIAS_WIDTH)
    ) u_quant (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(accum_valid),
        .accum_in(accum_val),
        .quant_scale(quant_scale),
        .quant_shift(quant_shift),
        .quant_bias(quant_bias),
        .quant_out(quant_val),
        .valid_out(quant_valid)
    );

    // 4. ReLU Unit: Optional bypassable activation
    relu_unit #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_relu (
        .clk(clk),
        .rst_n(rst_n),
        .relu_en(relu_en),
        .data_in(quant_val),
        .data_out(feature_out)
    );

    assign feature_valid = quant_valid;

endmodule