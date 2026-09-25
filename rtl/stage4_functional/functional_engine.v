//==============================================================================
// Module: functional_engine.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 4 - Top Functional & Dropout Engine Subsystem
//------------------------------------------------------------------------------
// Purpose:
//   Top-level wrapper for Stage 4 (Section III-A2, Figure 4 in Fan et al.).
//   Integrates Shortcut Addition, 2D Spatial Pooling, and Bayesian Dropout Masking.
//
// Architectural Inputs:
//   - clk, rst_n          : Clock (220 MHz) and active-low reset.
//   - valid_in            : Valid strobe from Stage 3 Processing Engine.
//   - sc_en               : 1-bit control flag for ResNet skip connection.
//   - pool_mode           : POOL_MODE_WIDTH-bit mode (Bypass, Max Pool, Avg Pool).
//   - pool_win_done       : 1-cycle strobe indicating end of 2x2 pooling window.
//   - pool_step           : POOL_CNT_WIDTH-bit index of current pooling pixel.
//   - mcd_en              : 1-bit control flag for Monte Carlo Dropout.
//   - conv_features_in    : PF x PV parallel INT8 channels from Stage 3 PE.
//   - sc_features_in      : PF x PV parallel INT8 cached channels from SC Buffer.
//   - mask_in             : PF-bit (64-bit) Bernoulli mask from Stage 2 Sampler.
//   - mask_valid          : Valid handshake from Stage 2 Sampler FIFO.
//
// Architectural Outputs:
//   - mask_pop            : 1-cycle pop strobe sent to Stage 2 Sampler FIFO.
//   - stage4_features_out : PF x PV parallel INT8 final processed features to Stage 5 / DRAM.
//   - stage4_valid_out    : 1-cycle strobe indicating stage4_features_out is valid.
//
// Description:
//   Chains sc_addition_unit -> pooling_unit_2d -> dropout_engine into a 
//   fully pipelined post-processing datapath with zero intra-stage bubbles.
//==============================================================================

`include "../bcnn_pkg.vh"

module functional_engine #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC = `PC,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter POOL_MODE_WIDTH = `POOL_MODE_WIDTH,
    parameter POOL_CNT_WIDTH = `POOL_CNT_WIDTH
)(
    input wire clk, rst_n, valid_in, sc_en,

    input wire [POOL_MODE_WIDTH-1:0] pool_mode,
    input wire pool_win_done,
    input wire [POOL_CNT_WIDTH-1:0] pool_step,
    input wire mcd_en,

    input  wire [(PF * PV * DATA_WIDTH)-1:0] conv_features_in,
    input  wire [(PF * PV * DATA_WIDTH)-1:0] sc_features_in,

    input  wire [PF-1:0] mask_in,
    input  wire mask_valid,
    output wire mask_pop,

    output wire [(PF * PV * DATA_WIDTH)-1:0] stage4_features_out,
    output wire stage4_valid_out
);

    wire [(PF*PV*DATA_WIDTH)-1:0] sc_out_bus;
    wire sc_valid;

    wire [(PF*PV*DATA_WIDTH)-1:0] pool_out_bus;
    wire pool_valid;

    // 1. Shortcut (SC) Addition Unit (ResNet Skip Connection)
    sc_addition_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .PC(PC),
        .PF(PF),
        .PV(PV)
    ) u_sc_add (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .sc_en(sc_en),
        .conv_features_in(conv_features_in),
        .sc_features_in(sc_features_in),
        .sc_features_out(sc_out_bus),
        .valid_out(sc_valid)
    );

    // 2. 2D Spatial Pooling Unit (Max / Avg / Bypass)
    pooling_unit_2d #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV),
        .POOL_MODE_WIDTH(POOL_MODE_WIDTH),
        .POOL_CNT_WIDTH(POOL_CNT_WIDTH)
    ) u_pool (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(sc_valid),
        .pool_mode(pool_mode),
        .pool_win_done(pool_win_done),
        .pool_step(pool_step),
        .features_in(sc_out_bus),
        .pooled_features(pool_out_bus),
        .valid_out(pool_valid)
    );

    // 3. Dropout Engine (Applies Stage 2 Bernoulli Mask)
    dropout_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV)
    ) u_dropout (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(pool_valid),
        .mcd_en(mcd_en),
        .features_in(pool_out_bus),
        .mask_in(mask_in),
        .mask_valid(mask_valid),
        .mask_pop(mask_pop),
        .masked_features(stage4_features_out),
        .valid_out(stage4_valid_out)
    );

endmodule