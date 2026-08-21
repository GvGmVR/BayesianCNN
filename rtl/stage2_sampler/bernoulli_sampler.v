//==============================================================================
// Module: bernoulli_sampler.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 2 - Top Bernoulli Sampler Subsystem
//------------------------------------------------------------------------------
// Purpose:
//   Top-level wrapper for the stochastic noise generation engine (Section III-A3,
//   Figure 5 in Fan et al.). Integrates LFSR(s), SIPO, and Mask FIFO.
//
// Architectural Inputs:
//   - Control: sampler_en (enable generation), load_seed (re-seed trigger).
//   - Seed   : 128-bit non-zero initialization seed (seed_in).
//   - Demand : mask_pop request from Stage 4 (Dropout Engine).
//
// Architectural Outputs:
//   - Mask Bus: PF-bit (64-bit) Bernoulli mask vector (mask_out) to Stage 4.
//   - Status  : mask_valid, mask_empty, mask_full, and mask_count.
//
// Description:
//   Generates pseudo-random binary masks with user-defined keep probability p
//   in the background, overlapping noise generation with convolution compute.
//==============================================================================

`include "bcnn_pkg.vh"

module bernoulli_sampler #(
    parameter PF            = `PF,
    parameter LFSR_WIDTH    = `LFSR_WIDTH,
    parameter N_LFSR        = `N_LFSR, // N_lfsr is one - prob 50%;; 2 lfsr - prob 25 and so on
    parameter FIFO_DEPTH    = `MASK_FIFO_DEPTH,
    parameter FIFO_ADDR     = `MASK_FIFO_ADDR
)(
    input wire clk,rst_n,

    input wire sampler_en,load_seed,
    input wire [LFSR_WIDTH-1:0] seed_in,

    // stgae 4 (Dropout signals)
    input wire mask_pop,
    output wire [PF-1:0] mask_out,
    output wire mask_valid,
    output wire mask_empty,
    output wire mask_full,
    output wire [FIFO_ADDR:0] mask_count
);

    wire [N_LFSR-1:0] lfsr_bits;
    wire bernoulli_bit;
    wire [PF-1:0] sipo_parallel_word;
    wire sipo_word_valid;


    assign bernoulli_bit = &lfsr_bits; 

    genvar i;
    generate
        for (i=0;i<N_LFSR;i=i+1) begin : GEN_LFSRS
            wire [LFSR_WIDTH-1:0] instance_seed;
            assign instance_seed = seed_in ^ {{(LFSR_WIDTH-8){1'b0}},i[7:0]};

            lfsr_128bit #(
                .LFSR_WIDTH(LFSR_WIDTH),
                .TAP1(`LFSR_TAP1),
                .TAP2(`LFSR_TAP2),
                .TAP3(`LFSR_TAP3),
                .TAP4(`LFSR_TAP4)
            ) u_lfsr (
                .clk(clk),
                .rst_n(rst_n),
                .en(sampler_en && !mask_full), // Pause when FIFO is full
                .load_seed(load_seed),
                .seed_in(instance_seed),
                .lfsr_bit_out(lfsr_bits[i]),
                .lfsr_state_out()
            );
        end
    endgenerate

    sipo_shift_reg #(
        .PF(PF),
        .CNT_WIDTH(`SIPO_CNT_WIDTH)
    ) u_sipo (
        .clk(clk),
        .rst_n(rst_n),
        .shift_en(sampler_en && !mask_full),
        .bit_in(bernoulli_bit),
        .parallel_mask(sipo_parallel_word),
        .word_valid(sipo_word_valid)
    );

    mask_fifo #(
        .DATA_WIDTH(PF),
        .FIFO_DEPTH(FIFO_DEPTH),
        .ADDR_WIDTH(FIFO_ADDR)
    ) u_mask_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .push(sipo_word_valid),
        .din(sipo_parallel_word),
        .full(mask_full),
        .pop(mask_pop),
        .dout(mask_out),
        .empty(mask_empty),
        .occupancy(mask_count)
    );

    assign mask_valid = !mask_empty;
    
endmodule