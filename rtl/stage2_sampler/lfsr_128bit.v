//==============================================================================
// Module: lfsr_128bit.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 2 - Stochastic Noise Generation (Primitive)
//------------------------------------------------------------------------------
// Purpose:
//   128-bit 4-Tap Linear Feedback Shift Register (LFSR) pseudo-random generator.
//
// Architectural Inputs:
//   - clk, rst_n    : Clock (220 MHz) and active-low reset.
//   - en            : Clock enable signal to advance the shift register.
//   - load_seed     : Synchronous load trigger to initialize the LFSR state.
//   - seed_in       : 128-bit initial non-zero random seed vector.
//
// Architectural Outputs:
//   - lfsr_bit_out  : 1-bit pseudo-random output per cycle (from register R_127).
//   - lfsr_state_out: Full 128-bit internal state register (for telemetry/debug).
//
// Description:
//   Implements a maximum-length Galois/Fibonacci 4-tap shift register with period 
//   2^128 - 1 states (~1500 years at 220 MHz), ensuring zero noise correlation.
//==============================================================================

`include "bcnn_pkg.vh"

module lfsr_128bit #(
    parameter LFSR_WIDTH = `LFSR_WIDTH,
    parameter TAP1       = `LFSR_TAP1, //127
    parameter TAP2       = `LFSR_TAP2,
    parameter TAP3       = `LFSR_TAP3,
    parameter TAP4       = `LFSR_TAP4  // 120
)(
    input wire clk,rst_n,en,load_seed,
    input wire [LFSR_WIDTH-1:0] seed_in,

    output wire lfsr_bit_out,
    output wire [LFSR_WIDTH-1:0] lfsr_state_out
);

    reg [LFSR_WIDTH-1:0] r_state; //register state 0 - 127

    wire feedback_bit;
    assign feedback_bit = r_state[TAP1]^ r_state[TAP2]^r_state[TAP3]^r_state[TAP4];

    assign lfsr_bit_out = r_state[LFSR_WIDTH-1];
    assign lfsr_state_out = r_state;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            r_state <= {{(LFSR_WIDTH-1){1'b0},1'b1}};
        end else if(load_seed) begin 
            r_state <= (|seed_in) ? seed_in : {{(LFSR_WIDTH-1){1'b0},1'b1}};
        end else if (en) begin 
            r_state <= {r_state[LFSR_WIDTH-2:0],feedback_bit}; // Left or |right
        end
    end

endmodule