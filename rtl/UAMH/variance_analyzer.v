//==============================================================================
// Module: variance_analyzer.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: UAMH Innovation 1 - Uncertainty-Modulated Precision Storage (UMPS)
//------------------------------------------------------------------------------
// Purpose:
//   Classifies each of the PF x PV channels of a Stage 4 output pixel as
//   low-activity (storable as INT4 without loss) or high-activity (needs INT8)
//   before the pixel is written into the Intermediate-layer Cache.
//
// Architectural Inputs:
//   - clk, rst_n        : Clock (220 MHz) and active-low reset.
//   - valid_in          : Strobe when a Stage 4 pre-dropout pixel is to be cached.
//   - features_in       : PF x PV signed INT8 channels (PF x PV x DATA_WIDTH bits).
//   - thresh_in         : Signed activity threshold tau (UMPS_THRESH_WIDTH bits).
//
// Architectural Outputs:
//   - is_low_var        : Bit f = 1 when channel f lies in [-tau, +tau] and in the
//                         signed INT4 range [-8, +7]; 0 when it needs INT8.
//   - features_out      : features_in registered alongside is_low_var.
//   - valid_out         : 1-cycle strobe.
//
// Description:
//   The cached layer N-B is deterministic, so its output does not vary across
//   MC samples while it is being written; the per-pixel channel magnitude is
//   used as the activity / uncertainty proxy instead. A channel is only marked
//   low when it also fits INT4, so truncation to INT4 is always lossless and
//   tau can only make the decision stricter (tau >= 7 gives a pure range test).
//==============================================================================

`include "bcnn_pkg.vh"

module variance_analyzer #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter INT4_WIDTH = `INT4_WIDTH,
    parameter UMPS_THRESH_WIDTH = `UMPS_THRESH_WIDTH
)(
    input wire clk, rst_n, valid_in,

    input wire [(PF*PV*DATA_WIDTH)-1:0] features_in,
    input wire signed [UMPS_THRESH_WIDTH-1:0] thresh_in,

    output reg [(PF*PV)-1:0] is_low_var,
    output reg [(PF*PV*DATA_WIDTH)-1:0] features_out,
    output reg valid_out
);

    // Signed INT4 limits expressed at INT8 width: [-8, +7]
    localparam signed [DATA_WIDTH-1:0] INT4_MAX = (1 << (INT4_WIDTH-1)) - 1;
    localparam signed [DATA_WIDTH-1:0] INT4_MIN = -(1 << (INT4_WIDTH-1));

    wire [(PF*PV)-1:0] low_var;

    genvar f;
    generate
        for(f=0; f<(PF*PV); f=f+1) begin : GEN_CLASSIFY
            wire signed [DATA_WIDTH-1:0] val = features_in[f*DATA_WIDTH +: DATA_WIDTH];

            // One extra bit so -tau and |val| never overflow
            wire signed [DATA_WIDTH:0] val_x = val;
            wire signed [UMPS_THRESH_WIDTH:0] tau = thresh_in;

            wire fits_int4 = (val >= INT4_MIN) && (val <= INT4_MAX);
            wire within_tau = (val_x <= tau) && (val_x >= -tau);

            assign low_var[f] = fits_int4 && within_tau;
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            is_low_var <= {(PF*PV){1'b0}};
            features_out <= {(PF*PV*DATA_WIDTH){1'b0}};
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if(valid_in) begin
                is_low_var <= low_var;
                features_out <= features_in;
            end
        end
    end

endmodule
