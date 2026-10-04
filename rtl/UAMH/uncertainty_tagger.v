//==============================================================================
// Module: uncertainty_tagger.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: UAMH Innovation 3 - Uncertainty-Tagged Cache Lines (U-Tagging)
//------------------------------------------------------------------------------
// Purpose:
//   Assigns a 2-bit uncertainty tag to every pixel line headed for the
//   Intermediate-layer Cache, so the cache policy can prioritise informative
//   lines over flat background when on-chip memory is constrained.
//
// Architectural Inputs:
//   - clk, rst_n        : Clock (220 MHz) and active-low reset.
//   - valid_in          : Line strobe (aligned with variance_analyzer outputs).
//   - features_in       : PF x PV signed INT8 channels of the line.
//   - is_low_var        : Per-channel INT4 eligibility from variance_analyzer.
//   - zero_thresh       : Channels with |val| <= zero_thresh count as inactive.
//   - high_count_thresh : Minimum number of INT8-precision channels for UTAG_HIGH.
//
// Architectural Outputs:
//   - u_tag_out         : UTAG_ZERO / UTAG_LOW / UTAG_HIGH for the line.
//   - valid_out         : 1-cycle strobe aligned with umps_packer's output.
//
// Description:
//   active = #channels with |val| > zero_thresh, high = #channels with
//   is_low_var == 0 (needs full INT8 precision).
//     active == 0                 -> UTAG_ZERO (flat background)
//     high >= high_count_thresh   -> UTAG_HIGH (wide dynamic range / ambiguous)
//     otherwise                   -> UTAG_LOW  (confident, well bounded)
//   UTAG_PINNED is never produced here; it is reserved for externally pinned
//   lines and is treated by u_tag_manager as never filtered.
//==============================================================================

`include "bcnn_pkg.vh"

module uncertainty_tagger #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter UTAG_WIDTH = `UTAG_WIDTH,
    parameter CNT_WIDTH = $clog2(PF*PV+1)
)(
    input wire clk, rst_n, valid_in,

    input wire [(PF*PV*DATA_WIDTH)-1:0] features_in,
    input wire [(PF*PV)-1:0] is_low_var,
    input wire signed [DATA_WIDTH-1:0] zero_thresh,
    input wire [CNT_WIDTH-1:0] high_count_thresh,

    output reg [UTAG_WIDTH-1:0] u_tag_out,
    output reg valid_out
);

    reg [CNT_WIDTH-1:0] active_channels;
    reg [CNT_WIDTH-1:0] high_channels;
    reg signed [DATA_WIDTH:0] abs_val;
    reg [UTAG_WIDTH-1:0] tag;
    integer f;

    always @(*) begin
        active_channels = {CNT_WIDTH{1'b0}};
        high_channels = {CNT_WIDTH{1'b0}};
        for(f=0; f<(PF*PV); f=f+1) begin
            // One extra bit so |-128| is representable
            abs_val = $signed(features_in[f*DATA_WIDTH +: DATA_WIDTH]);
            if(abs_val < 0) begin
                abs_val = -abs_val;
            end
            if(abs_val > zero_thresh) begin
                active_channels = active_channels + 1'b1;
            end
            if(!is_low_var[f]) begin
                high_channels = high_channels + 1'b1;
            end
        end

        if(active_channels == {CNT_WIDTH{1'b0}}) begin
            tag = `UTAG_ZERO;
        end else if(high_channels >= high_count_thresh) begin
            tag = `UTAG_HIGH;
        end else begin
            tag = `UTAG_LOW;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            u_tag_out <= `UTAG_ZERO;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if(valid_in) begin
                u_tag_out <= tag;
            end
        end
    end

endmodule
