//==============================================================================
// Module: u_tag_manager.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: UAMH Innovation 3 - Uncertainty-Tagged Cache Lines (U-Tagging)
//------------------------------------------------------------------------------
// Purpose:
//   Uncertainty-aware admission policy for the Intermediate-layer Cache: under
//   memory pressure, zero-information lines are spilled to DRAM so that on-chip
//   capacity stays available for informative (high-uncertainty) lines.
//
// Architectural Inputs:
//   - clk, rst_n         : Clock (220 MHz) and active-low reset.
//   - clear              : Inference start; clears the telemetry counters.
//   - utag_en            : 1 = U-Tagging policy, 0 = baseline (admit while not full).
//   - line_valid_in      : A line is ready to be cached this cycle.
//   - u_tag_in           : Tag of that line.
//   - ic_occupancy       : IC storage in use (bytes, so UMPS compression counts).
//   - ic_capacity        : IC storage capacity (same unit).
//   - ic_full            : IC cannot take another full-length line.
//
// Architectural Outputs:
//   - admit_to_bram      : Write the line into the IC buffer.
//   - spill_to_dram      : Send the line to off-chip memory instead.
//   - high_u_cached_count: UTAG_HIGH lines admitted on-chip (saturating).
//   - low_u_bypassed_count: UTAG_ZERO / UTAG_LOW lines spilled (saturating).
//
// Description:
//   pressure = occupancy >= capacity x UTAG_CAP_THRESH_PCT / 100.
//     utag_en = 0                 : admit = !full, spill = 0 (baseline).
//     pressure && tag == ZERO     : spill (keep BRAM for informative lines).
//     !full                       : admit.
//     full                        : spill (UTAG_PINNED included; resident lines
//                                   are never evicted, so their order is kept).
//   Decisions are combinational so they act in the same cycle as the write.
//==============================================================================

`include "bcnn_pkg.vh"

module u_tag_manager #(
    parameter UTAG_WIDTH = `UTAG_WIDTH,
    parameter OCC_WIDTH = `IC_ADDR_WIDTH+$clog2(`PF*`PV)+1,
    parameter CNT_WIDTH = `IC_ADDR_WIDTH+2
)(
    input wire clk, rst_n, clear,
    input wire utag_en,

    input wire line_valid_in,
    input wire [UTAG_WIDTH-1:0] u_tag_in,
    input wire [OCC_WIDTH-1:0] ic_occupancy,
    input wire [OCC_WIDTH-1:0] ic_capacity,
    input wire ic_full,

    output wire admit_to_bram,
    output wire spill_to_dram,
    output reg [CNT_WIDTH-1:0] high_u_cached_count,
    output reg [CNT_WIDTH-1:0] low_u_bypassed_count
);

    wire pressure_active = (ic_occupancy >= (ic_capacity * `UTAG_CAP_THRESH_PCT) / 100);
    wire filter_zero = pressure_active && (u_tag_in == `UTAG_ZERO);

    assign admit_to_bram = line_valid_in && (utag_en ? (!filter_zero && !ic_full) : !ic_full);
    assign spill_to_dram = line_valid_in && utag_en && (filter_zero || ic_full);

    wire count_high = admit_to_bram && (u_tag_in == `UTAG_HIGH);
    wire count_low = spill_to_dram && ((u_tag_in == `UTAG_ZERO) || (u_tag_in == `UTAG_LOW));

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            high_u_cached_count <= {CNT_WIDTH{1'b0}};
            low_u_bypassed_count <= {CNT_WIDTH{1'b0}};
        end else if(clear) begin
            high_u_cached_count <= {CNT_WIDTH{1'b0}};
            low_u_bypassed_count <= {CNT_WIDTH{1'b0}};
        end else if(utag_en) begin
            if(count_high && (high_u_cached_count != {CNT_WIDTH{1'b1}})) begin
                high_u_cached_count <= high_u_cached_count + 1'b1;
            end
            if(count_low && (low_u_bypassed_count != {CNT_WIDTH{1'b1}})) begin
                low_u_bypassed_count <= low_u_bypassed_count + 1'b1;
            end
        end
    end

endmodule
