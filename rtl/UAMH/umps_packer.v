//==============================================================================
// Module: umps_packer.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: UAMH Innovation 1 - Uncertainty-Modulated Precision Storage (UMPS)
//------------------------------------------------------------------------------
// Purpose:
//   Compresses one PF x PV channel pixel into a variable-length line for the
//   IC buffer: each low-activity channel pair (2k, 2k+1) shares one byte as two
//   INT4 nibbles, every other channel keeps a full INT8 byte.
//
// Architectural Inputs:
//   - clk, rst_n          : Clock (220 MHz) and active-low reset.
//   - valid_in            : Pixel strobe from variance_analyzer.
//   - umps_en             : 1 = pack, 0 = bypass (plain INT8 line).
//   - is_low_var          : Per-channel INT4 eligibility from variance_analyzer.
//   - features_in         : PF x PV signed INT8 channels.
//
// Architectural Outputs:
//   - packed_features_out : Compacted line, byte 0 at the LSB; bytes at and above
//                           packed_len_out are zero.
//   - pack_mask_out       : Bit f = 1 when channel f is stored as INT4
//                           (always equal within a pair 2k, 2k+1).
//   - packed_len_out      : Line length in bytes, PF x PV / 2 .. PF x PV.
//   - valid_out           : 1-cycle strobe.
//
// Description:
//   Pairs are processed in order k = 0 .. PF x PV/2-1 and appended at a running
//   byte position: a packed pair emits {ch[2k+1][3:0], ch[2k][3:0]} (1 byte), an
//   unpacked pair emits ch[2k] then ch[2k+1] (2 bytes). A pair is packed only
//   when both of its channels are low-activity, so the line can shrink to half
//   size. Requires DATA_WIDTH == 2 x INT4_WIDTH and an even PF x PV.
//==============================================================================

`include "bcnn_pkg.vh"

module umps_packer #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter INT4_WIDTH = `INT4_WIDTH,
    parameter LEN_WIDTH = $clog2(PF*PV+1)
)(
    input wire clk, rst_n, valid_in,
    input wire umps_en,

    input wire [(PF*PV)-1:0] is_low_var,
    input wire [(PF*PV*DATA_WIDTH)-1:0] features_in,

    output reg [(PF*PV*DATA_WIDTH)-1:0] packed_features_out,
    output reg [(PF*PV)-1:0] pack_mask_out,
    output reg [LEN_WIDTH-1:0] packed_len_out,
    output reg valid_out
);

    localparam NPAIRS = (PF*PV)/2;

    reg [(PF*PV*DATA_WIDTH)-1:0] line;
    reg [(PF*PV)-1:0] mask;
    reg [LEN_WIDTH-1:0] pos;
    integer k;

    // Compaction: append each pair at the running byte position
    always @(*) begin
        line = {(PF*PV*DATA_WIDTH){1'b0}};
        mask = {(PF*PV){1'b0}};
        pos = {LEN_WIDTH{1'b0}};
        for(k=0; k<NPAIRS; k=k+1) begin
            if(umps_en && is_low_var[2*k] && is_low_var[2*k+1]) begin
                line[pos*DATA_WIDTH +: DATA_WIDTH] = {features_in[(2*k+1)*DATA_WIDTH +: INT4_WIDTH], features_in[(2*k)*DATA_WIDTH +: INT4_WIDTH]};
                mask[2*k] = `PRECISION_MODE_INT4;
                mask[2*k+1] = `PRECISION_MODE_INT4;
                pos = pos + 1'b1;
            end else begin
                line[pos*DATA_WIDTH +: DATA_WIDTH] = features_in[(2*k)*DATA_WIDTH +: DATA_WIDTH];
                line[(pos+1)*DATA_WIDTH +: DATA_WIDTH] = features_in[(2*k+1)*DATA_WIDTH +: DATA_WIDTH];
                pos = pos + 2;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            packed_features_out <= {(PF*PV*DATA_WIDTH){1'b0}};
            pack_mask_out <= {(PF*PV){1'b0}};
            packed_len_out <= {LEN_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if(valid_in) begin
                packed_features_out <= line;
                pack_mask_out <= mask;
                packed_len_out <= pos;
            end
        end
    end

endmodule
