//==============================================================================
// Module: umps_unpacker.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: UAMH Innovation 1 - Uncertainty-Modulated Precision Storage (UMPS)
//------------------------------------------------------------------------------
// Purpose:
//   Restores a compacted IC line into the standard PF x PV signed INT8 pixel
//   before it enters the replay dropout engine (samples 2..S).
//
// Architectural Inputs:
//   - clk, rst_n           : Clock (220 MHz) and active-low reset.
//   - valid_in             : Line strobe from ic_buffer.
//   - pack_mask_in         : Per-channel precision metadata stored with the line.
//   - packed_features_in   : Compacted line read from ic_buffer (byte 0 at LSB).
//
// Architectural Outputs:
//   - unpacked_features_out: PF x PV signed INT8 channels.
//   - valid_out            : 1-cycle strobe.
//
// Description:
//   Walks the pairs in the same order as umps_packer and recomputes each pair's
//   byte position from the mask. An INT4 pair is split into its two nibbles and
//   each is sign-extended ({{4{n[3]}}, n}); an INT8 pair is copied as two bytes.
//   Decoding depends only on the metadata stored with each line, so a line is
//   always decoded the way it was written. With an all-zero mask the line is
//   passed through unchanged.
//==============================================================================

`include "bcnn_pkg.vh"

module umps_unpacker #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter INT4_WIDTH = `INT4_WIDTH,
    parameter LEN_WIDTH = $clog2(PF*PV+1)
)(
    input wire clk, rst_n, valid_in,

    input wire [(PF*PV)-1:0] pack_mask_in,
    input wire [(PF*PV*DATA_WIDTH)-1:0] packed_features_in,

    output reg [(PF*PV*DATA_WIDTH)-1:0] unpacked_features_out,
    output reg valid_out
);

    localparam NPAIRS = (PF*PV)/2;

    reg [(PF*PV*DATA_WIDTH)-1:0] pixel;
    reg [DATA_WIDTH-1:0] lane;
    reg [LEN_WIDTH-1:0] pos;
    integer k;

    // Gather: recompute each pair's byte position exactly as the packer did
    always @(*) begin
        pixel = {(PF*PV*DATA_WIDTH){1'b0}};
        pos = {LEN_WIDTH{1'b0}};
        for(k=0; k<NPAIRS; k=k+1) begin
            lane = packed_features_in[pos*DATA_WIDTH +: DATA_WIDTH];
            if(pack_mask_in[2*k] == `PRECISION_MODE_INT4) begin
                pixel[(2*k)*DATA_WIDTH +: DATA_WIDTH] = {{(DATA_WIDTH-INT4_WIDTH){lane[INT4_WIDTH-1]}}, lane[INT4_WIDTH-1:0]};
                pixel[(2*k+1)*DATA_WIDTH +: DATA_WIDTH] = {{(DATA_WIDTH-INT4_WIDTH){lane[2*INT4_WIDTH-1]}}, lane[INT4_WIDTH +: INT4_WIDTH]};
                pos = pos + 1'b1;
            end else begin
                pixel[(2*k)*DATA_WIDTH +: DATA_WIDTH] = lane;
                pixel[(2*k+1)*DATA_WIDTH +: DATA_WIDTH] = packed_features_in[(pos+1)*DATA_WIDTH +: DATA_WIDTH];
                pos = pos + 2;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            unpacked_features_out <= {(PF*PV*DATA_WIDTH){1'b0}};
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if(valid_in) begin
                unpacked_features_out <= pixel;
            end
        end
    end

endmodule
