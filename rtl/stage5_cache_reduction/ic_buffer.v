//==============================================================================
// Module: ic_buffer.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 5 - Intermediate Caching & Output Reduction
//------------------------------------------------------------------------------
// Purpose:
//   On-chip cache for the output feature map of the last deterministic layer
//   (layer N-B) during MC sample 1, so samples 2..S can skip layers 1..N-B
//   (Section IV-B, Fig. 11(c) in Fan et al.). Stores variable-length lines so
//   that UMPS-compressed pixels really occupy less memory.
//
// Architectural Inputs:
//   - clk, rst_n        : Clock (220 MHz) and active-low reset.
//   - clear             : Inference start; empties the cache.
//   - rd_rewind         : Layer boundary; restarts replay from the first line.
//   - wr_en             : Append one line (Stage 4 tap, sample 1, layer N-B).
//   - wr_data           : Line bytes, byte 0 at the LSB (PF x PV x DATA_WIDTH bits).
//   - wr_len            : Line length in bytes (PF x PV for an uncompressed line).
//   - wr_mask           : UMPS per-channel precision metadata of the line.
//   - wr_u_tag          : U-Tagging uncertainty tag of the line.
//   - rd_en             : Read the next line (replay, samples 2..S, layer N-B+1).
//
// Architectural Outputs:
//   - rd_data           : Line bytes, byte 0 at the LSB (1-cycle read latency).
//   - rd_mask           : Metadata of the line on rd_data.
//   - rd_u_tag          : Uncertainty tag of the line on rd_data.
//   - rd_valid          : 1-cycle strobe indicating rd_data / rd_mask / rd_u_tag are valid.
//   - line_count        : Lines (pixels) cached.
//   - byte_count        : Bytes of data storage used.
//   - full              : No room for another full-length line.
//
// Description:
//   Data storage is PF x PV byte-wide RAM lanes of IC_RAM_DEPTH rows, addressed
//   as one byte stream: byte address A lives in lane A mod (PF x PV), row
//   A / (PF x PV). Lines are appended back to back, so a line may span two rows;
//   every lane still sees a single row per cycle, so each lane maps onto its own
//   block RAM. A rotator aligns the line to the lanes on write and back on read.
//   Per line, the length (async-read, needed to address the next line), the
//   pair-level precision mask and the U-tag (sync-read with the data) are kept
//   in side arrays of 2 x IC_RAM_DEPTH entries, the most lines a half-size
//   line format allows.
//   Admission is decided upstream (u_tag_manager); wr_en is already gated.
//   Lines are replayed in the order they were written. With full-length lines
//   every line starts on a row boundary and the cache behaves exactly like the
//   baseline one-pixel-per-row buffer.
//   Requires PF x PV to be a power of two and even.
//==============================================================================

`include "bcnn_pkg.vh"

module ic_buffer #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter IC_RAM_DEPTH = `IC_RAM_DEPTH,
    parameter IC_ADDR_WIDTH = `IC_ADDR_WIDTH,
    parameter LEN_WIDTH = $clog2(PF*PV+1),
    parameter UTAG_WIDTH = `UTAG_WIDTH
)(
    input wire clk, rst_n,
    input wire clear, rd_rewind,

    // Write side - append a line
    input wire wr_en,
    input wire [(PF*PV*DATA_WIDTH)-1:0] wr_data,
    input wire [LEN_WIDTH-1:0] wr_len,
    input wire [(PF*PV)-1:0] wr_mask,
    input wire [UTAG_WIDTH-1:0] wr_u_tag,

    // Read side - next line in write order
    input wire rd_en,
    output wire [(PF*PV*DATA_WIDTH)-1:0] rd_data,
    output wire [(PF*PV)-1:0] rd_mask,
    output reg [UTAG_WIDTH-1:0] rd_u_tag,
    output reg rd_valid,

    // Occupancy
    output wire [IC_ADDR_WIDTH+1:0] line_count,
    output wire [IC_ADDR_WIDTH+$clog2(PF*PV):0] byte_count,
    output wire full
);

    localparam NLANES = PF*PV;
    localparam LINE_W = PF*PV*DATA_WIDTH;
    localparam LANE_IDX_W = $clog2(NLANES);
    localparam NPAIRS = NLANES/2;
    localparam LINE_DEPTH = 2*IC_RAM_DEPTH;
    localparam LINE_ADDR_W = IC_ADDR_WIDTH+1;
    localparam LINE_CNT_W = IC_ADDR_WIDTH+2;
    localparam BYTE_CNT_W = IC_ADDR_WIDTH+LANE_IDX_W+1;
    localparam CAPACITY = IC_RAM_DEPTH*NLANES;

    reg [BYTE_CNT_W-1:0] wr_byte, rd_byte;
    reg [LINE_CNT_W-1:0] wr_line, rd_line;

    // Per-line side arrays
    reg [LEN_WIDTH-1:0] len_mem [0:LINE_DEPTH-1];
    reg [NPAIRS-1:0] pair_mem [0:LINE_DEPTH-1];
    reg [UTAG_WIDTH-1:0] u_tag_mem [0:LINE_DEPTH-1];
    reg [NPAIRS-1:0] rd_pairs_q;
    reg [LANE_IDX_W-1:0] rd_off_q;

    assign line_count = wr_line;
    assign byte_count = wr_byte;
    assign full = (wr_line == LINE_DEPTH) || (wr_byte > CAPACITY - NLANES);

    wire wr_ok = wr_en && !full;
    wire rd_ok = rd_en && (rd_line < wr_line);
    wire [LEN_WIDTH-1:0] rd_len = len_mem[rd_line[LINE_ADDR_W-1:0]];

    // Byte address -> (row, starting lane)
    wire [IC_ADDR_WIDTH-1:0] wr_base = wr_byte[LANE_IDX_W +: IC_ADDR_WIDTH];
    wire [LANE_IDX_W-1:0] wr_off = wr_byte[LANE_IDX_W-1:0];
    wire [IC_ADDR_WIDTH-1:0] rd_base = rd_byte[LANE_IDX_W +: IC_ADDR_WIDTH];
    wire [LANE_IDX_W-1:0] rd_off = rd_byte[LANE_IDX_W-1:0];

    // Write rotator: line byte j goes to lane (wr_off + j) mod NLANES
    wire [(2*LINE_W)-1:0] wr_dbl = {wr_data, wr_data} << (wr_off*DATA_WIDTH);
    wire [LINE_W-1:0] wr_rot = wr_dbl[(2*LINE_W)-1:LINE_W];

    // Read rotator: lane (rd_off + j) mod NLANES returns to line byte j
    wire [LINE_W-1:0] rd_bus;
    wire [(2*LINE_W)-1:0] rd_dbl = {rd_bus, rd_bus} >> (rd_off_q*DATA_WIDTH);
    assign rd_data = rd_dbl[LINE_W-1:0];

`ifndef SYNTHESIS
    // Simulation-only row-major view of the lanes: mem[row] = all lanes of that row
    reg [LINE_W-1:0] mem [0:IC_RAM_DEPTH-1];
`endif

    genvar k;
    generate
        for(k=0; k<NLANES; k=k+1) begin : GEN_LANE
            reg [DATA_WIDTH-1:0] bank [0:IC_RAM_DEPTH-1];
            reg [DATA_WIDTH-1:0] q;

            // Lanes before the starting lane wrap into the next row
            wire [LANE_IDX_W-1:0] wr_rel = k - wr_off;
            wire [IC_ADDR_WIDTH-1:0] wr_row = (k >= wr_off) ? wr_base : wr_base + 1'b1;
            wire [IC_ADDR_WIDTH-1:0] rd_row = (k >= rd_off) ? rd_base : rd_base + 1'b1;
            wire wr_hit = wr_ok && (wr_rel < wr_len);

            always @(posedge clk) begin
                if(wr_hit) begin
                    bank[wr_row] <= wr_rot[k*DATA_WIDTH +: DATA_WIDTH];
                end
                if(rd_ok) begin
                    q <= bank[rd_row];
                end
            end

            assign rd_bus[k*DATA_WIDTH +: DATA_WIDTH] = q;

`ifndef SYNTHESIS
            always @(posedge clk) begin
                if(wr_hit) begin
                    mem[wr_row][k*DATA_WIDTH +: DATA_WIDTH] <= wr_rot[k*DATA_WIDTH +: DATA_WIDTH];
                end
            end
`endif
        end

        // Precision is decided per pair, so one metadata bit per pair is stored
        for(k=0; k<NPAIRS; k=k+1) begin : GEN_PAIR
            assign rd_mask[2*k] = rd_pairs_q[k];
            assign rd_mask[2*k+1] = rd_pairs_q[k];
        end
    endgenerate

    reg [NPAIRS-1:0] wr_pairs;
    integer p;

    always @(*) begin
        for(p=0; p<NPAIRS; p=p+1) begin
            wr_pairs[p] = wr_mask[2*p];
        end
    end

    always @(posedge clk) begin
        if(wr_ok) begin
            len_mem[wr_line[LINE_ADDR_W-1:0]] <= wr_len;
            pair_mem[wr_line[LINE_ADDR_W-1:0]] <= wr_pairs;
            u_tag_mem[wr_line[LINE_ADDR_W-1:0]] <= wr_u_tag;
        end
        if(rd_ok) begin
            rd_pairs_q <= pair_mem[rd_line[LINE_ADDR_W-1:0]];
            rd_u_tag <= u_tag_mem[rd_line[LINE_ADDR_W-1:0]];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            wr_byte <= {BYTE_CNT_W{1'b0}};
            wr_line <= {LINE_CNT_W{1'b0}};
            rd_byte <= {BYTE_CNT_W{1'b0}};
            rd_line <= {LINE_CNT_W{1'b0}};
            rd_off_q <= {LANE_IDX_W{1'b0}};
            rd_valid <= 1'b0;
        end else begin
            if(clear) begin
                wr_byte <= {BYTE_CNT_W{1'b0}};
                wr_line <= {LINE_CNT_W{1'b0}};
            end else if(wr_ok) begin
                wr_byte <= wr_byte + wr_len;
                wr_line <= wr_line + 1'b1;
            end

            if(clear || rd_rewind) begin
                rd_byte <= {BYTE_CNT_W{1'b0}};
                rd_line <= {LINE_CNT_W{1'b0}};
            end else if(rd_ok) begin
                rd_byte <= rd_byte + rd_len;
                rd_line <= rd_line + 1'b1;
            end

            rd_valid <= rd_ok;
            if(rd_ok) begin
                rd_off_q <= rd_off;
            end
        end
    end

endmodule
