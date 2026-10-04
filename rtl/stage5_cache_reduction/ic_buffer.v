//==============================================================================
// Module: ic_buffer.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 5 - Intermediate Caching & Output Reduction
//------------------------------------------------------------------------------
// Purpose:
//   On-chip simple dual-port BRAM that caches the output feature map of the
//   last deterministic layer (layer N-B) during MC sample 1, so samples 2..S
//   can skip layers 1..N-B (Section IV-B, Fig. 11(c) in Fan et al.).
//
// Architectural Inputs:
//   - clk, rst_n        : Clock (220 MHz) and active-low reset.
//   - wr_en             : Port A write enable (Stage 4 tap, sample 1, layer N-B).
//   - wr_addr           : Port A write address (IC_ADDR_WIDTH bits).
//   - wr_data           : Port A write word (PF x PV x DATA_WIDTH bits).
//   - rd_en             : Port B read enable (replay, samples 2..S, layer N-B+1).
//   - rd_addr           : Port B read address (IC_ADDR_WIDTH bits).
//
// Architectural Outputs:
//   - rd_data           : Port B registered read word (1-cycle read latency).
//   - rd_valid          : 1-cycle strobe indicating rd_data is valid.
//
// Description:
//   One word holds the PF x PV parallel channels of one output pixel, the same
//   granularity Stage 4 produces. Port A and Port B are independent, so caching
//   and replay never contend. The memory array has no reset and the read port
//   is registered so that it maps directly onto block RAM.
//==============================================================================

`include "bcnn_pkg.vh"

module ic_buffer #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter IC_RAM_DEPTH = `IC_RAM_DEPTH,
    parameter IC_ADDR_WIDTH = `IC_ADDR_WIDTH
)(
    input wire clk, rst_n,

    // Port A - write from Stage 4
    input wire wr_en,
    input wire [IC_ADDR_WIDTH-1:0] wr_addr,
    input wire [(PF*PV*DATA_WIDTH)-1:0] wr_data,

    // Port B - read for replay into layer N-B+1
    input wire rd_en,
    input wire [IC_ADDR_WIDTH-1:0] rd_addr,
    output reg [(PF*PV*DATA_WIDTH)-1:0] rd_data,
    output reg rd_valid
);

    reg [(PF*PV*DATA_WIDTH)-1:0] mem [IC_RAM_DEPTH-1:0];

    // Port A
    always @(posedge clk) begin
        if(wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // Port B
    always @(posedge clk) begin
        if(rd_en) begin
            rd_data <= mem[rd_addr];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            rd_valid <= 1'b0;
        end else begin
            rd_valid <= rd_en;
        end
    end

endmodule
