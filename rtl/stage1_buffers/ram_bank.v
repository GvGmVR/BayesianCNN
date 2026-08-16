//==============================================================================
// Module: ram_bank.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 1 - Memory Subsystem (Primitive)
//------------------------------------------------------------------------------
// Purpose:
//   Synchronous Simple Dual-Port on-chip Block RAM (BRAM) primitive.
//
// Architectural Inputs:
//   - Port A (Write): 8-bit Data word, row address, and write enable from DMA Ingress.
//   - Port B (Read) : Row address and read enable from Read Address Generator (RAG).
//
// Architectural Outputs:
//   - Port B: 8-bit registered data word output to compute datapath.
//
// Description:
//   Provides independent read/write ports to support concurrent background 
//   DRAM burst loading and foreground PE compute access.
//==============================================================================

`include "bcnn_pkg.vh"

module ram#(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter RAM_DEPTH = `RAM_DEPTH,
    parameter ADDR_WIDTH = `ADDR_WIDTH
)(
    input wire clk,
    input wire rst_n,

    input wire we_a,
    input wire [ADDR_WIDTH-1:0]addr_a,
    input wire [DATA_WIDTH-1:0]din_a,

    input wire re_b,
    input wire [ADDR_WIDTH-1:0]addr_b,
    output reg [DATA_WIDTH-1:0]dout_b

);

    reg [DATA_WIDTH-1:0] mem [RAM_DEPTH-1:0];

        always @(posedge clk) begin
            if(we_a) begin
                mem[addr_a] <= din_a;
            end
        end

        always @(posedge clk) begin
            if(!rst_n) begin
                dout_b <= {DATA_WIDTH{1'b0}}; // Repeat 0 DATA_WIDTH times
            end
            else
            if(re_b) begin
                dout_b <= mem[addr_b];
            end
        end

endmodule