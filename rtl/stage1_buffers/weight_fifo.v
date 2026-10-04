//==============================================================================
// Module: weight_fifo.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 1 - Weight Storage Primitive
//------------------------------------------------------------------------------
// Purpose:
//   Synchronous First-In, First-Out (FIFO) queue for filter weights.
//
// Architectural Inputs:
//   - Push Side: 8-bit weight byte from DRAM weight loader.
//   - Pop Side : Pop enable from PE compute execution controller.
//
// Architectural Outputs:
//   - Pop Side : 8-bit weight byte to the PE MAC unit, Full and Empty flags.
//
// Description:
//   Buffers filter weights and provides steady streaming to the MAC arrays.
//==============================================================================

`include "bcnn_pkg.vh"

module weight_fifo #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter FIFO_DEPTH = `FIFO_DEPTH,
    parameter ADDR_WIDTH = $clog2(FIFO_DEPTH)
)(
    input wire clk,rst_n,

    input wire push,
    input wire [DATA_WIDTH-1:0]din,
    output wire full,

    input wire pop,
    output reg [DATA_WIDTH-1:0]dout,
    output wire empty
);

    reg [DATA_WIDTH-1:0] mem [FIFO_DEPTH-1:0];
    reg [ADDR_WIDTH-1:0] wr_ptr;
    reg [ADDR_WIDTH-1:0] rd_ptr;
    reg [ADDR_WIDTH:0] count;

    assign full = (count == FIFO_DEPTH);
    assign empty = (count ==0);

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)begin 
            wr_ptr <= {ADDR_WIDTH{1'b0}};
            rd_ptr <= {ADDR_WIDTH{1'b0}};
            count <= {(ADDR_WIDTH+1){1'b0}};
            dout <= {DATA_WIDTH{1'b0}};
        end else begin 
            //Push operation
            if(push && !full)begin 
                mem[wr_ptr] <= din;
                wr_ptr <= wr_ptr + 1'b1;
            end
            // Pop operation
            if(pop && !empty)begin
                dout <= mem[rd_ptr];
                rd_ptr <= rd_ptr + 1'b1;
            end

            // Count update
            case({push && !full, pop && !empty}) 
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

endmodule