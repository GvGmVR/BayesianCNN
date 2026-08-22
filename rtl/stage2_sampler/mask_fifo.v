//==============================================================================
// Module: mask_fifo.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 2 - Dropout Mask Buffering
//------------------------------------------------------------------------------
// Purpose:
//   Synchronous First-In, First-Out (FIFO) queue for buffering PF-bit (64-bit) 
//   Bernoulli dropout mask vectors.
//
// Architectural Inputs:
//   - Push Side: PF-bit parallel mask word and word_valid strobe from SIPO.
//   - Pop Side : mask_pop strobe from Stage 4 (Dropout Engine).
//
// Architectural Outputs:
//   - Pop Side : PF-bit dropout mask vector (mask_dout) to Dropout Engine.
//   - Status   : full, empty, and occupancy word count.
//
// Description:
//   Decouples the background PRNG mask generation from foreground PE execution,
//   allowing masks to be pre-generated with zero compute stall cycles.
//==============================================================================

`include "bcnn_pkg.vh"

module mask_fifo #(
    parameter DATA_WIDTH = `PF,
    parameter FIFO_DEPTH = `MASK_FIFO_DEPTH,
    parameter ADDR_WIDTH = `MASK_FIFO_ADDR
)(
    input wire clk, rst_n,
    input wire push,
    input wire [DATA_WIDTH-1:0] din,
    output wire full,

    input wire pop,
    output reg [DATA_WIDTH-1:0] dout,
    output wire empty,
    output wire [ADDR_WIDTH:0] occupancy 
);

    reg [DATA_WIDTH-1:0] mem [FIFO_DEPTH-1:0];
    reg [ADDR_WIDTH-1:0] wr_ptr;
    reg [ADDR_WIDTH-1:0] rd_ptr;
    reg [ADDR_WIDTH:0] count;

    assign full = (count == FIFO_DEPTH);
    assign empty = (count == {(ADDR_WIDTH+1){1'b0}});
    assign occupancy = count;

    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin 
            wr_ptr <= {ADDR_WIDTH{1'b0}};
            rd_ptr <= {ADDR_WIDTH{1'b0}};
            count <= {(ADDR_WIDTH+1){1'b0}};
            dout <= {DATA_WIDTH{1'b0}};
        end else begin 
            //push
            if(push && !full)begin 
                mem[wr_ptr] <= din;
                wr_ptr <= wr_ptr + 1'b1;
            end
            // pop 
            if (pop && !empty)begin 
                dout <= mem[rd_ptr];
                rd_ptr <= rd_ptr + 1'b1;
            end

            //count if FIFO occupied
            case ({push && !full,pop && !empty})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

endmodule