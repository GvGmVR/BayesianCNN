//==============================================================================
// Module: sipo_shift_reg.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 2 - Serial-to-Parallel Conversion
//------------------------------------------------------------------------------
// Purpose:
//   Serial-In Parallel-Out (SIPO) shift register that converts a 1-bit/cycle 
//   random stream into PF-bit (64-bit) parallel mask vectors.
//
// Architectural Inputs:
//   - clk, rst_n    : Clock (220 MHz) and active-low reset.
//   - shift_en      : Enable signal indicating a valid bit is arriving.
//   - bit_in        : 1-bit random input from LFSR / Bit-wise logic.
//
// Architectural Outputs:
//   - parallel_mask : PF-bit (64-bit) assembled dropout mask word.
//   - word_valid    : 1-cycle strobe pulse indicating a complete 64-bit word is ready.
//
// Description:
//   Collects 64 sequential random bits over 64 clock cycles, then latches the 
//   assembled word into parallel_mask and fires word_valid to push into FIFO.
//==============================================================================

`include "bcnn_pkg.vh"

module sipo_shift_reg #(
    parameter PF            = `PF,
    parameter CNT_WIDTH     = `SIPO_CNT_WIDTH
)(
    input clk,rst_n,shift_en,bit_in,

    output reg [PF-1:0] parallel_mask,
    output reg word_valid
);
    reg [PF-1:0] shift_reg;
    reg [CNT_WIDTH-1:0] bit_cnt;

    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            shift_reg <= {PF{1'b0}};
            bit_cnt <= {CNT_WIDTH{1'b0}};
            parallel_mask <= {PF{1'b0}};
            word_valid <= 1'b0;
        end else if(shift_en) begin 
            shift_reg <= {shift_reg[PF-2:0], bit_in};

            if(bit_cnt == (PF-1'b1))begin 
                bit_cnt <= {CNT_WIDTH{1'b0}};
                parallel_mask <= {shift_reg[PF-2:0], bit_in};
                word_valid <= 1'b1;
            end else begin 
                bit_cnt <= bit_cnt + 1'b1;
                word_valid <= 1'b0;
            end 
        end else begin 
            word_valid <= 1'b0;
        end
    end

endmodule