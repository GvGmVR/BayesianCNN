//==============================================================================
// Module: relu_unit.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 3 - Activation Function
//------------------------------------------------------------------------------
// Purpose:
//   Rectified Linear Unit (ReLU) activation with dynamic bypass control.
//
// Architectural Inputs:
//   - clk, rst_n    : Clock (220 MHz) and active-low reset.
//   - relu_en       : 1-bit control flag (1 = enable ReLU max(0, x), 0 = bypass).
//   - data_in       : DATA_WIDTH-bit signed quantized activation input.
//
// Architectural Outputs:
//   - data_out      : DATA_WIDTH-bit signed activated feature output.
//
// Description:
//   If relu_en is asserted and data_in is negative (MSB == 1), data_out is 
//   forced to zero. If relu_en is 0, data_in passes through unaltered.
//==============================================================================

`include "bcnn_pkg.vh"

module relu_unit #(
    parameter DATA_WIDTH = `DATA_WIDTH
)(
    input wire clk, rst_n,relu_en,
    input wire signed[DATA_WIDTH-1:0] data_in,
    output reg signed [DATA_WIDTH-1:0] data_out
);

    always@(posedge clk or negedge rst_n) begin 
        if(!rst_n) begin 
            data_out <= {DATA_WIDTH{1'b0}};
        end else begin 
            if(relu_en && data_in[DATA_WIDTH-1]) begin 
                data_out <= {DATA_WIDTH{1'b0}};
            end else begin 
                data_out <= data_in;
            end
        end
    end
    
endmodule