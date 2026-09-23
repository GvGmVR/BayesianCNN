//==============================================================================
// Module: sc_addition_unit.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 4 - ResNet Residual Shortcut Addition
//------------------------------------------------------------------------------
// Purpose:
//   Executes element-wise signed addition between the convolved feature map 
//   from Stage 3 and the original input feature map cached in the SC Buffer.
//
// Architectural Inputs:
//   - clk, rst_n        : Clock (220 MHz) and active-low reset.
//   - valid_in          : Valid strobe from Stage 3 Processing Engine.
//   - sc_en             : 1-bit control flag (1 = enable ResNet skip add, 0 = bypass).
//   - conv_features_in  : PF parallel 8-bit convolved channels from Stage 3 (PF x DATA_WIDTH bits).
//   - sc_features_in    : PF parallel 8-bit cached input channels from SC Buffer (PF x DATA_WIDTH bits).
//
// Architectural Outputs:
//   - sc_features_out   : PF parallel 8-bit saturated sum channels (PF x DATA_WIDTH bits).
//   - valid_out         : 1-cycle strobe indicating sc_features_out is valid.
//
// Description:
//   Computes Out = Conv(X) + X for ResNet layers with symmetrical saturation
//   clamping to [INT_MIN, INT_MAX] to prevent 8-bit signed overflow.
//==============================================================================

`include "bcnn_pkg.vh"

module sc_addition_unit #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC = `PC,
    parameter PF = `PF,
    parameter PV = `PV
)(
    input wire clk, rst_n,valid_in,sc_en,

    //Convolved features from Stage 3
    input wire [(PV*PF*DATA_WIDTH)-1:0] conv_features_in,

    //Cached input features form SC buffer
    input  wire [(PF * PV * DATA_WIDTH)-1:0] sc_features_in,

    //Shortcut addition output features
    output reg [(PF * PV * DATA_WIDTH)-1:0] sc_features_out,
    output reg valid_out
);

    localparam signed [DATA_WIDTH-1:0] INT_MAX = {1'b0,{(DATA_WIDTH-1){1'b1}}}; // +127
    localparam signed [DATA_WIDTH-1:0] INT_MIN = {1'b1,{(DATA_WIDTH-1){1'b0}}}; // -128

    wire signed [(PF*PV*DATA_WIDTH)-1:0] added_bus;

    genvar f;
    generate 
        for(f=0;f<(PF*PV);f=f+1) begin: GEN_SC_ADDERS 
            wire signed [DATA_WIDTH-1:0] conv_val;
            wire signed [DATA_WIDTH-1:0] sc_val;
            wire signed [DATA_WIDTH:0] raw_sum;

            assign conv_val = conv_features_in[(f+1)*DATA_WIDTH-1: F*DATA_WIDTH];
            assign sc_val = sc_features_in[(f+1)*DATA_WIDTH-1: F*DATA_WIDTH];
            assign raw_sum = conv_val+sc_val;

            //symmetric saturation
            wire signed [DATA_WIDTH-1:0] clamped_sum;
            assign clamped_sum = (raw_sum > $signed({{1'b0},INT_MAX})) ? INT_MAX: (raw_sum < $signed({{1'b1},INT_MIN})) ? INT_MIN : raw_sum[DATA_WIDTH-1:0];
        
            //If sc_en=1 apply sum, else pass conv_val unaltered
            assign added_bus[(f+1)*DATA_WIDTH-1: f*DATA_WIDTH] = sc_en ? clamped_sum : conv_val;
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin 
            sc_features_out <= {(PF*PV*DATA_WIDTH){1'b0}};
            valid_out <= 1'b0;
        end else begin 
            sc_features_out <= added_bus;
            valid_out <= valid_in;
        end
    end
    
endmodule