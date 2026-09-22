//==============================================================================
// Module: multiplier_array.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 3 - Parallel Channel Multipliers
//------------------------------------------------------------------------------
// Purpose:
//   Performs PC parallel signed multiplications between input features and filter weights.
//
// Architectural Inputs:
//   - data_vec   : PC parallel signed activations from Smart Data Buffer (PC x DATA_WIDTH bits).
//   - weight_vec : PC parallel signed weights from Smart Weight Buffer (PC x DATA_WIDTH bits).
//
// Architectural Outputs:
//   - prod_vec   : PC parallel signed products (PC x MULT_OUT_WIDTH bits).
//
// Description:
//   Unrolls PC signed 8x8 multipliers. Synthesis tools map these pairs into 
//   FPGA DSP blocks via dual-multiplier DSP packing.
//==============================================================================

`include "bcnn_pkg.vh"

module multiplier_array #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC = `PC,
    parameter MULT_OUT_WIDTH = `MULT_OUT_WIDTH
)(
    input wire [(PC*DATA_WIDTH)-1:0] data_vec,
    input wire [(PC*DATA_WIDTH)-1:0] weight_vec,

    output wire [(PC*MULT_OUT_WIDTH)-1:0] prod_vec
);

    genvar i;
    generate
        for (i=0; i<PC;i=i+1) begin: GEN_MULT
            wire signed [DATA_WIDTH-1:0] in_elem;
            wire signed [DATA_WIDTH-1:0] w_elem;
            wire signed [MULT_OUT_WIDTH-1:0] prod_elem;

            assign in_elem = data_vec[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH];
            assign w_elem = weight_vec[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH];
            assign prod_elem = in_elem * w_elem;

            assign prod_vec[(i+1)*MULT_OUT_WIDTH-1:i*MULT_OUT_WIDTH] = prod_elem;
        end
    endgenerate
endmodule