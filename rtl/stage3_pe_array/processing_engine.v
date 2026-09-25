//==============================================================================
// Module: processing_engine.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 3 - Top Processing Engine (PE Array)
//------------------------------------------------------------------------------
// Purpose:
//   Instantiates PF parallel Processing Units (PUs) to compute PF = 64 output
//   feature channels concurrently in 1 clock cycle.
//
// Architectural Inputs:
//   - clk, rst_n      : Clock (220 MHz) and active-low reset.
//   - valid_in        : Input valid strobe from Stage 1 buffers.
//   - window_done     : Strobe from RAG indicating end of sliding window.
//   - relu_en         : 1-bit flag to enable or bypass ReLU activation.
//   - pe_data_in      : Replicated input feature bus from Stage 1 Tree Fan-Out.
//   - pe_weight_in    : Parallel filter weight bus from Stage 1 Weight Buffer.
//   - quant_scale     : 16-bit fixed-point scale factor.
//   - quant_shift     : 5-bit right shift amount.
//   - quant_bias      : 32-bit layer bias value.
//
// Architectural Outputs:
//   - pe_features_out : PF x PV parallel 8-bit output features (64 bytes = 512 bits).
//   - features_valid  : 1-cycle strobe indicating pe_features_out is ready.
//
// Description:
//   Top wrapper for the NNE Compute Core (Figure 4 in Fan et al.), executing
//   4,096 INT8 multiplications and 64 complete filter convolutions simultaneously.
//==============================================================================

`include "../bcnn_pkg.vh"

module processing_engine #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC = `PC,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter LOG2_PC = `LOG2_PC,
    parameter MULT_OUT_WIDTH = `MULT_OUT_WIDTH,
    parameter ADDR_TREE_WIDTH = `ADDR_TREE_WIDTH,
    parameter ACCUM_WIDTH = `ACCUM_WIDTH,
    parameter QUANT_SCALE_WIDTH = `QUANT_SCALE_WIDTH,
    parameter QUANT_SHIFT_WIDTH = `QUANT_SHIFT_WIDTH,
    parameter QUANT_BIAS_WIDTH = `QUANT_BIAS_WIDTH
)(
    input wire clk,rst_n,valid_in,window_done,relu_en,

    // Parallel Data and weights from Stage 1 Tree Fan-Out (PF x PV x PC bytes)
    input wire [(PF*PV*PC*DATA_WIDTH-1):0] pe_data_in,
    input wire [(PF*PV*PC*DATA_WIDTH-1):0] pe_weight_in,

    //quantization parameters
    input wire signed [QUANT_SCALE_WIDTH-1:0] quant_scale,
    input wire [QUANT_SHIFT_WIDTH-1:0] quant_shift,
    input wire signed [QUANT_BIAS_WIDTH-1:0] quant_bias,

    // Final Processed Output (PF Filter Pixels)
    output wire signed [PF*PV*DATA_WIDTH-1:0] pe_features_out,
    output wire features_valid
);

    wire [PF-1:0] pu_valid_bus;

    //Instantiate PF = 64 Parallelprocessing units
    genvar f;
    generate 
        for(f=0; f<PF; f=f+1) begin: GEN_PUS 
            //Slice PC channels  for filter
            wire [(PC*DATA_WIDTH)-1:0] pu_data_slice;
            wire [(PC*DATA_WIDTH)-1:0] pu_weight_slice;

            assign pu_data_slice = pe_data_in[(f+1)*(PC*DATA_WIDTH)-1: f*(PC*DATA_WIDTH)];
            assign pu_weight_slice = pe_weight_in[(f+1)*(PC*DATA_WIDTH)-1: f*(PC*DATA_WIDTH)];

            processing_unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .PC(PC),
                .LOG2_PC(LOG2_PC),
                .MULT_OUT_WIDTH(MULT_OUT_WIDTH),
                .ADDR_TREE_WIDTH(ADDR_TREE_WIDTH),
                .ACCUM_WIDTH(ACCUM_WIDTH),
                .QUANT_SCALE_WIDTH(QUANT_SCALE_WIDTH),
                .QUANT_SHIFT_WIDTH(QUANT_SHIFT_WIDTH),
                .QUANT_BIAS_WIDTH(QUANT_BIAS_WIDTH)
            ) u_pu (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in),
                .window_done(window_done),
                .relu_en(relu_en),
                .data_vec(pu_data_slice),
                .weight_vec(pu_weight_slice),
                .quant_scale(quant_scale),
                .quant_shift(quant_shift),
                .quant_bias(quant_bias),
                .feature_out(pe_features_out[(f+1)*DATA_WIDTH-1 : f*DATA_WIDTH]),
                .feature_valid(pu_valid_bus[f])
            );
        end
    endgenerate

    // All PUs are synchronous, so valid is asserted when PU 0 completes
    assign features_valid = pu_valid_bus[0];

endmodule