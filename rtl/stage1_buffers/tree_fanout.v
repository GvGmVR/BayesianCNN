//==============================================================================
// Module: tree_fanout.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 1 - Filter Parallelism Broadcast
//------------------------------------------------------------------------------
// Purpose:
//   Replicates input feature vectors across the Filter Parallelism (PF) dimension.
//
// Architectural Inputs:
//   - Aligned PV x PC input feature vector (e.g., 64 bytes).
//
// Architectural Outputs:
//   - Expanded matrix of PF copies (PF x PV x PC bytes, e.g., 4096 bytes).
//
// Description:
//   Broadcasts shared input feature map pixels to PF parallel filter processing 
//   units simultaneously with zero-latency combinational fan-out.
//==============================================================================

`include "bcnn_pkg.vh"

module tree_fanout #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC         = `PC,
    parameter PV         = `PV,
    parameter PF         = `PF
)(
    input wire [PV*PC*DATA_WIDTH-1:0] data_in,

    output wire [PF*PC*PV*DATA_WIDTH-1:0] data_out
);

    genvar f;
    generate
        for(f=0; f<PF; f=f+1) begin : GENERATE_FANOUT
            assign data_out[(f+1)*(PC*PV*DATA_WIDTH)-1:f*(PC*PV*DATA_WIDTH)] = data_in;
        end
    endgenerate


endmodule