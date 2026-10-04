//==============================================================================
// Module: crossbar_switch.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 1 - Spatial Vector Alignment
//------------------------------------------------------------------------------
// Purpose:
//   Connects the PV x PC RAM bank read outputs to the Tree Fan-Out stage.
//
// Architectural Inputs:
//   - Raw parallel data from PV x PC on-chip RAM banks.
//
// Architectural Outputs:
//   - Aligned PV x PC data vector for the PE processing engine.
//
// Description:
//   Provides spatial alignment across vector groups during sliding-window shifts.
//==============================================================================

`include "bcnn_pkg.vh"

module crossbar_switch #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC         = `PC,
    parameter PV         = `PV
)(
    input wire clk,rst_n,enable,

    input wire [(PC*PV*DATA_WIDTH)-1:0] raw_bank_data,

    output reg [(PC*PV*DATA_WIDTH)-1:0] aligned_data
);
    // No clock based - no delay
    assign aligned_data = enable ? raw_bank_data : {(PV * PC * DATA_WIDTH){1'b0}};

endmodule