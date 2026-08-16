//==============================================================================
// Module: smart_weight_buffer.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 1 - Top Weight Memory Wrapper
//------------------------------------------------------------------------------
// Purpose:
//   Top-level wrapper for the filter weight memory subsystem.
//
// Architectural Inputs:
//   - Parallel DRAM weight ingress bus (PC x PF bytes).
//   - Weight Pop enable from PE compute controller.
//
// Architectural Outputs:
//   - Expanded weight matrix duplicated PV times for spatial parallelism.
//   - Full and Empty FIFO status flags.
//
// Description:
//   Instantiates PC x PF parallel FIFOs and replicates weights across PV spatial
//   lanes to feed the 2D MAC array in 1 clock cycle.
//==============================================================================

`include "bcnn_pkg.vh"

module smart_weight_buffer #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC         = `PC,
    parameter PF         = `PF,
    parameter PV         = `PV,
    parameter FIFO_DEPTH = `FIFO_DEPTH
)(
    input clk,rst_n,

    // ingress from DRAM
    input wire weight_push,
    input wire [(PC*PF*DATA_WIDTH)-1:0] weight_din,
    input wire weight_full,

    // Weight usage - From PE controller
    input wire weight_pop,
    output wire [(PV*PC*PF*DATA_WIDTH)-1:0] pe_weight_out,
    output wire weight_empty
);

    wire [(PC*PF*DATA_WIDTH)-1:0] raw_weight_bus;
    wire [(PC*PF)-1:0] fifo_full_bus;
    wire [(PC*PF)-1:0] fifo_empty_bus;

    assign weight_full = |fifo_full_bus;
    assign weight_empty = |fifo_empty_bus;

    // PC x PF Parallel Weight FIFOs
    genvar idx;
    generate
        for(idx=0;idx<(PC*PF);idx=idx+1)begin: GEN_WEIGHT_FIFOS
            weight_fifo #(
                .DATA_WIDTH(DATA_WIDTH),
                .FIFO_DEPTH(FIFO_DEPTH)
            ) u_weight_fifo (
                .clk(clk),
                .rst_n(rst_n),
                .push(weight_push),
                .din(weight_din[(idx+1)*DATA_WIDTH-1 : idx*DATA_WIDTH]),
                .full(fifo_full_bus[idx]),
                .pop(weight_pop),
                .dout(raw_weight_bus[(idx+1)*DATA_WIDTH-1 : idx*DATA_WIDTH]),
                .empty(fifo_empty_bus[idx])
            );
        end
    endgenerate

    // Tree fanout PV times
    gevar v;
    generate
        for(v=0;v<PV;v=v+1)begin: GEN_WEIGHT_PV_FANOUT
            assign pe_weight_out[(v+1)*(PC*PF*DATA_WIDTH)-1: (v)*(PC*PF*DATA_WIDTH)] = raw_weight_out;
        end
    endgenerate

endmodule