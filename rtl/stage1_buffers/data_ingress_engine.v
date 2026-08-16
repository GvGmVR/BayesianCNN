//==============================================================================
// Module: data_ingress_engine.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 1 - Data Ingress & Write Marshalling
//------------------------------------------------------------------------------
// Purpose:
//   DMA Write Engine that receives flat DRAM burst streams and marshals them
//   into the on-chip BRAM bank array according to mapping function M(l,h,w,c).
//
// Architectural Inputs:
//   - Layer Geometry: Dimensions (H, W), channel tiles (C_tiles), temporal frames (L).
//   - DRAM Stream: 512-bit wide AXI burst words (PC=64 INT8 channels) with valid/ready.
//
// Architectural Outputs:
//   - BRAM Write Bus: 64 write enables, row address, and parallel 512-bit data.
//   - Ingress Handshake: Busy and Ingress Done status flags.
//
// Description:
//   Implements hardware coordinate tracking (l, h, w, c) to eliminate software 
//   packing overheads and load full channel slices into BRAM in 1 clock cycle.
/*
frame
  → height
      → width tile
          → channel tile
*/
//==============================================================================

`include "bcnn_pkg.vh"


module data_ingress_engine #(
    parameter DATA_WIDTH       = `DATA_WIDTH,
    parameter ADDR_WIDTH       = `ADDR_WIDTH,
    parameter DIM_WIDTH        = `DIM_WIDTH,
    parameter TILE_CNT_WIDTH   = `TILE_CNT_WIDTH,
    parameter PC               = `PC,
    parameter PV               = `PV
)(
    input wire clk,rst_n,start_ingress,

    output reg ingress_busy,
    output reg ingress_done,

    input wire [DIM_WIDTH-1:0] H,
    input wire [DIM_WIDTH-1:0] W,
    input wire [TILE_CNT_WIDTH-1:0] C_tiles,
    input wire [TILE_CNT_WIDTH-1:0] W_tiles,
    input wire [DIM_WIDTH-1:0] L_frames, // = 1 for 2D
    
    // AXI stream style
    input wire dram_valid,
    output reg dram_ready,
    input wire [(PC* DATA_WIDTH)-1:0] dram_data_in,

    output reg [(PV*PC)-1:0] we_a,
    output reg [ADDR_WIDTH-1:0] addr_a,
    output reg [(PV*PC*DATA_WIDTH)-1:0] din_a

);

// Ingress FSM States
    localparam STATE_IDLE  = 2'b00;
    localparam STATE_WRITE = 2'b01;
    localparam STATE_DONE  = 2'b10;

    reg [1:0] state;

// Logical Tensor Coordinate Counters
    reg [TILE_CNT_WIDTH-1:0] c_tile_cnt;
    reg [TILE_CNT_WIDTH-1:0] w_tile_cnt;
    reg [DIM_WIDTH-1:0]  h_cnt;
    reg [DIM_WIDTH-1:0]  l_cnt;

    wire [ADDR_WIDTH-1:0] calc_row_addr;

    assign calc_row_addr = (((l_cnt * H + h_cnt) * W_tiles + w_tile_cnt) * C_tiles) + c_tile_cnt;

    always @(posedge clk or negedge rst_n) begin
        if(rst_n) begin 
            state <= STATE_IDLE;
            ingress_busy <= 1'b0;
            ingress_done <= 1'b0;
            dram_ready <= 1'b0;
            we_a <= {(PV*PC){1'b0}};
            addr_a <= {ADDR_WIDTH{1'b0}};
            din_a <= {(PV*PC*DATA_WIDTH){1'b0}};
            c_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
            w_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
            h_cnt <= {DIM_WIDTH{1'b0}};
            l_cnt <= {DIM_WIDTH{1'b0}};
        end else begin
            case(state)
                STATE_IDLE: begin
                    if(start_ingress)begin
                        state <= STATE_WRITE;
                        ingress_busy <= 1'b1;
                        dram_ready <= 1'b1;
                        c_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
                        w_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
                        h_cnt <= {DIM_WIDTH{1'b0}};
                        l_cnt <= {DIM_WIDTH{1'b0}};
                    end
                end
                STATE_WRITE: begin
                    if(dram_valid && dram_ready) begin
                        addr_a <= calculated_row_addr;
                        we_a <= {(PV*PC){1'b1}};
                        din_a <= dram_data_in;

                        if (c_tile_cnt == (C_tiles - 1'b1)) begin
                            c_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
                            if (w_tile_cnt == (W_tiles - 1'b1)) begin
                                w_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
                                if (h_cnt == (H - 1'b1)) begin
                                    h_cnt <= {DIM_WIDTH{1'b0}};
                                    if (l_cnt == (L_frames - 1'b1)) begin
                                        l_cnt <= {DIM_WIDTH{1'b0}};
                                        state <= STATE_DONE;
                                        dram_ready <= 1'b0;
                                        ingress_busy <= 1'b0;
                                        ingress_done <= 1'b1;
                                    end else begin
                                        l_cnt <= l_cnt + 1'b1;
                                    end
                                end else begin
                                    h_cnt <= h_cnt + 1'b1;
                                end
                            end else begin
                                w_tile_cnt <= w_tile_cnt + 1'b1;
                            end
                        end else begin
                            c_tile_cnt <= c_tile_cnt + 1'b1;
                        end
                    end else begin
                        we_a <= {(PV * PC){1'b0}}; 
                    end
                end

                STATE_DONE: begin
                    we_a <= {(PV * PC){1'b0}};
                    ingress_done <= 1'b1;
                    state <= STATE_IDLE;
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule