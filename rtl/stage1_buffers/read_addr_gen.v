//==============================================================================
// Module: read_addr_gen.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 1 - Sliding Window Read Address Generation
//------------------------------------------------------------------------------
// Purpose:
//   Cascading 6-stage nested counter FSM implementing Algorithm 1 from Fan et al.
//   Traverses 2D/3D convolution sliding windows and drives Port B of BRAM banks.
//
// Architectural Inputs:
//   - Geometry: Feature map size (H, W), kernel shape (KH, KW, KL), stride, C_tiles.
//   - Control: Layer start trigger and 2D/3D mode selector.
//
// Architectural Outputs:
//   - BRAM Read Bus: Row address and read enable signal for all 64 RAM banks.
//   - Synchronization: Window Done (end of KHxKW patch) and Layer Done pulses.
//
// Description:
//   Uses the common memory map M(l,h,w,c) to read complete 64-channel sliding
//   window pixels every clock cycle without host CPU intervention.
//==============================================================================

`include "bcnn_pkg.vh"

module read_addr_gen#(
    parameter ADDR_WIDTH       = `ADDR_WIDTH,
    parameter DIM_WIDTH        = `DIM_WIDTH,
    parameter TILE_CNT_WIDTH   = `TILE_CNT_WIDTH,
    parameter KERNEL_DIM_WIDTH = `KERNEL_DIM_WIDTH,
    parameter STRIDE_WIDTH     = `STRIDE_WIDTH,
    parameter PV               = `PV
)(
    input wire clk,rst_n,start_layer, 
    input wire mode_3d, // 0: 2D, 1: 3D

    input wire [DIM_WIDTH-1:0] H,
    input wire [DIM_WIDTH-1:0] W,
    input wire [TILE_CNT_WIDTH-1:0] C_tiles,
    input wire [TILE_CNT_WIDTH-1:0] W_tiles,
    input wire [KERNEL_DIM_WIDTH-1:0] KH,
    input wire [KERNEL_DIM_WIDTH-1:0] KW,
    input wire [KERNEL_DIM_WIDTH-1:0] KL,
    input wire [STRIDE_WIDTH-1:0] stride,

    output reg [ADDR_WIDTH-1:0] read_addr,
    output reg re_b,
    output reg window_done,
    output reg layer_done
);

    //State definitions
    localparam STATE_IDLE = 2'b00;
    localparam STATE_READ = 2'b01;
    localparam STATE_DONE = 2'b10;

    reg [1:0] state;

    //COunters for looping
    reg [TILE_CNT_WIDTH-1:0] c_tile_cnt;
    reg [TILE_CNT_WIDTH-1:0] w_tile_cnt;  
    reg [DIM_WIDTH-1:0] h_cnt;
    reg [KERNEL_DIM_WIDTH-1:0] kw_cnt;  
    reg [KERNEL_DIM_WIDTH-1:0] kh_cnt;
    reg [KERNEL_DIM_WIDTH-1:0] kl_cnt;  // Kernel for Temporal - 3D conv

    wire [DIM_WIDTH-1:0] h_in;
    wire [DIM_WIDTH-1:0] w_in;
    wire [DIM_WIDTH-1:0] l_in;
    wire [TILE_CNT_WIDTH-1:0] w_in_tile;

    assign h_in = (h_cnt*stride)+ kh_cnt;
    assign w_in = (w_tile_cnt*PV*stride)+kw_cnt;
    assign l_in = kl_cnt;
    assign w_in_tile = (w_in/PV);

    wire [ADDR_WIDTH-1:0] calc_read_addr;
    assign calc_read_addr = (((l_in*H + h_in)*W_tiles+w__in_tile)*C_tiles)+c_cnt;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)begin
            state <= STATE_IDLE;
            c_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
            w_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
            kw_cnt <= {KERNEL_DIM_WIDTH{1'b0}};
            kh_cnt <= {KERNEL_DIM_WIDTH{1'b0}};
            kl_cnt <= {KERNEL_DIM_WIDTH{1'b0}};
            read_addr <= {ADDR_WIDTH{1'b0}};
            h_cnt <= {DIM_WIDTH{1'b0}};
            re_b <= 1'b0;
            window_done <= 1'b0;
            layer_done <= 1'b0;
        end else begin 
            case(state)
                STATE_IDLE: begin 
                    layer_done<=1'b0;
                    window_done<=1'b0;
                    re_b <= 1'b0;
                    if(start_layer) begin
                        c_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
                        w_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
                        kw_cnt <= {KERNEL_DIM_WIDTH{1'b0}};
                        kh_cnt <= {KERNEL_DIM_WIDTH{1'b0}};
                        kl_cnt <= {KERNEL_DIM_WIDTH{1'b0}};
                        h_cnt <= {DIM_WIDTH{1'b0}};
                        STATE <= STATE_READ;
                        re_b <= 1'b1;
                    end
                end

                STATE_READ: begin 
                    read_addr <=calc_read_addr;
                    re_b <= 1'b1;
                    window_done <= 1'b1;

                    if(c_tile_cnt == (C_tiles -1'b1))begin 
                        c_tile_cnt <= {TILE_CNT_WIDTH{1'b0}};
                        if(kw_cnt == (KW -1'b1))begin 
                            kw_cnt <= {KERNEL_DIM_WIDTH{1'b0}};
                            if(kh_cnt == (KH-1'b1)) begin 
                                kh_cnt <= {KERNEL_DIM_WIDTH{1'b0}};
                                if((!mode_3d)||(kl== (KL-1'b1)))begin 
                                    kl_cnt <= {KERNEL_DIM_WIDTH{1'b0}};
                                    window_done <= 1'b1;
                                    if(w_tile_cnt == (W_tiles -1'b1))begin 
                                        w_tile_cnt <= {TILE_CNT_WDTH{1'b0}};
                                        if(h_cnt == (H-1'b1))begin 
                                            h_cnt <={DIM_WIDTH{1'b0}};
                                            state <= STATE_DONE;
                                            re_b <= 1'b0;
                                            layer_done <= 1'b1;
                                        end else begin 
                                            h_cnt <= h_cnt + 1'b1;
                                        end
                                    end else begin 
                                        w_tile_cnt <= w_tile_cnt + 1'b1;
                                    end
                                end else begin 
                                    kl_cnt <= kl_cnt + 1'b1;
                                end
                            end else begin 
                                kh_cnt <= kh_cnt + 1'b1;
                            end
                        end else begin 
                            kw_cnt <= kw_cnt + 1'b1;
                        end
                    end else begin 
                        c_tile_cnt <= c_tile_cnt + 1'b1;
                    end
                end

                STATE_DONE: begin 
                    re_b <=1'b0;
                    window_done <= 1'b0;
                    layer_done <=1'b0;
                    state <= STATE_IDLE;
                end

                default: begin 
                    state<= STATE_IDLE;
                end
            endcase
        end
    end
endmodule