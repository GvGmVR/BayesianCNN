`include "bcnn_pkg.vh"

module read_addr_gen#(
    parameter ADDR_WIDTH       = `ADDR_WIDTH,
    parameter DIM_WIDTH        = `DIM_WIDTH,
    parameter TILE_CNT_WIDTH   = `TILE_CNT_WIDTH,
    parameter KERNEL_DIM_WIDTH = `KERNEL_DIM_WIDTH,
    parameter STRIDE_WIDTH     = `STRIDE_WIDTH,
    parameter PV               = `PV
)(
    input wire clk,rst_n,start_layer, mode_3d,

    input wire [DIM_WIDTH-1:0] H,
    input wire [DIM_WIDTH-1:0] W,
    input wire [TILE_CNT_WIDTH-1:0] C_tiles,
    input wire [TILE_CNT_WIDTH-1:0] W_tiles,
    input wire [KERNEL_DIM_WIDTH-1:0] KH,
    input wire [KERNEL_DIM_WIDTH-1:0] KW,
    input wire [KERNEL_DIM_WIDTH-1:0] KL,
    input wire [STRIDE_WIDTH-1:0] stride,

    output reg [ADDR_WIDTH-1:0] read_addr,
    output reg addr_valid,
    output reg window_done,
    output reg layer_done
);

//State definitions
localparam STATE_IDLE = 2'b00;
localparam STATE_RUN = 2'b01;
localparam STATE_DONE = 2'b10;

reg [1:0] state;

//COunters for looping
reg [DIM_WIDTH-1:0] h_cnt;
reg [TILE_CNT_WIDTH-1:0] w_cnt;  
reg [TILE_CNT_WIDTH-1:0] c_cnt;
reg [KERNEL_DIM_WIDTH-1:0] kw_cnt;  
reg [KERNEL_DIM_WIDTH-1:0] kh_cnt;
reg [KERNEL_DIM_WIDTH-1:0] kl_cnt;  // Kernel for Temporal - 3D conv

wire [DIM_WIDTH-1:0] h_pixel;
wire [DIM_WIDTH-1:0] w_pixel;
wire [ADDR_WIDTH-1:0] calc_addr;

assign h_pixel = (h_cnt*stride)+ kh_cnt;
assign w_pixel = (w_cnt*PV*stride)+kw_cnt;

assign calc_addr = (((kl_cnt*H + h_pixel)*W+w_pixel)*C_tiles)+c_cnt;


endmodule