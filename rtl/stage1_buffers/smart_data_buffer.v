//==============================================================================
// Module: smart_data_buffer.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 1 - Top Data Memory Wrapper
//------------------------------------------------------------------------------
// Purpose:
//   Top-level wrapper for the entire feature data memory subsystem.
//
// Architectural Inputs:
//   - Layer geometries, Ping-Pong buffer select, DRAM AXI ingress stream.
//   - Compute control triggers from Main Accelerator Controller.
//
// Architectural Outputs:
//   - Broadcasted PE data matrix (PF x PV x PC bytes).
//   - Handshake status flags (ingress_done, window_done, layer_done).
//
// Description:
//   Integrates Ingress DMA, RAG, Crossbar, Tree Fan-Out, and two sets of 
//   PV x PC RAM banks to provide stall-free Ping-Pong double buffering.
//==============================================================================

`include "bcnn_pkg.vh"

module smart_data_buffer #(
    parameter DATA_WIDTH       = `DATA_WIDTH,
    parameter RAM_DEPTH        = `RAM_DEPTH,
    parameter ADDR_WIDTH       = `ADDR_WIDTH,
    parameter DIM_WIDTH        = `DIM_WIDTH,
    parameter TILE_CNT_WIDTH   = `TILE_CNT_WIDTH,
    parameter KERNEL_DIM_WIDTH = `KERNEL_DIM_WIDTH,
    parameter STRIDE_WIDTH     = `STRIDE_WIDTH,
    parameter PC               = `PC,
    parameter PV               = `PV,
    parameter PF               = `PF
)(
    input wire clk,rst_n,
    
    input wire ping_pong_sel, // 0- Write ping - Read Pong // 1- Write Pong - Read ping

    input wire [DIM_WIDTH-1:0] H,
    input wire [DIM_WIDTH-1:0] W,
    input wire [DIM_WIDTH-1:0] L_frames,
    input wire [TILE_CNT_WIDTH-1:0] C_tiles,
    input wire [TILE_CNT_WIDTH-1:0] W_tiles,
    input wire [KERNEL_DIM_WIDTH-1:0] KH,
    input wire [KERNEL_DIM_WIDTH-1:0] KW,
    input wire [KERNEL_DIM_WIDTH-1:0] KL,
    input wire [STRIDE_WIDTH-1:0] stride,
    input wire mode_3d, // 0: 2D, 1: 3D

    // Ingress engine interface
    input wire start_ingress,
    input wire dram_valid,
    input wire [(PC*DATA_WIDTH)-1:0] dram_data_in,
    output wire dram_ready,
    output wire ingress_done,

    //egress engine interface
    input wire start_compute,
    output wire re_b_valid,
    output wire window_done,
    output wire layer_done,
    output wire [(PF*PV*PC*DATA_WIDTH)-1:0] pe_data_out
);

    //ingress write wires
    wire [(PV*PC)-1:0] ing_we_a;
    wire [ADDR_WIDTH-1:0] ing_addr_a;
    wire [(PV*PC*DATA_WIDTH)-1:0] ing_din_a;
    wire ingress_busy;

    // RAG wires
    wire [ADDR_WIDTH-1:0] rag_addr_b;
    wire rag_re_b;

    // RAM read data bus bank wires
    wire [(PV*PC*DATA_WIDTH)-1:0] raw_ram_out;
    wire [(PV*PC*DATA_WIDTH)-1:0] aligned_ram_out;

    // Ping bank array signals
    wire [(PV*PC)-1:0] ping_we_a;
    wire [ADDR_WIDTH-1:0] ping_addr_a;
    wire [(PV*PC*DATA_WIDTH)-1:0] ping_din_a;
    wire [ADDR_WIDTH-1:0] ping_addr_b;
    wire ping_re_b;
    wire [(PV*PC*DATA_WIDTH)-1:0] ping_dout_b;

    //Pong bank array signals
    wire [(PV*PC)-1:0] pong_we_a;
    wire [ADDR_WIDTH-1:0] pong_addr_a;
    wire [(PV*PC*DATA_WIDTH)-1:0] pong_din_a;
    wire [ADDR_WIDTH-1:0] pong_addr_b;
    wire pong_re_b;
    wire [(PV*PC*DATA_WIDTH)-1:0] pong_dout_b;

    //Ping-Pong routing
    assign ping_we_a = (ping_pong_sel==1'b0) ? ing_we_a : {(PV*PC){1'b0}};     // Write ping
    assign ping_addr_a = (ping_pong_sel==1'b0) ? ing_addr_a : {ADDR_WIDTH{1'b0}};
    assign ping_din_a = (ping_pong_sel==1'b0) ? ing_din_a : {(PV*PC*DATA_WIDTH){1'b0}};
    assign ping_addr_b = (ping_pong_sel==1'b1) ? rag_addr_b : {ADDR_WIDTH{1'b0}};     /// Read pong
    assign ping_re_b = (ping_pong_sel==1'b1) ? rag_re_b : 1'b0;

    assign pong_we_a = (ping_pong_sel == 1'b1) ? ing_we_a   : {(PV*PC){1'b0}};  // Write pong
    assign pong_addr_a = (ping_pong_sel == 1'b1) ? ing_addr_a : {ADDR_WIDTH{1'b0}};
    assign pong_din_a = (ping_pong_sel == 1'b1) ? ing_din_a  : {(PV*PC*DATA_WIDTH){1'b0}};
    assign pong_addr_b = (ping_pong_sel == 1'b0) ? rag_addr_b : {ADDR_WIDTH{1'b0}};   // Read ping
    assign pong_re_b = (ping_pong_sel == 1'b0) ? rag_re_b   : 1'b0;

    assign raw_ram_out = (ping_pong_sel == 1'b1) ? ping_dout_b : pong_dout_b;
    assign re_b_valid = rag_re_b;


    // Data ingress engine - Port A write side
    data_ingress_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DIM_WIDTH(DIM_WIDTH),
        .TILE_CNT_WIDTH(TILE_CNT_WIDTH),
        .PC(PC),
        .PV(PV)
    ) u_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start_ingress(start_ingress),
        .ingress_busy(ingress_busy),
        .ingress_done(ingress_done),
        .H(H),
        .W(W),
        .C_tiles(C_tiles),
        .W_tiles(W_tiles),
        .L_frames(L_frames),
        .dram_ready(dram_ready),
        .dram_valid(dram_valid),
        .dram_data_in(dram_data_in),
        .we_a(ing_we_a),
        .addr_a(ing_addr_a),
        .din_a(ing_din_a)
    );

    // Read address generator - Port B read side
    read_addr_gen #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DIM_WIDTH(DIM_WIDTH),
        .TILE_CNT_WIDTH(TILE_CNT_WIDTH),
        .KERNEL_DIM_WIDTH(KERNEL_DIM_WIDTH),
        .STRIDE_WIDTH(STRIDE_WIDTH),
        .PV(PV)
    ) u_rag (
        .clk(clk),
        .rst_n(rst_n),
        .start_layer(start_compute),
        .mode_3d(mode_3d),
        .H(H),
        .W(W),
        .C_tiles(C_tiles),
        .W_tiles(W_tiles),
        .KH(KH),
        .KW(KW),
        .KL(KL),
        .stride(stride),
        .read_addr(rag_addr_b),
        .re_b(rag_re_b),
        .window_done(window_done),
        .layer_done(layer_done)
    );

    // Ping and Pong RAM Bank Arrays (PV x PC banks each)

    genvar b;
    generate
        for(b=0;b<(PC*PV);b=b+1) begin: GEN_PING_PONG_RAMS
            // PING Bank
            ram_bank #(
                .DATA_WIDTH(DATA_WIDTH),
                .RAM_DEPTH(RAM_DEPTH),
                .ADDR_WIDTH(ADDR_WIDTH)
            ) u_ping_bank (
                .clk(clk),
                .rst_n(rst_n),
                .we_a(ping_we_a[b]),
                .addr_a(ping_addr_a),
                .din_a(ping_din_a[(b+1)*DATA_WIDTH-1 : b*DATA_WIDTH]),
                .re_b(ping_re_b),
                .addr_b(ping_addr_b),
                .dout_b(ping_dout_b[(b+1)*DATA_WIDTH-1 : b*DATA_WIDTH])
            );

            // PONG Bank
            ram_bank #(
                .DATA_WIDTH(DATA_WIDTH),
                .RAM_DEPTH(RAM_DEPTH),
                .ADDR_WIDTH(ADDR_WIDTH)
            ) u_pong_bank (
                .clk(clk),
                .rst_n(rst_n),
                .we_a(pong_we_a[b]),
                .addr_a(pong_addr_a),
                .din_a(pong_din_a[(b+1)*DATA_WIDTH-1 : b*DATA_WIDTH]),
                .re_b(pong_re_b),
                .addr_b(pong_addr_b),
                .dout_b(pong_dout_b[(b+1)*DATA_WIDTH-1 : b*DATA_WIDTH])
            );
        end
    endgenerate

    // Crossbar switch
    crossbar_switch #(
        .DATA_WIDTH(DATA_WIDTH),
        .PC(PC),
        .PV(PV)
    ) u_crossbar (
        .clk(clk),
        .rst_n(rst_n),
        .enable(rag_re_b),
        .raw_bank_data(raw_ram_out),
        .aligned_data(aligned_ram_out)
    );

    // Tree Fan-Out
    tree_fanout #(
        .DATA_WIDTH(DATA_WIDTH),
        .PC(PC),
        .PV(PV),
        .PF(PF)
    ) u_fanout (
        .data_in(aligned_ram_out),
        .data_out(pe_data_out)
    );

endmodule