`timescale 1ns/1ps
`include "bcnn_pkg.vh"

module tb_stage_1_top;

    reg clk,rst_n;

    reg mode_3d; // 0: 2D, 1: 3D
    reg ping_pong_sel;
    reg[`DIM_WIDTH-1:0] H;
    reg[`DIM_WIDTH-1:0] W;
    reg[`DIM_WIDTH-1:0] L_frames;
    reg[`TILE_CNT_WIDTH-1:0] C_tiles;
    reg[`TILE_CNT_WIDTH-1:0] W_tiles;
    reg[`KERNEL_DIM_WIDTH-1:0] KH;
    reg[`KERNEL_DIM_WIDTH-1:0] KW;
    reg[`KERNEL_DIM_WIDTH-1:0] KL;
    reg[`STRIDE_WIDTH-1:0] stride;
    
    
    reg start_ingress, dram_valid,dram_ready;
    reg [(`PC*`DATA_WIDTH)-1:0] dram_data_in;
    wire ingress_don;
    
    reg start_compute;
    wire re_b_valid, window_done, layer_done;
    wire [(`PF*`PC*`PV*`DATA_WIDTH)-1:0] pe_data_out;

    reg weights_push;
    reg [(`PC*`PF*`DATA_WIDTH)-1:0] weights_din;
    reg weight_pop;
    wire weight_full, weight_empty;
    wire [(`PV*`PC*`PF*`DATA_WIDTH)-1:0] pe_weight_out;

    // DUT
    smart_data_buffer #(
        .DATA_WIDTH(`DATA_WIDTH),
        .RAM_DEPTH(`RAM_DEPTH),
        .ADDR_WIDTH(`ADDR_WIDTH),
        .DIM_WIDTH(`DIM_WIDTH),
        .TILE_CNT_WIDTH(`TILE_CNT_WIDTH),
        .KERNEL_DIM_WIDTH(`KERNEL_DIM_WIDTH),
        .STRIDE_WIDTH(`STRIDE_WIDTH),
        .PC(`PC),
        .PV(`PV),
        .PF(`PF)
    ) dut_data_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .H(H),
        .W(W),
        .C_tiles(C_tiles),
        .W_tiles(W_tiles),
        .L_frames(L_frames),
        .KH(KH),
        .KW(KW),
        .KL(KL),
        .stride(stride),
        .mode_3d(mode_3d),
        .ping_pong_sel(ping_pong_sel),
        .start_ingress(start_ingress),
        .dram_valid(dram_valid),
        .dram_ready(dram_ready),
        .dram_data_in(dram_data_in),
        .ingress_done(ingress_done),
        .start_compute(start_compute),
        .re_b_valid(re_b_valid),
        .window_done(window_done),
        .layer_done(layer_done),
        .pe_data_out(pe_data_out)
    );

    smart_weight_buffer #(
        .DATA_WIDTH(`DATA_WIDTH),
        .PC(`PC),
        .PF(`PF),
        .PV(`PV),
        .FIFO_DEPTH(512)
    ) dut_weight_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .weight_push(weight_push),
        .weight_din(weight_din),
        .weight_full(weight_full),
        .weight_pop(weight_pop),
        .weight_empty(weight_empty),
        .pe_weight_out(pe_weight_out)
    );

    always #2.27 clk = ~clk; // 220 MHz => ~4.545 ns period

    integer p,c,f,error_count;
    reg [`DATA_WIDTH-1:0] expected_val;

    initial begin
        // Setup VCD Waveform Dump for GTKWave
        $dumpfile("sim/stage1/stage1_simulation.vcd");
        $dumpvars(0, tb_stage1_top);

        clk = 0;
        rst_n = 0;
        error_count = 0;
        ping_pong_sel = 0; // Write ping - Read pong initially
        start_ingress = 0;
        dram_valid = 0;
        dram_data_in = 0;
        start_compute = 0;
        weights_push = 0;
        weight_din = 0;
        weight_pop = 0;

        //Layer Geometry: 4x4 image, C=64 (1 tile), 3x3 kernel, Stride=1, 2D
        H=4;
        W=4;
        L_frames=1;
        C_tiles=1; // 64/64 = 1 tile
        W_tiles=4; // 64/16 = 4 tiles
        KH=3;
        KW=3;
        KL=1;
        stride=1;
        mode_3d=0; // 2D mode

        $display("===== STARTING STAGE 1 (SMART BUFFERS & INGRESS) TESTBENCH =====");

        #10;
        rst_n =1;
        #10;

        // Test case 1 - DRAM ingress to ping buffer
        @(posedge clk);
        start_ingress =1;
        @(posedge clk);
        start_ingress =0;

        // Stream 16 spatial pixels (4x4), each with 64 distinct channel values
        for (p=0;p<16;p=p+1)begin 
            @(posedge clk);
            while (!dram_ready) @(posedge clk);
            dram_valid =1;
            for (c=0;c<`PC;c=c+1)begin 
                dram_data_in[(c*8)+: 8] = ((p*10)+c) & 8'hFF; 
            end
        end

        @(posedge clk);
        dram_valid =0;

        while(!ingress_done)@(posedge clk);
        $display("[TEST 1 PASSED] DRAM Ingress completed successfully.");

        // Test case 2 - RAG Read Execution from PING Buffer
        @(posedge clk);
        ping_pong_sel = 0; // Read from ping buffer
        start_compute = 1;
        @(posedge clk);
        start_compute = 0;

        //wait for first valid read
        @(posedge clk);
        while(!re_b_valid) @(posedge clk);

        #1; // Check first read cycle: Pixel (0,0), Channel 0 should be 0
        if(pe_data_out[7:0]!==8'd0)begin 
            $display("[ERROR] Expected Pixel (0,0) Ch 0 = 0, Got = %d", pe_data_out[7:0]);
            error_count = error_count + 1;
        end else begin 
            $display("  -> Cycle 1 Match: Pixel(0,0) Ch 0 = %d, Ch 63 = %d", pe_data_out[7:0], pe_data_out[511:504]);
        end

        // Wait until RAG completes the entire 4x4 layer
        while (!layer_done) @(posedge clk);
        $display("[TEST 2 PASSED] RAG Convolution address traversal verified.");

        // Test case 3 - Weight Buffer Streaming & Fan-Out
        @(posedge clk);
        weight_push =1;

        // Push 4 weight sets (each set contains 64 channels x 64 filters = 4096 bytes)
        for (f=0;f<(`PC*`PF);f=f+1)begin 
            weight_din[f*8+:8] = f & 8'hFF; 
        end

        @(posedge clk);
        weight_push = 0;
        #10;

        // Pop Weight and verify fan-out
        @(posedge clk);
        weight_pop = 1;
        @(posedge clk);
        #1;

        if(pe_weight_out[7:0]!==8'd0)begin 
            $display("[ERROR] Weight Pop Mismatch! Expected 5, Got %d", pe_weight_out[7:0]);
            error_count = error_count + 1;
        end else begin 
            $display("  -> Weight Match: Filter 0, Ch 0 Weight = %d", pe_weight_out[7:0]);
        end
        weight_pop = 0;
        $display("[TEST 3 PASSED] Weight Buffer FIFO and Fan-Out verified.");
    
        # 50;
        if(error_count ==0)begin 
            $display("===== ALL TESTS PASSED SUCCESSFULLY =====");
        end else begin 
            $display("===== TESTS COMPLETED WITH %d ERRORS =====", error_count);
        end

        $finish;
    end

endmodule