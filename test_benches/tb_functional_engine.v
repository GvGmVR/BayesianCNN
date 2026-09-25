`timescale 1ns / 1ps
`include "bcnn_pkg.vh"

module tb_functional_engine;

    reg clk;
    reg rst_n;

    reg valid_in;
    reg sc_en;
    reg [`POOL_MODE_WIDTH-1:0] pool_mode;
    reg pool_win_done;
    reg [`POOL_CNT_WIDTH-1:0] pool_step;
    reg mcd_en;

    // Feature Map Data Inputs
    reg [(`PF * `PV * `DATA_WIDTH)-1:0] conv_features_in;
    reg [(`PF * `PV * `DATA_WIDTH)-1:0] sc_features_in;

    // Stage 2 Sampler Signals (Integrated)
    reg sampler_en;
    reg load_seed;
    reg [`LFSR_WIDTH-1:0] seed_in;
    wire [`PF-1:0] sampler_mask_out;
    wire sampler_mask_valid;
    wire sampler_mask_empty;
    wire sampler_mask_full;
    wire [`MASK_FIFO_ADDR:0] sampler_mask_count;
    wire stage4_mask_pop;

    // Stage 4 Outputs
    wire [(`PF * `PV * `DATA_WIDTH)-1:0] stage4_features_out;
    wire stage4_valid_out;


    // Instantiate Stage 2 (Bernoulli Sampler)
    bernoulli_sampler #(
        .PF(`PF),
        .LFSR_WIDTH(`LFSR_WIDTH),
        .N_LFSR(`N_LFSR),
        .FIFO_DEPTH(`MASK_FIFO_DEPTH),
        .FIFO_ADDR(`MASK_FIFO_ADDR)
    ) u_sampler (
        .clk(clk),
        .rst_n(rst_n),
        .sampler_en(sampler_en),
        .load_seed(load_seed),
        .seed_in(seed_in),
        .mask_pop(stage4_mask_pop),
        .mask_out(sampler_mask_out),
        .mask_valid(sampler_mask_valid),
        .mask_empty(sampler_mask_empty),
        .mask_full(sampler_mask_full),
        .mask_count(sampler_mask_count)
    );

    // Instantiate Stage 4 (Functional Engine DUT)
    functional_engine #(
        .DATA_WIDTH(`DATA_WIDTH),
        .PC(`PC),
        .PF(`PF),
        .PV(`PV),
        .POOL_MODE_WIDTH(`POOL_MODE_WIDTH),
        .POOL_CNT_WIDTH(`POOL_CNT_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .sc_en(sc_en),
        .pool_mode(pool_mode),
        .pool_win_done(pool_win_done),
        .pool_step(pool_step),
        .mcd_en(mcd_en),
        .conv_features_in(conv_features_in),
        .sc_features_in(sc_features_in),
        .mask_in(sampler_mask_out),
        .mask_valid(sampler_mask_valid),
        .mask_pop(stage4_mask_pop),
        .stage4_features_out(stage4_features_out),
        .stage4_valid_out(stage4_valid_out)
    );

    always #2.27 clk =~clk;

    integer f,error_count;

    reg signed [7:0] ch0_out, ch1_out;

    // Watchdog Timeout
    initial begin
        #60000;
        $display("\n[ERROR] Simulation timed out!");
        $finish;
    end

    initial begin
        $dumpfile("sim/stage4/stage4_simulation.vcd");
        $dumpvars(0, tb_functional_engine);

        clk= 0;
        rst_n = 0;
        valid_in = 0;
        sc_en = 0;
        pool_mode = `POOL_MODE_BYPASS;
        pool_win_done = 0;
        pool_step= 0;
        mcd_en = 0;
        conv_features_in = 0;
        sc_features_in = 0;
        sampler_en = 0;
        load_seed = 0;
        seed_in = { {(`LFSR_WIDTH-16){1'b0}}, 16'hACE1 };
        error_count = 0;

        $display("   STARTING STAGE 4 (FUNCTIONAL & DROPOUT ENGINE) TESTBENCH");


        #10
        rst_n =1;
        #10

        // TEST CASE 1: Reset State Check
        if (stage4_valid_out !== 1'b0) begin
            $display("[ERROR] Reset state invalid! stage4_valid_out should be 0.");
            error_count = error_count + 1;
        end else begin
            $display("[TEST 1 PASSED] Reset & Idle state verified.");
        end

        // TEST CASE 2: ResNet Shortcut (SC) Addition & Saturation Test
        $display("\n[TEST 2] Testing ResNet Shortcut (SC) Skip Addition & Saturation...");

        sc_en =1'b1;
        pool_mode = `POOL_MODE_BYPASS;
        mcd_en = 1'b0;

        // Channel 0: 20 + 15 = 35
        // Channel 1: 100 + 50 = 150 -> Clamped to +127 (Saturation)
        for(f=0;f<`PF;f=f+1)begin 
            if(f==0) begin 
                conv_features_in[f*8 +: 8] = 8'sd20;
                sc_features_in[f*8 +: 8]   = 8'sd15;
            end else if (f == 1) begin
                conv_features_in[f*8 +: 8] = 8'sd100;
                sc_features_in[f*8 +: 8]   = 8'sd50;
            end else begin
                conv_features_in[f*8 +: 8] = 8'sd10;
                sc_features_in[f*8 +: 8]   = 8'sd5;
            end
        end

        @(posedge clk);
        valid_in = 1'b1;
        @(posedge clk)
        valid_in = 1'b0;

        while (!stage4_valid_out) @(posedge clk);
        #1;

        ch0_out = stage4_features_out[7:0];
        ch1_out = stage4_features_out[15:8];
        $display("  -> SC Output Ch 0: %d (Expected: 35)", ch0_out);
        $display("  -> SC Output Ch 1: %d (Expected: 127 - Saturated)", ch1_out);

        if (ch0_out !== 8'sd35 || ch1_out !== 8'sd127) begin
            $display("[ERROR] SC Addition Failed! Ch0=%d, Ch1=%d", ch0_out, ch1_out);
            error_count = error_count + 1;
        end else begin
            $display("[TEST 2 PASSED] ResNet Shortcut addition and saturation verified.");
        end

        // TEST CASE 3: 2D Spatial Max Pooling Test (2x2 Window)
        $display("\n[TEST 3] Testing 2D Spatial Max Pooling (4 pixels: 10, 45, 30, 22)...");
        #20;

        sc_en = 1'b0;
        pool_mode = `POOL_MODE_MAX;
        mcd_en = 1'b0;

        // Step 0: Pixel = 10
        for (f = 0; f < `PF; f = f + 1) conv_features_in[f*8 +: 8] = 8'sd10;
        @(posedge clk);
        valid_in = 1'b1;
        pool_step = 2'd0;
        pool_win_done = 1'b0;

        // Step 1: Pixel = 45
        @(posedge clk);
        for (f = 0; f < `PF; f = f + 1) conv_features_in[f*8 +: 8] = 8'sd45;
        pool_step = 2'd2;

        // Step 2: Pixel = 30
        @(posedge clk);
        for (f = 0; f < `PF; f = f + 1) conv_features_in[f*8 +: 8] = 8'sd30;
        pool_step = 2'd2;

        // Step 3: Pixel = 22
        @(posedge clk);
        for (f = 0; f < `PF; f = f + 1) conv_features_in[f*8 +: 8] = 8'sd22;
        pool_step = 2'd3;
        pool_win_done = 1'b1;

        @(posedge clk);
        valid_in = 1'b0;
        pool_win_done = 1'b0;

        while(!stage4_valid_out)@(posedge clk);
        #1;

        ch0_out = stage4_features_out[7:0];
        $display("  -> Max Pool Output: %d (Expected: 45)", ch0_out);

        if (ch0_out !== 8'sd45) begin
            $display("[ERROR] Max Pool Failed! Expected: 45, Got: %d", ch0_out);
            error_count = error_count + 1;
        end else begin
            $display("[TEST 3 PASSED] 2D Max Pooling verified.");
        end

        // TEST CASE 4: 2D Spatial Average Pooling Test (2x2 Window)
        $display("\n[TEST 4] Testing 2D Spatial Avg Pooling (4 pixels: 12, 24, 36, 48)...");
        #20;

        sc_en = 1'b0;
        pool_mode = `POOL_MODE_AVG;
        mcd_en = 1'b0;

        // Step 0: 12
        for (f = 0; f < `PF; f = f + 1) conv_features_in[f*8 +: 8] = 8'sd12;
        @(posedge clk);
        valid_in = 1'b1;
        pool_step = 2'd0;
        pool_win_done = 1'b0;

        // Step 1: 24
        @(posedge clk);
        for (f = 0; f < `PF; f = f + 1) conv_features_in[f*8 +: 8] = 8'sd24;
        pool_step = 2'd1;

        // Step 2: 36
        @(posedge clk);
        for (f = 0; f < `PF; f = f + 1) conv_features_in[f*8 +: 8] = 8'sd36;
        pool_step = 2'd2;

        // Step 3: 48 (Window Complete: Sum=120, Avg=30)
        @(posedge clk);
        for (f = 0; f < `PF; f = f + 1) conv_features_in[f*8 +: 8] = 8'sd48;
        pool_step = 2'd3;
        pool_win_done = 1'b1;

        @(posedge clk);
        valid_in = 1'b0;
        pool_win_done = 1'b0;

        while (!stage4_valid_out) @(posedge clk);
        #1;

        ch0_out = stage4_features_out[7:0];
        $display("  -> Avg Pool Output: %d (Expected: 30)", ch0_out);

        if (ch0_out !== 8'sd30) begin
            $display("[ERROR] Avg Pool Failed! Expected: 30, Got: %d", ch0_out);
            error_count = error_count + 1;
        end else begin
            $display("[TEST 4 PASSED] 2D Average Pooling verified.");
        end

        // TEST CASE 5: Integrated Bayesian Dropout Masking Test
        $display("\n[TEST 5] Testing Integrated Bayesian Dropout Masking with Stage 2...");
        #20;
        // Start Stage 2 PRNG Sampler to fill FIFO with random masks
        @(posedge clk);
        sampler_en = 1'b1;
        while(sampler_mask_count < 5) @(posedge clk); //Wait for masks

        sc_en = 1'b0;
        pool_mode = `POOL_MODE_BYPASS;
        mcd_en = 1'b1;

        // Send feature value = 50 across all 64 channels
        for (f = 0; f < `PF; f = f + 1) conv_features_in[f*8 +: 8] = 8'sd50;

        @(posedge clk);
        valid_in = 1'b1;
        @(posedge clk);
        valid_in = 1'b0;

        while (!stage4_valid_out) @(posedge clk);
        #1;

        $display("  -> Applied Mask: 0x%016X", sampler_mask_out[63:0]);
        $display("  -> Output Ch 0: %d | Output Ch 1: %d", stage4_features_out[7:0], stage4_features_out[15:8]);

        // Verify that dropped channels (mask bit == 0) are 0, and kept channels (mask bit == 1) are 50
        for (f = 0; f < `PF; f = f + 1) begin
            if (sampler_mask_out[f] == 1'b1 && stage4_features_out[f*8 +: 8] !== 8'sd50) begin
                $display("[ERROR] Kept channel %d corrupted! Got: %d", f, stage4_features_out[f*8 +: 8]);
                error_count = error_count + 1;
            end else if (sampler_mask_out[f] == 1'b0 && stage4_features_out[f*8 +: 8] !== 8'sd0) begin
                $display("[ERROR] Dropped channel %d not zeroed! Got: %d", f, stage4_features_out[f*8 +: 8]);
                error_count = error_count + 1;
            end
        end

        if (error_count == 0) begin
            $display("[TEST 5 PASSED] Integrated Stage 2 & Stage 4 Dropout Masking verified.");
        end

        // Final Summary
        #50;

        if (error_count == 0) begin
            $display("   ALL STAGE 4 TEST CASES PASSED PERFECTLY! (0 ERRORS)        ");
        end else begin
            $display("   TESTBENCH COMPLETED WITH %d ERRORS.                        ", error_count);
        end

        $finish;
    end
    
endmodule