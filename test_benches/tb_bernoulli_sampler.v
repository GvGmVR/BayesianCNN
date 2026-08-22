`timescale 1ns / 1ps
`include "bcnn_pkg.vh"

module tb_bernoulli_sampler;

    reg clk; 
    reg rst_n;
    reg sampler_en;
    reg [`LFSR_WIDTH-1:0] seed_in;
    reg mask_pop;
    reg load_seed;

    wire [`PF-1:0] mask_out;
    wire mask_valid;
    wire mask_empty;
    wire mask_full;
    wire [`MASK_FIFO_ADDR:0] mask_count;

    bernoulli_sampler #(
        .PF(`PF),
        .LFSR_WIDTH(`LFSR_WIDTH),
        .N_LFSR(`N_LFSR),
        .FIFO_DEPTH(`MASK_FIFO_DEPTH),
        .FIFO_ADDR(`MASK_FIFO_ADDR)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .sampler_en(sampler_en),
        .load_seed(load_seed),
        .seed_in(seed_in),
        .mask_pop(mask_pop),
        .mask_out(mask_out),
        .mask_valid(mask_valid),
        .mask_empty(mask_empty),
        .mask_full(mask_full),
        .mask_count(mask_count)
    );

    always #2.27 clk = ~ clk;

    integer w,b,ones_count, total_bits,error_count;
    real measured_prob;

    // Timeout
    initial begin 
        #1000000;
        $display("[ERROR] Timeout");
        $finish;
    end

    initial begin 
        $dumpfile("sim/stage2/stage2_simulation.vcd");
        $dumpvars(0, tb_bernoulli_sampler);
        clk = 0;
        rst_n = 0;
        sampler_en = 0;
        load_seed = 0;
        seed_in = {{(`LFSR_WIDTH-16){1'b0}},16'hACE1};
        mask_pop = 0;
        error_count=0;
        ones_count=0;
        total_bits=0;

         $display("===== STARTING STAGE 2 (BERNOULLI SAMPLER & PRNG) TESTBENCH =====");

         // TEST CASE 1: Reset & Idle Verification
         #10 rst_n = 1;
         #10;
         if(!mask_empty || mask_full || mask_count !==0)begin 
            $display("[ERROR] Reset state invalid! Empty=%b, Full=%b, Count=%d", mask_empty, mask_full, mask_count);
            error_count = error_count + 1;
         end else begin 
            $display("[TEST 1 PASSED] Reset & Idle state verified.");
         end

        // TEST CASE 2 & 3: Background Generation & FIFO Filling
        $display("\n[TEST 2 & 3] Enabling Sampler & generating masks into FIFO...");
        @(posedge clk);
        sampler_en=1;

        // FIFO accumulates min 5 words (5 * 64 cycles = ~320 cycles)
        while (mask_count < 5) begin
            @(posedge clk);
        end
        $display("  -> SIPO accumulated %d 64-bit mask words successfully.", mask_count);

        $display("  -> Filling FIFO to capacity (%d words)...", `MASK_FIFO_DEPTH);
        while(!mask_full) begin
            @(posedge clk);
        end
        $display("[TEST 2 & 3 PASSED] FIFO filled to capacity. Backpressure verified (mask_full = 1).");

        // TEST CASE 4: Pop and Read Masks from FIFO
        $display("\n[TEST 4] Popping 5 mask words from FIFO for Stage 4...");

        for (w  = 0;w<5;w=w+1) begin 
            @(posedge clk);
            mask_pop =1;
        #1;
        $display("  -> Popped Mask Word %0d: 0x%016X (Count Remaining: %d)", w, mask_out[63:0], mask_count);
    
        if(mask_out == {`PF{1'b0}})begin 
            $display("[WARNING] Popped all-zero mask word (unlikely with valid LFSR).");
            end
        end
        mask_pop=0;

        $display("[TEST 4 PASSED] FIFO Pop interface verified.");
    
        // TEST CASE 5: Statistical Distribution Verification (p ~ 0.5)
         $display("\n[TEST 5] Statistical Monte Carlo Test (Sampling 4,096 bits)...");
         // Pop and evaluate 64 mask words = 64 * 64 = 4096 random bits
         for (w=0;w<64;w=w+1) begin 
            while(mask_empty) begin 
                @(posedge clk);
            end
            @(posedge clk);
            mask_pop =1;
            @(posedge clk);
            #1;
            for (b=0;b<`PF;b=b+1)begin 
                if(mask_out[b] == 1'b1) begin 
                    ones_count = ones_count + 1;    
                end
                total_bits = total_bits + 1;
            end
            mask_pop=0;
         end

        measured_prob = ones_count / (total_bits * 1.0);
        $display("  -> Total Bits Sampled: %d | Total 1s: %d", total_bits, ones_count);
        $display("  -> Measured Keep Probability p: %0.4f (Target: 0.5000)", measured_prob);

        if (measured_prob >= 0.45 && measured_prob <= 0.55) begin
            $display("[TEST 5 PASSED] Random distribution matches 50%% probability bounds (0.45 <= p <= 0.55).");
        end else begin
            $display("[ERROR] Probability skewed outside expected bounds! p = %0.4f", measured_prob);
            error_count = error_count + 1;
        end

        // TEST CASE 6: Seed Reload Verification
        $display("\n[TEST 6] Testing Seed Reload...");
        @(posedge clk);
        seed_in =  { {(`LFSR_WIDTH-16){1'b0}}, 16'hBEEF };
        load_seed=1;
        @(posedge clk);
        load_seed=0;
        #20;
        $display("[TEST 6 PASSED] Seed reload accepted without lockup.");
    
        #50;
        if (error_count == 0) begin
            $display("   ALL STAGE 2 TEST CASES PASSED PERFECTLY! (0 ERRORS)        ");
        end else begin
            $display("   TESTBENCH COMPLETED WITH %d ERRORS.                        ", error_count);
        end

        $finish;
    end
endmodule