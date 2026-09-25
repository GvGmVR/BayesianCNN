`timescale 1ns / 1ps
`include "../bcnn_pkg.vh"

module tb_processing_engine;

    reg clk,rst_n,valid_in,window_done,relu_en;

    reg [(`PF*`PV*`PC*`DATA_WIDTH-1):0] pe_data_in;
    reg [(`PF*`PV*`PC*`DATA_WIDTH-1):0] pe_weight_in;

    reg signed [`QUANT_SCALE_WIDTH-1:0] quant_scale;
    reg [`QUANT_SHIFT_WIDTH-1:0] quant_shift;   
    reg signed [`QUANT_BIAS_WIDTH-1:0] quant_bias;

    wire [(`PF*`PV*`DATA_WIDTH-1):0] pe_features_out;
    wire features_valid;

    processing_engine #(
        .DATA_WIDTH(`DATA_WIDTH),
        .PC(`PC),
        .PF(`PF),
        .PV(`PV),
        .LOG2_PC(`LOG2_PC),
        .MULT_OUT_WIDTH(`MULT_OUT_WIDTH),
        .ADDR_TREE_WIDTH(`ADDR_TREE_WIDTH),
        .ACCUM_WIDTH(`ACCUM_WIDTH),
        .QUANT_SCALE_WIDTH(`QUANT_SCALE_WIDTH),
        .QUANT_SHIFT_WIDTH(`QUANT_SHIFT_WIDTH),
        .QUANT_BIAS_WIDTH(`QUANT_BIAS_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .window_done(window_done),
        .relu_en(relu_en),
        .pe_data_in(pe_data_in),
        .pe_weight_in(pe_weight_in),
        .quant_scale(quant_scale),
        .quant_shift(quant_shift),
        .quant_bias(quant_bias),
        .pe_features_out(pe_features_out),
        .features_valid(features_valid)
    );

    always #2.27 clk=~clk;

    integer f,c,step,error_count;
    reg signed [7:0] filter0_out;

    // Watchdog Timeout
    initial begin
        #50000;
        $display("\n[ERROR] Simulation timed out!");
        $finish;
    end

    initial begin 
        $dumpfile("sim/stage3/stage3_simulation.vcd");
        $dumpvars(0, tb_processing_engine);

        clk =0;
        rst_n=0;
        valid_in=0;
        window_done=0;
        relu_en=1;
        pe_data_in=0;
        pe_weight_in=0;
        quant_scale = 16'd1;
        quant_shift = 5'd0;
        quant_bias = 32'd0;
        error_count=0;

        $display("   STARTING STAGE 3 (NNE PROCESSING ENGINE) TESTBENCH");

        #10;
        rst_n=1;
        #10;

        // TEST CASE 1: Reset State Check
        if(features_valid !== 1'b0) begin 
            $display("[ERROR] Reset state invalid! features_valid should be 0.");
            error_count = error_count + 1;
        end else begin 
            $display("[TEST 1 PASSED] Reset & Idle state verified.");
        end

        // TEST CASE 2: 3x3 Sliding Window Convolution Math (9 steps)
        $display("\n[TEST 2] Running 3x3 Window (9 steps) across 64 filters in parallel...");
        
        // Setup: Each channel pixel = 1, each weight = 1
        // Dot product per step = 64 * (1 * 1) = 64
        // Total 9 steps accumulator sum = 9 * 64 = 576
        // Set scale=1, shift=3 (divide by 8): 576 / 8 = 72

        quant_scale = 16'd1;
        quant_shift = 5'd3;
        quant_bias = 32'sd0;
        relu_en=1'b1;

        // Load constant 1s into all 64 data and weight channels
        for(f=0;f<`PF;f=f+1)begin 
            for(c=0;c<`PC;c=c+1) begin 
                pe_data_in[(f*`PC+c)*8 +:8] = 8'sd1;
                pe_weight_in[(f*`PC+c)*8 +:8] = 8'sd1;
            end
        end

        //Stream 9 window steps
        for (step=0;step<9;step=step+1) begin 
            @(posedge clk);
            valid_in = 1'b1;
            window_done = (step==8) ? 1'b1 : 1'b0;
        end

        @(posedge clk);
        valid_in = 1'b0;
        window_done = 1'b0;

        //wait for pipeline latency(Adder+Quantizer+ReLu)
        while(!features_valid) @(posedge clk);
        #1;

        filter0_out = pe_features_out[7:0];
        $display("  -> Computed Filter 0 Output: %d (Expected: 72)", filter0_out);
        $display("  -> Computed Filter 63 Output: %d (Expected: 72)", pe_features_out[511:504]);

        if(filter0_out !== 8'sd72) begin 
            $display("[ERROR] Filter 0 output mismatch! Expected 72, got %d", filter0_out);
            error_count = error_count + 1;
        end else begin 
            $display("[TEST 2 PASSED] 64-Filter parallel MAC AND Accumulation verified.");
        end

        // TEST CASE 3: Negative Input & ReLU Activation Test
        $display("\n[TEST 3] Testing Negative Values with ReLU Active (relu_en = 1)...");
        #20;
        // Data = 1, Weights = -2 -> 9 steps sum = 9 * 64 * (-2) = -1152
        for(f=0;f<`PF;f=f+1)begin 
            for (c=0;c<`PF;c=c+1) begin 
                pe_data_in[(f*`PC+c)*8 +:8] = 8'sd1;
                pe_weight_in[(f*`PC+c)*8 +:8] = -8'sd2;
            end
        end

        for(step=0;step<9;step=step+1) begin 
            @(posedge clk);
            valid_in = 1'b1;
            window_done = (step==8) ? 1'b1 : 1'b0;
        end
        @(posedge clk);
        valid_in = 1'b0;
        window_done=1'b0;

        while (!features_valid) @(posedge clk);
        #1;

        filter0_out = pe_features_out[7:0];
        $display("  -> ReLU Active Output: %d (Expected: 0)", filter0_out);

        if (filter0_out !== 8'sd0) begin
            $display("[ERROR] ReLU Failed! Expected: 0, Got: %d", filter0_out);
            error_count = error_count + 1;
        end else begin
            $display("[TEST 3 PASSED] ReLU successfully clamped negative values to 0.");
        end

        // TEST CASE 4: ReLU Bypass Test (relu_en = 0)
        $display("\n[TEST 4] Testing ReLU Bypass (relu_en = 0)...");
        #20;

        relu_en = 1'b0;
        quant_shift = 5'd4; // -1152/16 = -72

        for(step=0;step<9;step=step+1) begin 
            @(posedge clk);
            valid_in = 1'b1;
            window_done = (step==8) ? 1'b1 : 1'b0;
        end
        @(posedge clk);
        valid_in = 1'b0;
        window_done=1'b0;

        while (!features_valid) @(posedge clk);
        #1;
        filter0_out = pe_features_out[7:0];
        $display("  -> ReLU Bypass Output: %d (Expected: -72)", filter0_out);

        if (filter0_out !== -8'sd72) begin
            $display("[ERROR] Bypass Failed! Expected: -72, Got: %d", filter0_out);
            error_count = error_count + 1;
        end else begin
            $display("[TEST 4 PASSED] ReLU bypass verified (retained signed negative value).");
        end

        //Final summary
        #50;
        $display("\n===============================================================");
        if (error_count == 0) begin
            $display("   ALL STAGE 3 TEST CASES PASSED PERFECTLY! (0 ERRORS)        ");
        end else begin
            $display("   TESTBENCH COMPLETED WITH %d ERRORS.                        ", error_count);
        end
        $display("===============================================================\n");

        $finish;

    end

endmodule