//==============================================================================
// Module: tb_cache_reduction_engine.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 5 - Intermediate Caching & Output Reduction
//------------------------------------------------------------------------------
// Purpose:
//   Self-checking testbench for cache_reduction_engine.
//
// Architectural Inputs:
//   - None (stimulus generated internally; Stage 2 mask FIFO and Stage 1/4
//     handshakes are modelled by the testbench).
//
// Architectural Outputs:
//   - Console pass/fail report and sim/stage5/stage5_simulation.vcd.
//
// Description:
//   Test 1: Reset & idle state.
//   Test 2: IC caching of layer N-B in sample 1 (N=3, B=1).
//   Test 3: IC replay in sample 2 (raw cache data + fresh MCD mask).
//   Test 4: Output reduction over S=4 samples (10, 20, 30, 40 -> 25, 125).
//   Test 5: Full orchestration of a second inference (N=4, B=2, S=3).
//   All stimulus changes on the falling edge to avoid races with the DUT.
//==============================================================================

`timescale 1ns / 1ps
`include "bcnn_pkg.vh"

module tb_cache_reduction_engine;

    localparam DW = `DATA_WIDTH;
    localparam NCH = `PF * `PV;
    localparam BUS_W = `PF * `PV * `DATA_WIDTH;
    localparam VW = 2 * `DATA_WIDTH;

    // UAMH configuration sized to the DUT ports
    localparam [`SAMPLE_CNT_WIDTH-1:0] EE_MIN_SAMPLES = `MIN_SAMPLES_EXIT;
    localparam [$clog2(`PF*`PV+1)-1:0] UTAG_HIGH_COUNT = `UTAG_HIGH_COUNT_TH;

    reg clk, rst_n;
    reg start_inference, layer_done;
    reg [`LAYER_CNT_WIDTH-1:0] total_layers_N, bayesian_layers_B;
    reg [`SAMPLE_CNT_WIDTH-1:0] total_samples_S;
    reg [BUS_W-1:0] premask_features_in, stage4_features_in;
    reg premask_valid_in, stage4_valid_in;
    reg ic_rd_req, replay_mask_load, mask_valid;
    reg [`PF-1:0] mask_in;

    wire replay_mask_pop;
    wire [BUS_W-1:0] replay_features;
    wire replay_valid;
    wire [`IC_ADDR_WIDTH+1:0] ic_word_count;
    wire ic_full;
    wire [`SAMPLE_CNT_WIDTH-1:0] sample_idx;
    wire [`LAYER_CNT_WIDTH-1:0] layer_idx;
    wire busy, ic_write_en, ic_read_en, bypass_feature_extractor, mcd_en, is_final_layer, inference_done;
    wire [BUS_W-1:0] mean_prediction;
    wire [(NCH*VW)-1:0] uncertainty_score;
    wire reduction_done;

    cache_reduction_engine dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_inference(start_inference),
        .layer_done(layer_done),
        .total_layers_N(total_layers_N),
        .bayesian_layers_B(bayesian_layers_B),
        .total_samples_S(total_samples_S),
        .premask_features_in(premask_features_in),
        .premask_valid_in(premask_valid_in),
        .stage4_features_in(stage4_features_in),
        .stage4_valid_in(stage4_valid_in),
        .ic_rd_req(ic_rd_req),
        .replay_mask_load(replay_mask_load),
        .mask_in(mask_in),
        .mask_valid(mask_valid),
        .replay_mask_pop(replay_mask_pop),
        .replay_features(replay_features),
        .replay_valid(replay_valid),
        .ic_word_count(ic_word_count),
        .ic_byte_count(),
        .ic_full(ic_full),
        .umps_en(1'b0),
        .umps_thresh(`UMPS_DEFAULT_THRESH),
        .utag_en(1'b0),
        .utag_zero_thresh(`UTAG_ZERO_THRESH),
        .utag_high_count_th(UTAG_HIGH_COUNT),
        .spill_features(),
        .spill_valid(),
        .spill_req(),
        .spill_ret_features({(NCH*DW){1'b0}}),
        .spill_ret_valid(1'b0),
        .high_u_cached_count(),
        .low_u_bypassed_count(),
        .current_line_u_tag(),
        .early_exit_en(1'b0),
        .early_exit_thresh(`DEFAULT_EXIT_THRESH),
        .early_exit_min_samples(EE_MIN_SAMPLES),
        .eval_pending(),
        .early_exit_triggered(),
        .samples_executed(),
        .sample_idx(sample_idx),
        .layer_idx(layer_idx),
        .busy(busy),
        .ic_write_en(ic_write_en),
        .ic_read_en(ic_read_en),
        .bypass_feature_extractor(bypass_feature_extractor),
        .mcd_en(mcd_en),
        .is_final_layer(is_final_layer),
        .inference_done(inference_done),
        .mean_prediction(mean_prediction),
        .uncertainty_score(uncertainty_score),
        .reduction_done(reduction_done)
    );

    always #2.27 clk = ~clk;

    integer error_count, test_errors;
    integer f, w, s, wait_cnt;
    integer infer_id;              // selects the cached feature pattern of the run
    integer raw_idx, rp_idx;       // replay checker counters
    integer replay_pops, done_pulses;
    integer layers_run;
    integer exp_layer [0:7];
    integer exp_sample [0:7];
    integer got_m, got_v;
    reg [`PF-1:0] replay_mask;

    // Cached word pattern for run k, word wi, channel ci (always within INT8)
    function signed [DW-1:0] cache_val(input integer k, input integer wi, input integer ci);
        begin
            cache_val = ((k * 31 + wi * 17 + ci * 5) % 250) - 125;
        end
    endfunction

    // Final-layer output of run k, MC sample sm, channel ci
    function signed [DW-1:0] final_val(input integer k, input integer sm, input integer ci);
        begin
            if(ci == 0) final_val = sm * 10;
            else if(ci == 1) final_val = -(sm * 10);
            else if(ci == 2) final_val = 7;
            else final_val = ((k * 13 + sm * 29 + ci * 11) % 240) - 120;
        end
    endfunction

    // Filter-wise Bernoulli mask standing in for the Stage 2 FIFO head
    function [`PF-1:0] gen_mask(input integer seed);
        integer i;
        begin
            for(i=0; i<`PF; i=i+1) gen_mask[i] = (((i * 7) + (seed * 13)) % 5) > 1;
        end
    endfunction

    // Replay checker: raw BRAM data must match what was cached, replayed data must be masked
    always @(posedge clk) begin
        if(dut.ic_rd_valid) begin
            for(f=0; f<NCH; f=f+1) begin
                if($signed(dut.ic_rd_data[f*DW +: DW]) !== cache_val(infer_id, raw_idx, f)) begin
                    $display("[ERROR] IC read word %0d ch %0d: got %0d, expected %0d", raw_idx, f,
                             $signed(dut.ic_rd_data[f*DW +: DW]), cache_val(infer_id, raw_idx, f));
                    error_count = error_count + 1;
                end
            end
            raw_idx = raw_idx + 1;
        end
        if(replay_valid) begin
            for(f=0; f<NCH; f=f+1) begin
                if($signed(replay_features[f*DW +: DW]) !== (replay_mask[f % `PF] ? cache_val(infer_id, rp_idx, f) : 0)) begin
                    $display("[ERROR] Replay word %0d ch %0d (mask=%b): got %0d", rp_idx, f, replay_mask[f % `PF],
                             $signed(replay_features[f*DW +: DW]));
                    error_count = error_count + 1;
                end
            end
            rp_idx = rp_idx + 1;
        end
        if(replay_mask_pop) replay_pops = replay_pops + 1;
        if(inference_done) done_pulses = done_pulses + 1;
    end

    // Watchdog Timeout
    initial begin
        #100000;
        $display("\n[ERROR] Simulation timed out!");
        $finish;
    end

    task start_run(input integer n, input integer b, input integer sm);
        begin
            total_layers_N = n;
            bayesian_layers_B = b;
            total_samples_S = sm;
            start_inference = 1'b1;
            @(negedge clk);
            start_inference = 1'b0;
        end
    endtask

    task pulse_layer_done;
        begin
            layer_done = 1'b1;
            @(negedge clk);
            layer_done = 1'b0;
            @(negedge clk);
        end
    endtask

    // Stage 4 pre-dropout tap streaming the layer N-B output map
    task stream_cache(input integer k, input integer nwords);
        integer wi, ci;
        begin
            for(wi=0; wi<nwords; wi=wi+1) begin
                for(ci=0; ci<NCH; ci=ci+1) premask_features_in[ci*DW +: DW] = cache_val(k, wi, ci);
                premask_valid_in = 1'b1;
                @(negedge clk);
            end
            premask_valid_in = 1'b0;
        end
    endtask

    // Stage 1 ingress pulling the cache for one MC sample (one extra request must be ignored)
    task replay_cache(input [`PF-1:0] mask, input integer nwords);
        integer ri;
        begin
            raw_idx = 0;
            rp_idx = 0;
            replay_pops = 0;
            mask_in = mask;
            replay_mask = mask;
            replay_mask_load = 1'b1;
            @(negedge clk);
            replay_mask_load = 1'b0;
            for(ri=0; ri<=nwords; ri=ri+1) begin
                ic_rd_req = 1'b1;
                @(negedge clk);
            end
            ic_rd_req = 1'b0;
            repeat(4) @(negedge clk);
            if(raw_idx !== nwords || rp_idx !== nwords || replay_pops !== 1) begin
                $display("[ERROR] Replay counts: raw=%0d replayed=%0d pops=%0d (expected %0d, %0d, 1)",
                         raw_idx, rp_idx, replay_pops, nwords, nwords);
                test_errors = test_errors + 1;
            end
        end
    endtask

    // Stage 4 final-layer output of one MC sample
    task feed_final(input integer k, input integer sm);
        integer ci;
        begin
            for(ci=0; ci<NCH; ci=ci+1) stage4_features_in[ci*DW +: DW] = final_val(k, sm, ci);
            stage4_valid_in = 1'b1;
            @(negedge clk);
            stage4_valid_in = 1'b0;
        end
    endtask

    task wait_reduction;
        begin
            wait_cnt = 0;
            while(!reduction_done && wait_cnt < 200) begin
                @(negedge clk);
                wait_cnt = wait_cnt + 1;
            end
            if(!reduction_done) begin
                $display("[ERROR] reduction_done never asserted");
                test_errors = test_errors + 1;
            end
        end
    endtask

    // Golden model: Mean = trunc(Sum/S), Var = Sum(x^2)/S - Mean^2
    task check_reduction(input integer k, input integer n_samples);
        integer ci, sm, v, sum, sum_sq, exp_mean, exp_var;
        begin
            for(ci=0; ci<NCH; ci=ci+1) begin
                sum = 0;
                sum_sq = 0;
                for(sm=1; sm<=n_samples; sm=sm+1) begin
                    v = final_val(k, sm, ci);
                    sum = sum + v;
                    sum_sq = sum_sq + v * v;
                end
                exp_mean = sum / n_samples;
                exp_var = (sum_sq / n_samples) - exp_mean * exp_mean;
                got_m = $signed(mean_prediction[ci*DW +: DW]);
                got_v = uncertainty_score[ci*VW +: VW];
                if(got_m !== exp_mean || got_v !== exp_var) begin
                    $display("[ERROR] Ch %0d: Mean=%0d Var=%0d (Expected Mean=%0d Var=%0d)", ci, got_m, got_v, exp_mean, exp_var);
                    test_errors = test_errors + 1;
                end
            end
        end
    endtask

    initial begin
        $dumpfile("sim/stage5/stage5_simulation.vcd");
        $dumpvars(0, tb_cache_reduction_engine);

        clk = 0;
        rst_n = 0;
        start_inference = 0;
        layer_done = 0;
        total_layers_N = 0;
        bayesian_layers_B = 0;
        total_samples_S = 0;
        premask_features_in = {BUS_W{1'b0}};
        stage4_features_in = {BUS_W{1'b0}};
        premask_valid_in = 0;
        stage4_valid_in = 0;
        ic_rd_req = 0;
        replay_mask_load = 0;
        mask_valid = 1'b1;
        mask_in = {`PF{1'b1}};
        replay_mask = {`PF{1'b1}};
        error_count = 0;
        raw_idx = 0;
        rp_idx = 0;
        replay_pops = 0;
        done_pulses = 0;

        // Expected layer trace for Test 5 (N=4, B=2, S=3): 1 2 3 4 | 3 4 | 3 4
        exp_layer[0] = 1; exp_layer[1] = 2; exp_layer[2] = 3; exp_layer[3] = 4;
        exp_layer[4] = 3; exp_layer[5] = 4; exp_layer[6] = 3; exp_layer[7] = 4;
        exp_sample[0] = 1; exp_sample[1] = 1; exp_sample[2] = 1; exp_sample[3] = 1;
        exp_sample[4] = 2; exp_sample[5] = 2; exp_sample[6] = 3; exp_sample[7] = 3;

        $display("   STARTING STAGE 5 (CACHE & OUTPUT REDUCTION) TESTBENCH");

        repeat(3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // TEST CASE 1: Reset & Idle
        test_errors = 0;
        if(sample_idx !== 0 || layer_idx !== 0 || busy !== 0 || ic_write_en !== 0 || ic_read_en !== 0 ||
           bypass_feature_extractor !== 0 || mcd_en !== 0 || is_final_layer !== 0 || inference_done !== 0 ||
           reduction_done !== 0 || replay_valid !== 0 || replay_mask_pop !== 0 || ic_word_count !== 0 || ic_full !== 0) begin
            $display("[ERROR] Reset state invalid! sample=%0d layer=%0d busy=%b wr=%b rd=%b byp=%b mcd=%b final=%b done=%b red=%b",
                     sample_idx, layer_idx, busy, ic_write_en, ic_read_en, bypass_feature_extractor, mcd_en,
                     is_final_layer, inference_done, reduction_done);
            test_errors = test_errors + 1;
        end
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 1 PASSED] Reset & Idle state verified.");

        // TEST CASE 2: IC caching of layer N-B during sample 1
        $display("\n[TEST 2] Intermediate-layer Caching (N=3, B=1, S=4 -> cache layer 2, sample 1)...");
        test_errors = 0;
        infer_id = 1;
        start_run(3, 1, 4);

        if(busy !== 1 || sample_idx !== 1 || layer_idx !== 1 || ic_write_en !== 0 || bypass_feature_extractor !== 0 || mcd_en !== 0) begin
            $display("[ERROR] Layer 1 state: busy=%b sample=%0d layer=%0d wr=%b byp=%b mcd=%b",
                     busy, sample_idx, layer_idx, ic_write_en, bypass_feature_extractor, mcd_en);
            test_errors = test_errors + 1;
        end

        // Layer 1 output must not reach the cache
        stream_cache(infer_id, 2);
        if(ic_word_count !== 0) begin
            $display("[ERROR] Layer 1 output was cached! ic_word_count=%0d", ic_word_count);
            test_errors = test_errors + 1;
        end

        pulse_layer_done;
        if(layer_idx !== 2 || ic_write_en !== 1 || mcd_en !== 1 || is_final_layer !== 0) begin
            $display("[ERROR] Layer 2 state: layer=%0d wr=%b mcd=%b final=%b", layer_idx, ic_write_en, mcd_en, is_final_layer);
            test_errors = test_errors + 1;
        end

        stream_cache(infer_id, 4);
        $display("  -> ic_write_en=%b, words cached=%0d (Expected: 4)", ic_write_en, ic_word_count);
        if(ic_word_count !== 4) begin
            $display("[ERROR] ic_word_count=%0d, expected 4", ic_word_count);
            test_errors = test_errors + 1;
        end
        for(w=0; w<4; w=w+1) begin
            for(f=0; f<NCH; f=f+1) begin
                if($signed(dut.u_ic_buffer.mem[w][f*DW +: DW]) !== cache_val(infer_id, w, f)) begin
                    $display("[ERROR] IC mem[%0d] ch %0d = %0d, expected %0d", w, f,
                             $signed(dut.u_ic_buffer.mem[w][f*DW +: DW]), cache_val(infer_id, w, f));
                    test_errors = test_errors + 1;
                end
            end
        end
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 2 PASSED] Layer N-B output cached in IC buffer during sample 1.");

        // TEST CASE 3: IC replay in sample 2
        $display("\n[TEST 3] Intermediate-layer Playback (Sample 2)...");
        test_errors = 0;

        // Layer 3 (final) of sample 1 runs normally
        pulse_layer_done;
        if(is_final_layer !== 1 || bypass_feature_extractor !== 0 || ic_read_en !== 0 || ic_write_en !== 0 || mcd_en !== 0) begin
            $display("[ERROR] Sample 1 layer 3 state: final=%b byp=%b rd=%b wr=%b mcd=%b",
                     is_final_layer, bypass_feature_extractor, ic_read_en, ic_write_en, mcd_en);
            test_errors = test_errors + 1;
        end
        feed_final(infer_id, 1);
        pulse_layer_done;

        $display("  -> sample_idx=%0d layer_idx=%0d bypass_feature_extractor=%b ic_read_en=%b",
                 sample_idx, layer_idx, bypass_feature_extractor, ic_read_en);
        if(sample_idx !== 2 || layer_idx !== 3 || bypass_feature_extractor !== 1 || ic_read_en !== 1 || ic_write_en !== 0) begin
            $display("[ERROR] Sample 2 did not skip to layer N-B+1 with bypass active");
            test_errors = test_errors + 1;
        end

        replay_cache(gen_mask(2), 4);
        $display("  -> Replayed %0d words, mask 0x%h applied, %0d mask pop", rp_idx, replay_mask, replay_pops);
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 3 PASSED] Cached data read back intact and re-masked for sample 2.");

        // TEST CASE 4: Output reduction over S=4 samples
        $display("\n[TEST 4] Output Reduction (Ch0 samples: 10, 20, 30, 40)...");
        test_errors = 0;
        feed_final(infer_id, 2);
        pulse_layer_done;

        for(s=3; s<=4; s=s+1) begin
            if(sample_idx !== s || layer_idx !== 3 || ic_read_en !== 1) begin
                $display("[ERROR] Sample %0d state: sample=%0d layer=%0d rd=%b", s, sample_idx, layer_idx, ic_read_en);
                test_errors = test_errors + 1;
            end
            replay_cache(gen_mask(s), 4);
            if(reduction_done !== 0) begin
                $display("[ERROR] reduction_done asserted before sample S");
                test_errors = test_errors + 1;
            end
            feed_final(infer_id, s);
            pulse_layer_done;
        end

        wait_reduction;
        $display("  -> Ch0 Mean: %0d (Expected: 25) | Ch0 Variance: %0d (Expected: 125)",
                 $signed(mean_prediction[0 +: DW]), uncertainty_score[0 +: VW]);
        $display("  -> Ch1 Mean: %0d (Expected: -25) | Ch1 Variance: %0d (Expected: 125)",
                 $signed(mean_prediction[DW +: DW]), uncertainty_score[VW +: VW]);
        if($signed(mean_prediction[0 +: DW]) !== 25 || uncertainty_score[0 +: VW] !== 125 ||
           $signed(mean_prediction[DW +: DW]) !== -25 || uncertainty_score[VW +: VW] !== 125 ||
           $signed(mean_prediction[2*DW +: DW]) !== 7 || uncertainty_score[2*VW +: VW] !== 0) begin
            $display("[ERROR] Reference channels mismatch");
            test_errors = test_errors + 1;
        end
        check_reduction(infer_id, 4);

        if(busy !== 0 || done_pulses !== 1) begin
            $display("[ERROR] Inference end: busy=%b inference_done pulses=%0d", busy, done_pulses);
            test_errors = test_errors + 1;
        end
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 4 PASSED] Mean & variance verified bit-for-bit on all %0d channels.", NCH);

        // TEST CASE 5: Full orchestration of a second inference
        $display("\n[TEST 5] Full Stage 5 Orchestration (N=4, B=2, S=3)...");
        test_errors = 0;
        infer_id = 2;
        start_run(4, 2, 3);

        if(reduction_done !== 0 || ic_word_count !== 0) begin
            $display("[ERROR] New run did not clear state: reduction_done=%b ic_word_count=%0d", reduction_done, ic_word_count);
            test_errors = test_errors + 1;
        end

        layers_run = 0;
        while(busy && layers_run < 8) begin
            if(layer_idx !== exp_layer[layers_run] || sample_idx !== exp_sample[layers_run]) begin
                $display("[ERROR] Step %0d: sample=%0d layer=%0d, expected sample=%0d layer=%0d",
                         layers_run, sample_idx, layer_idx, exp_sample[layers_run], exp_layer[layers_run]);
                test_errors = test_errors + 1;
            end
            if(mcd_en !== (layer_idx >= 2 && layer_idx < 4)) begin
                $display("[ERROR] mcd_en=%b at layer %0d", mcd_en, layer_idx);
                test_errors = test_errors + 1;
            end
            $display("  -> Sample %0d Layer %0d : ic_write_en=%b ic_read_en=%b bypass=%b mcd_en=%b final=%b",
                     sample_idx, layer_idx, ic_write_en, ic_read_en, bypass_feature_extractor, mcd_en, is_final_layer);

            if(ic_write_en) stream_cache(infer_id, 6);
            if(ic_read_en) replay_cache(gen_mask(sample_idx + 10), 6);
            if(is_final_layer) feed_final(infer_id, sample_idx);
            layers_run = layers_run + 1;
            pulse_layer_done;
        end

        wait_reduction;
        check_reduction(infer_id, 3);
        $display("  -> Ch0 Mean: %0d (Expected: 20) | Ch0 Variance: %0d (Expected: 66)",
                 $signed(mean_prediction[0 +: DW]), uncertainty_score[0 +: VW]);
        $display("  -> Layers executed: %0d (naive N*S = %0d)", layers_run, 4 * 3);
        if(layers_run !== 8 || busy !== 0 || done_pulses !== 2 || ic_word_count !== 6) begin
            $display("[ERROR] Orchestration: layers=%0d busy=%b done pulses=%0d cached=%0d", layers_run, busy, done_pulses, ic_word_count);
            test_errors = test_errors + 1;
        end
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 5 PASSED] Caching, replay and reduction hand over seamlessly.");

        // Final Summary
        #50;
        if(error_count == 0) begin
            $display("   ALL STAGE 5 TEST CASES PASSED PERFECTLY! (0 ERRORS)        ");
        end else begin
            $display("   TESTBENCH COMPLETED WITH %0d ERRORS.                        ", error_count);
        end

        $finish;
    end

endmodule
