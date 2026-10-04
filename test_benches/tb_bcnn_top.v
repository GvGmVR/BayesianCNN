//==============================================================================
// Module: tb_bcnn_top.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: Top - End-to-End System Testbench
//------------------------------------------------------------------------------
// Purpose:
//   Self-checking end-to-end verification of bcnn_top running a 3-layer
//   partially Bayesian CNN (N=3, B=1) for S=3 Monte Carlo samples.
//
// Architectural Inputs:
//   - None (the testbench models the host and the off-chip DRAM).
//
// Architectural Outputs:
//   - Console pass/fail report and sim/bcnn_top_simulation.vcd.
//
// Description:
//   Network: three 1x1 convolutions on a 2x2x64 map, PF=64 filters each.
//     Layer 1: Conv + ReLU (deterministic).
//     Layer 2: Conv + ReLU (deterministic, N-B: cached in IC, MCD on output).
//     Layer 3: Conv + 2x2 avg pool -> one 64-wide logit vector per sample.
//   The host streams each layer's input from "DRAM", which is the egress of
//   the previous layer, and pushes the layer weights. A bit-exact golden model
//   (conv -> quant -> ReLU -> pool -> mask) checks every layer, using the masks
//   actually popped from Stage 2, and the final mean / variance.
//   Test 1: Reset & idle.
//   Test 2: Layer 1 from DRAM.
//   Test 3: Layer 2, IC caching and sample-1 MCD mask.
//   Test 4: Samples 2..3 replayed from the IC buffer (layers 1..2 skipped).
//   Test 5: Output reduction, layer count and DRAM traffic.
//==============================================================================

`timescale 1ns / 1ps
`include "bcnn_pkg.vh"

module tb_bcnn_top;

    localparam DW = `DATA_WIDTH;
    localparam NCH = `PF * `PV;
    localparam VW = 2 * `DATA_WIDTH;
    localparam INT_MAX = (1 << (DW-1)) - 1;
    localparam INT_MIN = -(1 << (DW-1));
    localparam POOL_WIN = 1 << `POOL_CNT_WIDTH;

    // UAMH configuration sized to the DUT ports
    localparam [`SAMPLE_CNT_WIDTH-1:0] EE_MIN_SAMPLES = `MIN_SAMPLES_EXIT;
    localparam [$clog2(`PF*`PV+1)-1:0] UTAG_HIGH_COUNT = `UTAG_HIGH_COUNT_TH;

    // Network under test
    localparam N_RUN = 3;
    localparam B_RUN = 1;
    localparam S_RUN = 3;
    localparam IMG_H = 2;
    localparam IMG_W = 2;
    localparam NPIX = IMG_H * IMG_W;
    localparam STEPS = (N_RUN - B_RUN) + B_RUN * S_RUN;

    reg clk, rst_n;
    reg start_inference;
    reg [`LAYER_CNT_WIDTH-1:0] total_layers_N, bayesian_layers_B;
    reg [`SAMPLE_CNT_WIDTH-1:0] total_samples_S;
    reg [`DIM_WIDTH-1:0] H, W, L_frames;
    reg [`TILE_CNT_WIDTH-1:0] C_tiles, W_tiles;
    reg [`KERNEL_DIM_WIDTH-1:0] KH, KW, KL;
    reg [`STRIDE_WIDTH-1:0] stride;
    reg mode_3d, relu_en, sc_en;
    reg [`POOL_MODE_WIDTH-1:0] pool_mode;
    reg signed [`QUANT_SCALE_WIDTH-1:0] quant_scale;
    reg [`QUANT_SHIFT_WIDTH-1:0] quant_shift;
    reg signed [`QUANT_BIAS_WIDTH-1:0] quant_bias;
    reg dram_data_valid;
    reg [(`PC*DW)-1:0] dram_data_in;
    reg weight_push;
    reg [(`PC*`PF*DW)-1:0] weight_din;
    reg [(NCH*DW)-1:0] sc_features_in;
    reg load_seed;
    reg [`LFSR_WIDTH-1:0] seed_in;

    wire busy, inference_done, dram_data_ready, weight_full;
    wire [(NCH*DW)-1:0] mean_prediction;
    wire [(NCH*VW)-1:0] uncertainty_score;
    wire reduction_done;
    wire [(NCH*DW)-1:0] layer_features_out;
    wire layer_features_valid, layer_done;

    bcnn_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_inference(start_inference),
        .busy(busy),
        .inference_done(inference_done),
        .total_layers_N(total_layers_N),
        .bayesian_layers_B(bayesian_layers_B),
        .total_samples_S(total_samples_S),
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
        .relu_en(relu_en),
        .sc_en(sc_en),
        .pool_mode(pool_mode),
        .quant_scale(quant_scale),
        .quant_shift(quant_shift),
        .quant_bias(quant_bias),
        .dram_data_valid(dram_data_valid),
        .dram_data_ready(dram_data_ready),
        .dram_data_in(dram_data_in),
        .weight_push(weight_push),
        .weight_din(weight_din),
        .weight_full(weight_full),
        .sc_features_in(sc_features_in),
        .load_seed(load_seed),
        .seed_in(seed_in),
        .umps_en(1'b0),
        .umps_thresh(`UMPS_DEFAULT_THRESH),
        .utag_en(1'b0),
        .utag_zero_thresh(`UTAG_ZERO_THRESH),
        .utag_high_count_th(UTAG_HIGH_COUNT),
        .early_exit_en(1'b0),
        .early_exit_thresh(`DEFAULT_EXIT_THRESH),
        .early_exit_min_samples(EE_MIN_SAMPLES),
        .mean_prediction(mean_prediction),
        .uncertainty_score(uncertainty_score),
        .reduction_done(reduction_done),
        .layer_features_out(layer_features_out),
        .layer_features_valid(layer_features_valid),
        .layer_done(layer_done),
        .ic_lines_cached(),
        .ic_bytes_used(),
        .ic_spill_data(),
        .ic_spill_valid(),
        .high_u_cached_count(),
        .low_u_bypassed_count(),
        .current_line_u_tag(),
        .early_exit_triggered(),
        .samples_executed(),
        .eval_pending()
    );

    always #2.27 clk = ~clk;

    // Per-layer operator configuration (index = layer number)
    integer cfg_relu [1:N_RUN];
    integer cfg_pool [1:N_RUN];
    integer cfg_scale [1:N_RUN];
    integer cfg_shift [1:N_RUN];
    integer cfg_bias [1:N_RUN];

    // Expected execution trace with IC: 1 2 3 | 3 | 3
    integer exp_layer [0:STEPS-1];
    integer exp_sample [0:STEPS-1];

    // Host DRAM image of the current layer input, and captured egress
    integer cur_in [0:(NPIX*`PC)-1];
    integer egress [0:(NPIX*NCH)-1];
    integer feed_idx, feed_len, eg_cnt;

    // Golden model buffers
    integer gin [0:(NPIX*`PC)-1];
    integer gpre [0:(NPIX*NCH)-1];
    integer gout [0:(NPIX*NCH)-1];
    integer gold1 [0:(NPIX*NCH)-1];
    integer gold_u2 [0:(NPIX*NCH)-1];
    integer gold3 [0:(S_RUN*NCH)-1];
    reg [`PF-1:0] mask_s [1:S_RUN];

    integer error_count, test_errors;
    integer step, l, s, p, f, c, nw, hc, ec;
    integer dram_beats, beats_prev, layer_pulses, done_pulses;
    integer stage4_pops, replay_pops, ic_writes;
    integer sum, sum_sq, exp_mean, exp_var, got_m, got_v, wait_cnt;

    // Layer 1 input feature map, pixel p, channel c
    function integer inval(input integer pi, input integer ci);
        begin
            inval = ((pi * 7 + ci * 3) % 11) - 5;
        end
    endfunction

    // Weight of layer li, filter fi, channel ci
    function integer wval(input integer li, input integer fi, input integer ci);
        begin
            wval = ((li * 17 + fi * 31 + ci * 13 + fi * ci * 7) % 9) - 4;
        end
    endfunction

    // Linear quantizer + saturation + optional ReLU (Stage 3 golden)
    function integer quant(input integer li, input integer acc);
        integer v;
        begin
            v = (acc * cfg_scale[li] + (cfg_bias[li] <<< cfg_shift[li])) >>> cfg_shift[li];
            if(v > INT_MAX) v = INT_MAX;
            else if(v < INT_MIN) v = INT_MIN;
            if(cfg_relu[li] && v < 0) v = 0;
            quant = v;
        end
    endfunction

    // Golden layer: gin -> conv/quant/ReLU -> pool -> (mask) -> gout, nwords output pixels
    task golden_layer(input integer li, input integer mask_on, input [`PF-1:0] mask, output integer nwords);
        integer pi, fi, ci, acc, wi;
        begin
            for(pi=0; pi<NPIX; pi=pi+1) begin
                for(fi=0; fi<NCH; fi=fi+1) begin
                    acc = 0;
                    for(ci=0; ci<`PC; ci=ci+1) acc = acc + gin[pi*`PC+ci] * wval(li, fi, ci);
                    gpre[pi*NCH+fi] = quant(li, acc);
                end
            end
            if(cfg_pool[li] == `POOL_MODE_AVG) begin
                nwords = NPIX / POOL_WIN;
                for(wi=0; wi<nwords; wi=wi+1) begin
                    for(fi=0; fi<NCH; fi=fi+1) begin
                        acc = 0;
                        for(pi=0; pi<POOL_WIN; pi=pi+1) acc = acc + gpre[(wi*POOL_WIN+pi)*NCH+fi];
                        gout[wi*NCH+fi] = acc >>> `POOL_CNT_WIDTH;
                    end
                end
            end else begin
                nwords = NPIX;
                for(pi=0; pi<NPIX*NCH; pi=pi+1) gout[pi] = gpre[pi];
            end
            if(mask_on) begin
                for(wi=0; wi<nwords; wi=wi+1) begin
                    for(fi=0; fi<NCH; fi=fi+1) if(!mask[fi % `PF]) gout[wi*NCH+fi] = 0;
                end
            end
        end
    endtask

    task check_egress(input integer nwords);
        integer wi, fi;
        begin
            if(eg_cnt !== nwords) begin
                $display("[ERROR] Egress words: got %0d, expected %0d", eg_cnt, nwords);
                test_errors = test_errors + 1;
            end
            for(wi=0; wi<nwords; wi=wi+1) begin
                for(fi=0; fi<NCH; fi=fi+1) begin
                    if(egress[wi*NCH+fi] !== gout[wi*NCH+fi]) begin
                        $display("[ERROR] Egress word %0d ch %0d: got %0d, expected %0d", wi, fi, egress[wi*NCH+fi], gout[wi*NCH+fi]);
                        test_errors = test_errors + 1;
                    end
                end
            end
        end
    endtask

    task set_layer_config(input integer li);
        begin
            relu_en = cfg_relu[li];
            pool_mode = cfg_pool[li];
            quant_scale = cfg_scale[li];
            quant_shift = cfg_shift[li];
            quant_bias = cfg_bias[li];
        end
    endtask

    // One window of PF-filter weights (1x1 kernel, C_tiles=1 -> one word per layer)
    task push_weights(input integer li);
        integer fi, ci;
        begin
            for(fi=0; fi<`PF; fi=fi+1) begin
                for(ci=0; ci<`PC; ci=ci+1) weight_din[(fi*`PC+ci)*DW +: DW] = wval(li, fi, ci);
            end
            weight_push = 1'b1;
            @(negedge clk);
            weight_push = 1'b0;
        end
    endtask

    // Host DRAM: every beat driven while ready is accepted on the next rising edge
    always @(negedge clk) begin
        if(dram_data_valid) begin
            feed_idx = feed_idx + 1;
            dram_beats = dram_beats + 1;
        end
        if(dram_data_ready && feed_idx < feed_len) begin
            for(hc=0; hc<`PC; hc=hc+1) dram_data_in[hc*DW +: DW] = cur_in[feed_idx*`PC+hc];
            dram_data_valid = 1'b1;
        end else begin
            dram_data_valid = 1'b0;
        end
    end

    // Egress capture and system monitors
    always @(posedge clk) begin
        if(layer_features_valid) begin
            for(ec=0; ec<NCH; ec=ec+1) egress[eg_cnt*NCH+ec] = $signed(layer_features_out[ec*DW +: DW]);
            eg_cnt = eg_cnt + 1;
        end
        if(dut.stage4_mask_pop) begin
            mask_s[dut.s5_sample_idx] = dut.sampler_mask_out;
            stage4_pops = stage4_pops + 1;
        end
        if(dut.replay_mask_pop) begin
            mask_s[dut.s5_sample_idx] = dut.sampler_mask_out;
            replay_pops = replay_pops + 1;
        end
        if(dut.u_stage5.ic_wr) ic_writes = ic_writes + 1;
        if(dut.s5_ic_write_en && !(dut.s5_layer_idx == N_RUN-B_RUN && dut.s5_sample_idx == 1)) begin
            $display("[ERROR] ic_write_en high at sample %0d layer %0d", dut.s5_sample_idx, dut.s5_layer_idx);
            error_count = error_count + 1;
        end
        if(dut.s5_busy && dut.s5_bypass !== (dut.s5_sample_idx > 1)) begin
            $display("[ERROR] bypass_feature_extractor=%b at sample %0d", dut.s5_bypass, dut.s5_sample_idx);
            error_count = error_count + 1;
        end
        if(layer_done) layer_pulses = layer_pulses + 1;
        if(inference_done) done_pulses = done_pulses + 1;
    end

    // Watchdog Timeout
    initial begin
        #500000;
        $display("TIMEOUT");
        $finish;
    end

    initial begin
        $dumpfile("sim/bcnn_top_simulation.vcd");
        $dumpvars(0, tb_bcnn_top);

        clk = 0;
        rst_n = 0;
        start_inference = 0;
        total_layers_N = N_RUN;
        bayesian_layers_B = B_RUN;
        total_samples_S = S_RUN;
        H = IMG_H;
        W = IMG_W;
        L_frames = 1;
        C_tiles = 1;
        W_tiles = IMG_W / `PV;
        KH = 1;
        KW = 1;
        KL = 1;
        stride = 1;
        mode_3d = 0;
        sc_en = 0;
        dram_data_valid = 0;
        dram_data_in = {(`PC*DW){1'b0}};
        weight_push = 0;
        weight_din = {(`PC*`PF*DW){1'b0}};
        sc_features_in = {(NCH*DW){1'b0}};
        load_seed = 0;
        seed_in = {`LFSR_WIDTH{1'b0}};
        feed_idx = 0;
        feed_len = 0;
        eg_cnt = 0;
        dram_beats = 0;
        beats_prev = 0;
        layer_pulses = 0;
        done_pulses = 0;
        stage4_pops = 0;
        replay_pops = 0;
        ic_writes = 0;
        error_count = 0;

        // Layer 1: Conv+ReLU, Layer 2: Conv+ReLU, Layer 3: Conv + global 2x2 avg pool
        cfg_relu[1] = 1; cfg_pool[1] = `POOL_MODE_BYPASS; cfg_scale[1] = 1; cfg_shift[1] = 4; cfg_bias[1] = 2;
        cfg_relu[2] = 1; cfg_pool[2] = `POOL_MODE_BYPASS; cfg_scale[2] = 1; cfg_shift[2] = 3; cfg_bias[2] = 1;
        cfg_relu[3] = 0; cfg_pool[3] = `POOL_MODE_AVG; cfg_scale[3] = 1; cfg_shift[3] = 4; cfg_bias[3] = 3;

        exp_layer[0] = 1; exp_sample[0] = 1;
        exp_layer[1] = 2; exp_sample[1] = 1;
        exp_layer[2] = 3; exp_sample[2] = 1;
        exp_layer[3] = 3; exp_sample[3] = 2;
        exp_layer[4] = 3; exp_sample[4] = 3;

        $display("   STARTING BCNN_TOP END-TO-END SYSTEM TESTBENCH");
        $display("   Network: N=%0d, B=%0d, S=%0d | %0dx%0dx%0d input, 1x1 conv, PF=%0d", N_RUN, B_RUN, S_RUN, IMG_H, IMG_W, `PC, `PF);

        repeat(3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // TEST CASE 1: Reset & Idle
        test_errors = 0;
        if(busy !== 0 || inference_done !== 0 || reduction_done !== 0 || layer_done !== 0 ||
           layer_features_valid !== 0 || dram_data_ready !== 0 || weight_full !== 0) begin
            $display("[ERROR] Reset state invalid! busy=%b done=%b red=%b ldone=%b lvalid=%b ready=%b wfull=%b",
                     busy, inference_done, reduction_done, layer_done, layer_features_valid, dram_data_ready, weight_full);
            test_errors = test_errors + 1;
        end
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 1 PASSED] Reset & Idle state verified.");

        // Host prepares layer 1 and starts the inference
        for(p=0; p<NPIX; p=p+1) for(c=0; c<`PC; c=c+1) cur_in[p*`PC+c] = inval(p, c);
        feed_len = NPIX;
        set_layer_config(1);
        push_weights(1);
        start_inference = 1'b1;
        @(negedge clk);
        start_inference = 1'b0;

        for(step=0; step<STEPS; step=step+1) begin
            l = exp_layer[step];
            s = exp_sample[step];
            test_errors = 0;
            if(step == 0) $display("\n[TEST 2] Layer 1 (deterministic conv, input from DRAM)...");
            if(step == 1) $display("\n[TEST 3] Layer 2 (N-B, IC caching, MCD on output)...");
            if(step == 3) $display("\n[TEST 4] Samples 2..%0d (IC replay, layers 1..%0d skipped)...", S_RUN, N_RUN-B_RUN);

            while(!layer_done) @(negedge clk);

            if(dut.s5_layer_idx !== l || dut.s5_sample_idx !== s) begin
                $display("[ERROR] Executed sample %0d layer %0d, expected sample %0d layer %0d",
                         dut.s5_sample_idx, dut.s5_layer_idx, s, l);
                test_errors = test_errors + 1;
            end

            // Off-chip input traffic of this layer: none when replayed from the IC
            if((dram_beats - beats_prev) !== ((s > 1) ? 0 : NPIX)) begin
                $display("[ERROR] Layer %0d sample %0d read %0d DRAM beats", l, s, dram_beats - beats_prev);
                test_errors = test_errors + 1;
            end
            beats_prev = dram_beats;

            if(l == 1) begin
                for(p=0; p<NPIX*`PC; p=p+1) gin[p] = cur_in[p];
                golden_layer(1, 0, {`PF{1'b1}}, nw);
                for(p=0; p<NPIX*NCH; p=p+1) gold1[p] = gout[p];
                check_egress(nw);
                $display("  -> Layer 1 egress: %0d words, Ch0 = %0d %0d %0d %0d", eg_cnt, egress[0], egress[NCH], egress[2*NCH], egress[3*NCH]);
            end else if(l == 2) begin
                for(p=0; p<NPIX*NCH; p=p+1) gin[p] = gold1[p];
                golden_layer(2, 0, {`PF{1'b1}}, nw);
                for(p=0; p<NPIX*NCH; p=p+1) gold_u2[p] = gout[p];

                // IC must hold the pre-dropout layer N-B output
                for(p=0; p<NPIX; p=p+1) begin
                    for(f=0; f<NCH; f=f+1) begin
                        if($signed(dut.u_stage5.u_ic_buffer.mem[p][f*DW +: DW]) !== gold_u2[p*NCH+f]) begin
                            $display("[ERROR] IC word %0d ch %0d: got %0d, expected %0d", p, f,
                                     $signed(dut.u_stage5.u_ic_buffer.mem[p][f*DW +: DW]), gold_u2[p*NCH+f]);
                            test_errors = test_errors + 1;
                        end
                    end
                end
                if(ic_writes !== NPIX) begin
                    $display("[ERROR] IC writes: %0d, expected %0d", ic_writes, NPIX);
                    test_errors = test_errors + 1;
                end

                // DRAM egress of sample 1 carries mask 1 applied by Stage 4
                golden_layer(2, 1, mask_s[1], nw);
                check_egress(nw);
                $display("  -> ic_write_en during sample 1 layer 2: %0d words cached (pre-dropout)", ic_writes);
                $display("  -> Stage 4 Dropout Mask 1: 0x%h", mask_s[1]);
            end else begin
                for(p=0; p<NPIX*NCH; p=p+1) gin[p] = mask_s[s][(p % NCH) % `PF] ? gold_u2[p] : 0;
                golden_layer(3, 0, {`PF{1'b1}}, nw);
                for(f=0; f<NCH; f=f+1) gold3[(s-1)*NCH+f] = gout[f];
                check_egress(nw);
                $display("  -> Sample %0d layer 3: mask 0x%h (%s), logits Ch0..3 = %0d %0d %0d %0d", s, mask_s[s],
                         (s == 1) ? "Stage 4" : "IC replay", egress[0], egress[1], egress[2], egress[3]);
            end

            error_count = error_count + test_errors;
            if(step == 0 && test_errors == 0) $display("[TEST 2 PASSED] Layer 1 output matches golden model on all %0d words.", NPIX);
            if(step == 1 && test_errors == 0) $display("[TEST 3 PASSED] Layer 2 cached unmasked; DRAM egress masked with Mask 1.");
            if(step >= 3 && test_errors == 0) $display("[TEST 4 PASSED] Sample %0d: zero DRAM reads, fresh mask, layer 3 matches golden.", s);

            // Host prepares the next layer (config, weights, DRAM image)
            if(step + 1 < STEPS) begin
                set_layer_config(exp_layer[step+1]);
                for(p=0; p<NPIX*`PC; p=p+1) cur_in[p] = egress[p];
                feed_idx = 0;
                feed_len = (exp_sample[step+1] > 1) ? 0 : NPIX;
                eg_cnt = 0;
                push_weights(exp_layer[step+1]);
            end
        end

        // TEST CASE 5: Output reduction and IC savings
        $display("\n[TEST 5] Output Reduction & Intermediate-layer Caching savings...");
        test_errors = 0;
        wait_cnt = 0;
        while(!reduction_done && wait_cnt < 500) begin
            @(negedge clk);
            wait_cnt = wait_cnt + 1;
        end
        if(reduction_done !== 1) begin
            $display("[ERROR] reduction_done never asserted");
            test_errors = test_errors + 1;
        end

        for(f=0; f<NCH; f=f+1) begin
            sum = 0;
            sum_sq = 0;
            for(s=0; s<S_RUN; s=s+1) begin
                sum = sum + gold3[s*NCH+f];
                sum_sq = sum_sq + gold3[s*NCH+f] * gold3[s*NCH+f];
            end
            exp_mean = sum / S_RUN;
            exp_var = (sum_sq / S_RUN) - exp_mean * exp_mean;
            got_m = $signed(mean_prediction[f*DW +: DW]);
            got_v = uncertainty_score[f*VW +: VW];
            if(got_m !== exp_mean || got_v !== exp_var) begin
                $display("[ERROR] Ch %0d: Mean=%0d Var=%0d (Expected Mean=%0d Var=%0d)", f, got_m, got_v, exp_mean, exp_var);
                test_errors = test_errors + 1;
            end
            if(got_m == 0 || got_v == 0) begin
                $display("[ERROR] Ch %0d: zero statistic (Mean=%0d Var=%0d)", f, got_m, got_v);
                test_errors = test_errors + 1;
            end
        end
        $display("  -> Ch0 : Mean=%0d Var=%0d | Ch1 : Mean=%0d Var=%0d | Ch63 : Mean=%0d Var=%0d",
                 $signed(mean_prediction[0 +: DW]), uncertainty_score[0 +: VW],
                 $signed(mean_prediction[DW +: DW]), uncertainty_score[VW +: VW],
                 $signed(mean_prediction[(NCH-1)*DW +: DW]), uncertainty_score[(NCH-1)*VW +: VW]);

        if(mask_s[1] === mask_s[2] || mask_s[2] === mask_s[3] || mask_s[1] === mask_s[3]) begin
            $display("[ERROR] MC samples did not use independent masks");
            test_errors = test_errors + 1;
        end
        if(stage4_pops !== 1 || replay_pops !== S_RUN-1) begin
            $display("[ERROR] Mask pops: Stage 4=%0d, replay=%0d (expected 1, %0d)", stage4_pops, replay_pops, S_RUN-1);
            test_errors = test_errors + 1;
        end
        if(layer_pulses !== STEPS || done_pulses !== 1 || busy !== 0) begin
            $display("[ERROR] Layers executed=%0d (expected %0d), inference_done pulses=%0d, busy=%b", layer_pulses, STEPS, done_pulses, busy);
            test_errors = test_errors + 1;
        end
        $display("  -> Layers executed: %0d = (N-B) + B*S (naive N*S = %0d)", layer_pulses, N_RUN * S_RUN);
        $display("  -> DRAM input beats: %0d (naive %0d)", dram_beats, N_RUN * S_RUN * NPIX);
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 5 PASSED] Mean & uncertainty bit-exact and non-zero on all %0d channels.", NCH);

        // Final Summary
        #50;
        if(error_count == 0) begin
            $display("   ALL BCNN_TOP SYSTEM TEST CASES PASSED PERFECTLY! (0 ERRORS)        ");
        end else begin
            $display("   TESTBENCH COMPLETED WITH %0d ERRORS.                        ", error_count);
        end

        $finish;
    end

endmodule
