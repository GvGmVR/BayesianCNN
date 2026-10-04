//==============================================================================
// Module: tb_uamh_top.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: UAMH - Unified System Testbench (Innovations 1, 2 and 3)
//------------------------------------------------------------------------------
// Purpose:
//   Self-checking end-to-end verification of bcnn_top with the three UAMH
//   innovations disabled, enabled one at a time, and enabled together.
//
// Architectural Inputs:
//   - None (the testbench models the host CPU and the off-chip DRAM).
//
// Architectural Outputs:
//   - Console pass/fail report and sim/uamh_top_simulation.vcd.
//
// Description:
//   The DUT is built with a constrained 8-row (512-byte) IC so that U-Tagging
//   has something to manage. Two networks are used:
//     A: N=3, B=1, 2x2x64 map, conv layers with varied weights, global avg pool
//        in layer 3 (as in tb_bcnn_top.v). Its 4 cached lines always fit.
//     B: N=4, B=2, 4x4x64 map. Layers 1-2 are identity convolutions, so the
//        input image sets the content (and tag) of each of the 16 cached lines,
//        which overflow the IC. Layer 3 pools 16 -> 4 pixels, layer 4 4 -> 1.
//   Host / DRAM model:
//     - streams each layer's input (the previous layer's egress) and weights;
//     - stores ic_spill_data beats in dram_spill_mem and, during replay layers,
//       returns them in order whenever dram_data_ready requests a line;
//     - pushes the next layer's weights only once eval_pending has cleared and
//       busy shows that another layer will run.
//   Checking:
//     - every executed (sample, layer) step against a bit-exact golden model
//       (conv -> quant -> ReLU -> pool -> MCD) using the masks actually popped;
//     - every replayed line against the unmasked layer N-B output under that
//       sample's replay mask (UMPS unpacking and spill return included);
//     - UMPS / U-Tagging telemetry against a golden admission model;
//     - early exit point, samples_executed and the final mean / variance
//       against a golden convergence model (exact division by s_actual).
//   Test 1: Baseline parity (all innovations off), network A, S=3.
//   Test 2: UMPS, network A with small layer-2 activations, S=3.
//   Test 3: U-Tagging with spill / replay, network B, S=3.
//   Test 4: Early exit, network A, S=10, epsilon=40, S_min=4.
//   Test 5: UMPS + U-Tagging + early exit together, network B, S=10.
//   Every test also requires at least half of the channels to have non-zero
//   variance, so the dropout masks demonstrably shape the prediction.
//   Run with +nodump to skip the (large) waveform dump.
//==============================================================================

`timescale 1ns / 1ps
`include "bcnn_pkg.vh"

module tb_uamh_top;

    localparam DW = `DATA_WIDTH;
    localparam NCH = `PF * `PV;
    localparam VW = 2 * `DATA_WIDTH;
    localparam INT_MAX = (1 << (DW-1)) - 1;
    localparam INT_MIN = -(1 << (DW-1));
    localparam INT4_MAX = (1 << (`INT4_WIDTH-1)) - 1;
    localparam INT4_MIN = -(1 << (`INT4_WIDTH-1));
    localparam POOL_WIN = 1 << `POOL_CNT_WIDTH;

    // Constrained IC: 8 rows x 64 bytes = 512 bytes (8 full-length lines)
    localparam IC_DEPTH = 8;
    localparam IC_AW = 3;
    localparam IC_BYTES = IC_DEPTH * NCH;

    localparam MAX_PIX = 16;
    localparam MAX_L = 4;
    localparam MAX_S = 10;
    localparam EE_EPS = 40;

    // UAMH configuration sized to the DUT ports
    localparam signed [`UMPS_THRESH_WIDTH-1:0] UMPS_TAU = `UMPS_DEFAULT_THRESH;
    localparam signed [DW-1:0] UTAG_ZT = `UTAG_ZERO_THRESH;
    localparam [$clog2(NCH+1)-1:0] UTAG_HC = `UTAG_HIGH_COUNT_TH;
    localparam [`EARLY_EXIT_THRESH_WIDTH-1:0] EE_THRESH = EE_EPS;
    localparam [`SAMPLE_CNT_WIDTH-1:0] EE_MIN = `MIN_SAMPLES_EXIT;

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
    reg umps_en, utag_en, early_exit_en;

    wire busy, inference_done, dram_data_ready, weight_full;
    wire [(NCH*DW)-1:0] mean_prediction;
    wire [(NCH*VW)-1:0] uncertainty_score;
    wire reduction_done;
    wire [(NCH*DW)-1:0] layer_features_out;
    wire layer_features_valid, layer_done;
    wire [IC_AW+1:0] ic_lines_cached;
    wire [IC_AW+$clog2(NCH):0] ic_bytes_used;
    wire [(NCH*DW)-1:0] ic_spill_data;
    wire ic_spill_valid;
    wire [IC_AW+1:0] high_u_cached_count, low_u_bypassed_count;
    wire [`UTAG_WIDTH-1:0] current_line_u_tag;
    wire early_exit_triggered;
    wire [`SAMPLE_CNT_WIDTH-1:0] samples_executed;
    wire eval_pending;

    bcnn_top #(
        .IC_RAM_DEPTH(IC_DEPTH),
        .IC_ADDR_WIDTH(IC_AW)
    ) dut (
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
        .umps_en(umps_en),
        .umps_thresh(UMPS_TAU),
        .utag_en(utag_en),
        .utag_zero_thresh(UTAG_ZT),
        .utag_high_count_th(UTAG_HC),
        .early_exit_en(early_exit_en),
        .early_exit_thresh(EE_THRESH),
        .early_exit_min_samples(EE_MIN),
        .mean_prediction(mean_prediction),
        .uncertainty_score(uncertainty_score),
        .reduction_done(reduction_done),
        .layer_features_out(layer_features_out),
        .layer_features_valid(layer_features_valid),
        .layer_done(layer_done),
        .ic_lines_cached(ic_lines_cached),
        .ic_bytes_used(ic_bytes_used),
        .ic_spill_data(ic_spill_data),
        .ic_spill_valid(ic_spill_valid),
        .high_u_cached_count(high_u_cached_count),
        .low_u_bypassed_count(low_u_bypassed_count),
        .current_line_u_tag(current_line_u_tag),
        .early_exit_triggered(early_exit_triggered),
        .samples_executed(samples_executed),
        .eval_pending(eval_pending)
    );

    always #2.27 clk = ~clk;

    //--------------------------------------------------------------------------
    // Run configuration (set per test)
    //--------------------------------------------------------------------------
    integer run_N, run_B, run_S, img_kind;
    integer cfg_relu [1:MAX_L];
    integer cfg_pool [1:MAX_L];
    integer cfg_scale [1:MAX_L];
    integer cfg_shift [1:MAX_L];
    integer cfg_bias [1:MAX_L];
    integer cfg_h [1:MAX_L];
    integer cfg_w [1:MAX_L];
    integer cfg_ident [1:MAX_L];

    //--------------------------------------------------------------------------
    // Host DRAM model
    //--------------------------------------------------------------------------
    integer cur_in [0:(MAX_PIX*`PC)-1];
    integer egress [0:(MAX_PIX*NCH)-1];
    reg [(NCH*DW)-1:0] dram_spill_mem [0:MAX_PIX-1];
    integer feed_idx, feed_len, eg_cnt, spill_wr, spill_rd;
    integer dram_beats, spill_beats;
    reg drv_spill;

    //--------------------------------------------------------------------------
    // Golden model state
    //--------------------------------------------------------------------------
    integer gin [0:(MAX_PIX*NCH)-1];
    integer gpre [0:(MAX_PIX*NCH)-1];
    integer gout [0:(MAX_PIX*NCH)-1];
    integer gprev [0:(MAX_PIX*NCH)-1];
    integer gu [0:(MAX_PIX*NCH)-1];
    integer gfinal [0:(MAX_S*NCH)-1];
    integer gu_n, s_done;
    reg [`PF-1:0] mask_tab [0:((MAX_S+1)*(MAX_L+1))-1];

    // Golden admission model results
    integer exp_lines, exp_bytes, exp_spills, exp_hi, exp_lo, int4_channels;

    // Counters and bookkeeping
    integer error_count, test_errors;
    integer layers_run, s4_pops, rp_pops, rp_idx, rp_checked, done_pulses, rd_at_done;
    integer spills_t3, spills_t5, nz_var;
    integer i, p, c, f;

    //--------------------------------------------------------------------------
    // Data patterns
    //--------------------------------------------------------------------------
    // Network B pixel kinds: H L Z H L H Z L Z H Z L H Z L H
    function integer pix_kind(input integer pi);
        begin
            case(pi % 16)
                0, 3, 5, 9, 12, 15: pix_kind = 2;   // high activity
                1, 4, 7, 11, 14: pix_kind = 1;      // low activity
                default: pix_kind = 0;               // flat background
            endcase
        end
    endfunction

    function integer img_val(input integer pi, input integer ci);
        begin
            if(img_kind == 2) begin
                case(pix_kind(pi))
                    2: img_val = 20 + ((pi * 13 + ci * 7) % 80);
                    1: img_val = (ci < 3) ? 30 + pi : 5 + ((ci + pi) % 3);
                    default: img_val = 0;
                endcase
            end else begin
                img_val = ((pi * 7 + ci * 3) % 11) - 5;
            end
        end
    endfunction

    function integer wval(input integer li, input integer fi, input integer ci);
        begin
            if(cfg_ident[li]) wval = (fi == ci) ? 1 : 0;
            else wval = ((li * 17 + fi * 31 + ci * 13 + fi * ci * 7) % 9) - 4;
        end
    endfunction

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

    function integer mtab_idx(input integer si, input integer li);
        begin
            mtab_idx = si * (MAX_L + 1) + li;
        end
    endfunction

    //--------------------------------------------------------------------------
    // Golden layer: gin (npix pixels) -> conv / quant / ReLU -> pool -> gout
    //--------------------------------------------------------------------------
    task golden_layer(input integer li, input integer npix, output integer nwords);
        integer pi, fi, ci, acc, wi;
        begin
            for(pi=0; pi<npix; pi=pi+1) begin
                for(fi=0; fi<NCH; fi=fi+1) begin
                    acc = 0;
                    for(ci=0; ci<`PC; ci=ci+1) acc = acc + gin[pi*`PC+ci] * wval(li, fi, ci);
                    gpre[pi*NCH+fi] = quant(li, acc);
                end
            end
            if(cfg_pool[li] == `POOL_MODE_AVG) begin
                nwords = npix / POOL_WIN;
                for(wi=0; wi<nwords; wi=wi+1) begin
                    for(fi=0; fi<NCH; fi=fi+1) begin
                        acc = 0;
                        for(pi=0; pi<POOL_WIN; pi=pi+1) acc = acc + gpre[(wi*POOL_WIN+pi)*NCH+fi];
                        gout[wi*NCH+fi] = acc >>> `POOL_CNT_WIDTH;
                    end
                end
            end else begin
                nwords = npix;
                for(pi=0; pi<npix*NCH; pi=pi+1) gout[pi] = gpre[pi];
            end
        end
    endtask

    // Golden UMPS / U-Tagging decisions for the cached layer N-B (in write order)
    task admission_model;
        integer wi, ci, v, v1, act, high, len, occ, tag, admit, full;
        begin
            exp_lines = 0;
            exp_bytes = 0;
            exp_spills = 0;
            exp_hi = 0;
            exp_lo = 0;
            int4_channels = 0;
            occ = 0;
            for(wi=0; wi<gu_n; wi=wi+1) begin
                act = 0;
                high = 0;
                len = 0;
                for(ci=0; ci<NCH; ci=ci+1) begin
                    v = gu[wi*NCH+ci];
                    if(v > UTAG_ZT || -v > UTAG_ZT) act = act + 1;
                    if(v >= INT4_MIN && v <= INT4_MAX) int4_channels = int4_channels + 1;
                    if(!(v >= INT4_MIN && v <= INT4_MAX && v <= UMPS_TAU && -v <= UMPS_TAU)) high = high + 1;
                end
                for(ci=0; ci<NCH; ci=ci+2) begin
                    v = gu[wi*NCH+ci];
                    v1 = gu[wi*NCH+ci+1];
                    if(umps_en && v >= INT4_MIN && v <= INT4_MAX && v <= UMPS_TAU && -v <= UMPS_TAU &&
                       v1 >= INT4_MIN && v1 <= INT4_MAX && v1 <= UMPS_TAU && -v1 <= UMPS_TAU) len = len + 1;
                    else len = len + 2;
                end
                tag = (act == 0) ? 0 : (high >= UTAG_HC) ? 2 : 1;
                full = (occ > IC_BYTES - NCH) || (exp_lines == 2 * IC_DEPTH);
                if(utag_en && (occ >= (IC_BYTES * `UTAG_CAP_THRESH_PCT) / 100) && tag == 0) admit = 0;
                else admit = !full;
                if(admit) begin
                    occ = occ + len;
                    exp_lines = exp_lines + 1;
                    if(utag_en && tag == 2) exp_hi = exp_hi + 1;
                end else begin
                    exp_spills = exp_spills + 1;
                    if(utag_en && tag != 2) exp_lo = exp_lo + 1;
                end
            end
            exp_bytes = occ;
        end
    endtask

    //--------------------------------------------------------------------------
    // Golden check of one executed layer step (called at its layer_done)
    //--------------------------------------------------------------------------
    task golden_step(input integer si, input integer li);
        integer npix, nw, k, replay, exp_dram, exp_spill;
        begin
            npix = cfg_h[li] * cfg_w[li];
            replay = (si > 1) && (li == run_N - run_B + 1);

            if(li == 1) begin
                for(k=0; k<npix*`PC; k=k+1) gin[k] = img_val(k / `PC, k % `PC);
            end else if(replay) begin
                // Replayed input: unmasked layer N-B output under this sample's replay mask
                for(k=0; k<gu_n*NCH; k=k+1) gin[k] = mask_tab[mtab_idx(si, run_N-run_B)][(k % NCH) % `PF] ? gu[k] : 0;
            end else begin
                for(k=0; k<npix*NCH; k=k+1) gin[k] = gprev[k];
            end

            golden_layer(li, npix, nw);

            if(si == 1 && li == run_N - run_B) begin
                for(k=0; k<nw*NCH; k=k+1) gu[k] = gout[k];
                gu_n = nw;
                admission_model;
            end

            // Stage 4 MCD on the outputs of layers N-B .. N-1
            if(li >= run_N - run_B && li < run_N) begin
                for(k=0; k<nw*NCH; k=k+1) if(!mask_tab[mtab_idx(si, li)][(k % NCH) % `PF]) gout[k] = 0;
            end

            if(eg_cnt !== nw) begin
                $display("[ERROR] s%0d L%0d: %0d egress words, expected %0d", si, li, eg_cnt, nw);
                test_errors = test_errors + 1;
            end
            for(k=0; k<nw*NCH; k=k+1) begin
                if(egress[k] !== gout[k]) begin
                    if(test_errors < 10) $display("[ERROR] s%0d L%0d word %0d ch %0d: got %0d, expected %0d", si, li, k / NCH, k % NCH, egress[k], gout[k]);
                    test_errors = test_errors + 1;
                end
            end

            // Off-chip input traffic: DRAM beats for normal layers, spill returns for replay layers
            exp_dram = replay ? 0 : npix;
            exp_spill = replay ? exp_spills : 0;
            if(dram_beats !== exp_dram || spill_beats !== exp_spill) begin
                $display("[ERROR] s%0d L%0d: %0d DRAM beats / %0d spill returns, expected %0d / %0d", si, li, dram_beats, spill_beats, exp_dram, exp_spill);
                test_errors = test_errors + 1;
            end

            for(k=0; k<nw*NCH; k=k+1) gprev[k] = gout[k];
            if(li == run_N) begin
                for(k=0; k<NCH; k=k+1) gfinal[(si-1)*NCH+k] = gout[k];
                s_done = si;
            end
        end
    endtask

    //--------------------------------------------------------------------------
    // Golden reduction and convergence model over the executed samples
    //--------------------------------------------------------------------------
    task check_reduction;
        integer si, fi, v, m, var_c, d, delta, cnt, s_exit;
        integer sum [0:NCH-1];
        integer sq [0:NCH-1];
        integer pv [0:NCH-1];
        begin
            for(fi=0; fi<NCH; fi=fi+1) begin
                sum[fi] = 0;
                sq[fi] = 0;
                pv[fi] = 0;
            end
            cnt = 0;
            s_exit = run_S;
            for(si=1; si<=s_done && s_exit == run_S; si=si+1) begin
                delta = 0;
                for(fi=0; fi<NCH; fi=fi+1) begin
                    v = gfinal[(si-1)*NCH+fi];
                    sum[fi] = sum[fi] + v;
                    sq[fi] = sq[fi] + v * v;
                    m = sum[fi] / si;
                    var_c = sq[fi] / si - m * m;
                    d = (var_c > pv[fi]) ? var_c - pv[fi] : pv[fi] - var_c;
                    if(si > 1 && d > delta) delta = d;
                    pv[fi] = var_c;
                end
                if(si < run_S) begin
                    if(early_exit_en && si > 1 && si >= EE_MIN && delta <= EE_EPS) cnt = cnt + 1;
                    else cnt = 0;
                    if(cnt >= `CONV_STABILITY_COUNT) s_exit = si;
                end
            end

            if(s_done !== s_exit || samples_executed !== s_exit || early_exit_triggered !== (s_exit < run_S)) begin
                $display("[ERROR] Executed %0d samples (samples_executed=%0d, triggered=%b), golden convergence model expects %0d",
                         s_done, samples_executed, early_exit_triggered, s_exit);
                test_errors = test_errors + 1;
            end

            for(fi=0; fi<NCH; fi=fi+1) begin
                sum[fi] = 0;
                sq[fi] = 0;
                for(si=1; si<=s_exit; si=si+1) begin
                    v = gfinal[(si-1)*NCH+fi];
                    sum[fi] = sum[fi] + v;
                    sq[fi] = sq[fi] + v * v;
                end
                m = sum[fi] / s_exit;
                var_c = sq[fi] / s_exit - m * m;
                if($signed(mean_prediction[fi*DW +: DW]) !== m || uncertainty_score[fi*VW +: VW] !== var_c) begin
                    if(test_errors < 10) $display("[ERROR] Ch %0d: Mean=%0d Var=%0d, expected %0d %0d (over %0d samples)", fi,
                             $signed(mean_prediction[fi*DW +: DW]), uncertainty_score[fi*VW +: VW], m, var_c, s_exit);
                    test_errors = test_errors + 1;
                end
            end
        end
    endtask

    //--------------------------------------------------------------------------
    // Host helpers
    //--------------------------------------------------------------------------
    task set_layer_config(input integer li);
        begin
            H = cfg_h[li];
            W = cfg_w[li];
            W_tiles = cfg_w[li] / `PV;
            relu_en = cfg_relu[li];
            pool_mode = cfg_pool[li];
            quant_scale = cfg_scale[li];
            quant_shift = cfg_shift[li];
            quant_bias = cfg_bias[li];
        end
    endtask

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

    // Host DRAM: input beats, or spilled lines when the chip is replaying layer N-B+1
    always @(negedge clk) begin
        if(dram_data_valid) begin
            if(drv_spill) begin
                spill_rd = spill_rd + 1;
                spill_beats = spill_beats + 1;
            end else begin
                feed_idx = feed_idx + 1;
                dram_beats = dram_beats + 1;
            end
        end
        if(dram_data_ready && dut.src_replay && spill_rd < spill_wr) begin
            dram_data_in = dram_spill_mem[spill_rd];
            drv_spill = 1'b1;
            dram_data_valid = 1'b1;
        end else if(dram_data_ready && !dut.src_replay && feed_idx < feed_len) begin
            for(c=0; c<`PC; c=c+1) dram_data_in[c*DW +: DW] = cur_in[feed_idx*`PC+c];
            drv_spill = 1'b0;
            dram_data_valid = 1'b1;
        end else begin
            dram_data_valid = 1'b0;
        end
    end

    // Egress capture, spill capture, mask capture and protocol monitors
    integer ec, rc, rexp;
    always @(posedge clk) begin
        if(layer_features_valid) begin
            for(ec=0; ec<NCH; ec=ec+1) egress[eg_cnt*NCH+ec] = $signed(layer_features_out[ec*DW +: DW]);
            eg_cnt = eg_cnt + 1;
        end
        if(ic_spill_valid) begin
            dram_spill_mem[spill_wr] = ic_spill_data;
            spill_wr = spill_wr + 1;
        end
        if(dut.stage4_mask_pop) begin
            mask_tab[mtab_idx(dut.s5_sample_idx, dut.s5_layer_idx)] = dut.sampler_mask_out;
            s4_pops = s4_pops + 1;
        end
        if(dut.replay_mask_pop) begin
            mask_tab[mtab_idx(dut.s5_sample_idx, run_N-run_B)] = dut.sampler_mask_out;
            rp_pops = rp_pops + 1;
        end
        // Every replayed line = cached layer N-B output under this sample's replay mask
        if(dut.replay_valid) begin
            for(rc=0; rc<NCH; rc=rc+1) begin
                rexp = mask_tab[mtab_idx(dut.s5_sample_idx, run_N-run_B)][rc % `PF] ? gu[rp_idx*NCH+rc] : 0;
                if($signed(dut.replay_features[rc*DW +: DW]) !== rexp) begin
                    if(test_errors < 10) $display("[ERROR] Replay s%0d line %0d ch %0d: got %0d, expected %0d", dut.s5_sample_idx, rp_idx, rc,
                             $signed(dut.replay_features[rc*DW +: DW]), rexp);
                    test_errors = test_errors + 1;
                end
            end
            rp_idx = rp_idx + 1;
            rp_checked = rp_checked + 1;
        end
        if(dut.start_compute_r && dut.weight_empty) begin
            $display("[ERROR] Layer compute started before its weights were pushed");
            test_errors = test_errors + 1;
        end
        if(inference_done) begin
            done_pulses = done_pulses + 1;
            rd_at_done = reduction_done;
        end
    end

    //--------------------------------------------------------------------------
    // Run one inference with the current configuration
    //--------------------------------------------------------------------------
    task run_inference;
        integer si, li, l_next, more, k, wait_cnt;
        begin
            eg_cnt = 0;
            feed_idx = 0;
            spill_wr = 0;
            spill_rd = 0;
            dram_beats = 0;
            spill_beats = 0;
            layers_run = 0;
            s4_pops = 0;
            rp_pops = 0;
            rp_idx = 0;
            rp_checked = 0;
            done_pulses = 0;
            rd_at_done = 0;
            s_done = 0;
            gu_n = 0;
            exp_spills = 0;

            total_layers_N = run_N;
            bayesian_layers_B = run_B;
            total_samples_S = run_S;
            for(k=0; k<cfg_h[1]*cfg_w[1]*`PC; k=k+1) cur_in[k] = img_val(k / `PC, k % `PC);
            feed_len = cfg_h[1] * cfg_w[1];
            set_layer_config(1);
            push_weights(1);
            start_inference = 1'b1;
            @(negedge clk);
            start_inference = 1'b0;

            more = 1;
            while(more) begin
                while(!layer_done) @(negedge clk);
                si = dut.s5_sample_idx;
                li = dut.s5_layer_idx;
                layers_run = layers_run + 1;
                golden_step(si, li);

                // Next layer: same sample, or layer N-B+1 of the next sample (replay)
                l_next = (li < run_N) ? li + 1 : run_N - run_B + 1;
                for(k=0; k<eg_cnt*NCH; k=k+1) cur_in[k] = egress[k];
                feed_len = eg_cnt;
                feed_idx = 0;
                eg_cnt = 0;
                spill_rd = 0;
                rp_idx = 0;
                dram_beats = 0;
                spill_beats = 0;
                set_layer_config(l_next);

                // Weights only once the early-exit decision is known and another layer will run
                @(negedge clk);
                while(eval_pending) @(negedge clk);
                repeat(2) @(negedge clk);
                if(busy) push_weights(l_next);
                else more = 0;
            end

            wait_cnt = 0;
            while(!reduction_done && wait_cnt < 500) begin
                @(negedge clk);
                wait_cnt = wait_cnt + 1;
            end
            repeat(2) @(negedge clk);
            if(reduction_done !== 1 || done_pulses !== 1) begin
                $display("[ERROR] reduction_done=%b, inference_done pulses=%0d", reduction_done, done_pulses);
                test_errors = test_errors + 1;
            end
            check_reduction;

            if(ic_lines_cached !== exp_lines || ic_bytes_used !== exp_bytes ||
               high_u_cached_count !== exp_hi || low_u_bypassed_count !== exp_lo) begin
                $display("[ERROR] Telemetry lines=%0d bytes=%0d high_u=%0d low_u=%0d, expected %0d %0d %0d %0d",
                         ic_lines_cached, ic_bytes_used, high_u_cached_count, low_u_bypassed_count, exp_lines, exp_bytes, exp_hi, exp_lo);
                test_errors = test_errors + 1;
            end
            nz_var = 0;
            for(k=0; k<NCH; k=k+1) if(uncertainty_score[k*VW +: VW] != 0) nz_var = nz_var + 1;
            if(2 * nz_var < NCH) begin
                $display("[ERROR] Only %0d of %0d channels have non-zero variance (degenerate prediction)", nz_var, NCH);
                test_errors = test_errors + 1;
            end
            if(rp_checked !== (s_done - 1) * gu_n || rp_pops !== s_done - 1) begin
                $display("[ERROR] Replayed %0d lines with %0d replay masks, expected %0d / %0d", rp_checked, rp_pops, (s_done - 1) * gu_n, s_done - 1);
                test_errors = test_errors + 1;
            end
        end
    endtask

    task report(input integer tnum);
        begin
            $display("  -> layers executed=%0d, samples_executed=%0d of %0d, early_exit_triggered=%b, reduction_done at exit=%0d",
                     layers_run, samples_executed, run_S, early_exit_triggered, rd_at_done);
            $display("  -> IC: %0d lines in %0d bytes (uncompressed %0d), spilled=%0d, high_u_cached=%0d, low_u_bypassed=%0d, replayed lines checked=%0d",
                     ic_lines_cached, ic_bytes_used, ic_lines_cached * NCH, spill_wr, high_u_cached_count, low_u_bypassed_count, rp_checked);
            $display("  -> Ch0 mean=%0d var=%0d | Ch1 mean=%0d var=%0d | %0d of %0d channels with non-zero variance",
                     $signed(mean_prediction[0 +: DW]), uncertainty_score[0 +: VW], $signed(mean_prediction[DW +: DW]), uncertainty_score[VW +: VW], nz_var, NCH);
        end
    endtask

    // Network A: N=3, B=1, 2x2 map; l2_shift sets how small the cached activations are
    task setup_net_a(input integer samples, input integer l2_shift, input integer l3_shift);
        begin
            run_N = 3;
            run_B = 1;
            run_S = samples;
            img_kind = 0;
            for(i=1; i<=MAX_L; i=i+1) begin
                cfg_h[i] = 2;
                cfg_w[i] = 2;
                cfg_ident[i] = 0;
                cfg_scale[i] = 1;
            end
            cfg_relu[1] = 1; cfg_pool[1] = `POOL_MODE_BYPASS; cfg_shift[1] = 4; cfg_bias[1] = 2;
            cfg_relu[2] = 1; cfg_pool[2] = `POOL_MODE_BYPASS; cfg_shift[2] = l2_shift; cfg_bias[2] = 1;
            cfg_relu[3] = 0; cfg_pool[3] = `POOL_MODE_AVG; cfg_shift[3] = l3_shift; cfg_bias[3] = 3;
        end
    endtask

    // Network B: N=4, B=2, 4x4 map, identity layers 1-2 so the image sets each cached line
    task setup_net_b(input integer samples, input integer l3_shift, input integer l4_shift);
        begin
            run_N = 4;
            run_B = 2;
            run_S = samples;
            img_kind = 2;
            for(i=1; i<=MAX_L; i=i+1) begin
                cfg_scale[i] = 1;
                cfg_bias[i] = 0;
                cfg_shift[i] = 0;
                cfg_relu[i] = 1;
                cfg_pool[i] = `POOL_MODE_BYPASS;
                cfg_ident[i] = (i <= 2);
                cfg_h[i] = 4;
                cfg_w[i] = 4;
            end
            cfg_pool[3] = `POOL_MODE_AVG; cfg_shift[3] = l3_shift;
            cfg_h[4] = 2; cfg_w[4] = 2;
            cfg_pool[4] = `POOL_MODE_AVG; cfg_shift[4] = l4_shift; cfg_relu[4] = 0; cfg_bias[4] = 1;
        end
    endtask

    function integer masks_distinct(input integer layer_idx);
        integer a, b;
        begin
            masks_distinct = 1;
            for(a=1; a<=s_done; a=a+1)
                for(b=a+1; b<=s_done; b=b+1)
                    if(mask_tab[mtab_idx(a, layer_idx)] === mask_tab[mtab_idx(b, layer_idx)]) masks_distinct = 0;
        end
    endfunction

    // Watchdog Timeout
    initial begin
        #500000;
        $display("TIMEOUT");
        $finish;
    end

    initial begin
        // Waveforms by default; run "vvp sim/uamh_top_sim.out +nodump" for a fast regression
        if(!$test$plusargs("nodump")) begin
            $dumpfile("sim/uamh_top_simulation.vcd");
            $dumpvars(0, tb_uamh_top);
        end

        clk = 0;
        rst_n = 0;
        start_inference = 0;
        L_frames = 1;
        C_tiles = 1;
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
        umps_en = 0;
        utag_en = 0;
        early_exit_en = 0;
        drv_spill = 0;
        error_count = 0;
        spills_t3 = 0;
        spills_t5 = 0;

        $display("   STARTING UAMH UNIFIED SYSTEM TESTBENCH (IC = %0d rows / %0d bytes)", IC_DEPTH, IC_BYTES);

        repeat(3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // TEST 1: Baseline parity
        $display("\n[TEST 1] Baseline parity: UMPS, U-Tagging and early exit OFF (N=3, B=1, S=3)...");
        test_errors = 0;
        umps_en = 0; utag_en = 0; early_exit_en = 0;
        setup_net_a(3, 3, 4);
        run_inference;
        report(1);
        if(layers_run !== (run_N - run_B) + run_B * run_S || early_exit_triggered !== 0 || spill_wr !== 0 || ic_bytes_used !== ic_lines_cached * NCH) begin
            $display("[ERROR] Baseline behaviour changed: layers=%0d spills=%0d", layers_run, spill_wr);
            test_errors = test_errors + 1;
        end
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 1 PASSED] 5 layers instead of 9; every layer, replay and statistic bit-exact.");

        // TEST 2: UMPS
        $display("\n[TEST 2] Innovation 1: UMPS (umps_thresh=%0d)...", UMPS_TAU);
        test_errors = 0;
        umps_en = 1; utag_en = 0; early_exit_en = 0;
        setup_net_a(3, 6, 4);
        run_inference;
        report(2);
        $display("  -> %0d of %0d cached channels fit INT4, compression %0d.%02dx",
                 int4_channels, gu_n * NCH, (ic_lines_cached * NCH) / ic_bytes_used, ((ic_lines_cached * NCH * 100) / ic_bytes_used) % 100);
        if(2 * int4_channels < gu_n * NCH || !(ic_bytes_used < ic_lines_cached * NCH) || 5 * ic_bytes_used > 4 * ic_lines_cached * NCH) begin
            $display("[ERROR] UMPS compression below 1.25x or data not 50%% INT4");
            test_errors = test_errors + 1;
        end
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 2 PASSED] Lines compressed >= 1.25x; replay unpacked bit-exact; prediction matches.");

        // TEST 3: U-Tagging
        $display("\n[TEST 3] Innovation 3: U-Tagging with spill / replay (N=4, B=2, S=3, 16 lines into 8-line IC)...");
        test_errors = 0;
        umps_en = 0; utag_en = 1; early_exit_en = 0;
        setup_net_b(3, 7, 4);
        run_inference;
        report(3);
        spills_t3 = spill_wr;
        if(low_u_bypassed_count == 0 || high_u_cached_count == 0 || spill_wr == 0 || masks_distinct(run_N - run_B) == 0) begin
            $display("[ERROR] U-Tagging did not filter / prioritise / re-mask as expected");
            test_errors = test_errors + 1;
        end
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 3 PASSED] ZERO lines spilled under pressure, HIGH lines kept on-chip, spills returned in order, fresh masks.");

        // TEST 4: Early exit
        $display("\n[TEST 4] Innovation 2: Early exit (S=10, epsilon=%0d, S_min=%0d)...", EE_EPS, EE_MIN);
        test_errors = 0;
        umps_en = 0; utag_en = 0; early_exit_en = 1;
        setup_net_a(10, 3, 4);
        run_inference;
        report(4);
        if(early_exit_triggered !== 1 || !(samples_executed < run_S) || rd_at_done !== 1) begin
            $display("[ERROR] Early exit not taken or reduction not ready at exit");
            test_errors = test_errors + 1;
        end
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 4 PASSED] Exited after %0d of %0d samples; mean / variance exact over s_actual.", samples_executed, run_S);

        // TEST 5: All three together
        $display("\n[TEST 5] UMPS + U-Tagging + early exit together (N=4, B=2, S=10)...");
        test_errors = 0;
        umps_en = 1; utag_en = 1; early_exit_en = 1;
        setup_net_b(10, 7, 4);
        run_inference;
        report(5);
        spills_t5 = spill_wr;
        $display("  -> spills: %0d with UMPS vs %0d without (Test 3)", spills_t5, spills_t3);
        if(!(ic_bytes_used < ic_lines_cached * NCH) || !(spills_t5 < spills_t3) || low_u_bypassed_count == 0 ||
           high_u_cached_count == 0 || early_exit_triggered !== 1 || !(samples_executed < run_S)) begin
            $display("[ERROR] Not all three innovations reported savings");
            test_errors = test_errors + 1;
        end
        error_count = error_count + test_errors;
        if(test_errors == 0) $display("[TEST 5 PASSED] Compression, uncertainty-aware spilling and early exit active together.");

        #50;
        if(error_count == 0) begin
            $display("   ALL UAMH SYSTEM TEST CASES PASSED PERFECTLY! (0 ERRORS)        ");
        end else begin
            $display("   TESTBENCH COMPLETED WITH %0d ERRORS.                        ", error_count);
        end

        $finish;
    end

endmodule
