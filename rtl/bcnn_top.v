//==============================================================================
// Module: bcnn_top.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: Top - Master System Integration (NNE + Bernoulli Sampler + IC)
//------------------------------------------------------------------------------
// Purpose:
//   Master chip wrapper integrating all five stages into the accelerator of
//   Fig. 4 in Fan et al.: smart buffers, Bernoulli sampler, processing engine,
//   functional/dropout engine, and intermediate-layer caching with output
//   reduction, plus the layer-level Controller that sequences them.
//
// Architectural Inputs:
//   - clk, rst_n            : Clock (220 MHz) and active-low reset.
//   - start_inference       : 1-cycle strobe to run an S-sample inference.
//   - total_layers_N, bayesian_layers_B, total_samples_S : Network config.
//   - H, W, C_tiles, W_tiles, L_frames, KH, KW, KL, stride, mode_3d :
//                             Geometry of the layer being executed.
//   - relu_en, sc_en, pool_mode, quant_scale, quant_shift, quant_bias :
//                             Operator config of the layer being executed.
//   - dram_data_valid, dram_data_in : Layer input stream from off-chip memory.
//   - weight_push, weight_din       : Layer weights (PC x PF bytes per word).
//   - sc_features_in        : Cached input for ResNet shortcut addition.
//   - load_seed, seed_in    : Stage 2 LFSR re-seeding.
//   - umps_en, umps_thresh  : UAMH Innovation 1 (UMPS) enable and activity
//                             threshold; tie umps_en to 0 for baseline behaviour.
//   - utag_en, utag_zero_thresh, utag_high_count_th : UAMH Innovation 3
//                             (U-Tagging) enable and tag thresholds; tie utag_en
//                             to 0 for baseline behaviour.
//   - early_exit_en, early_exit_thresh, early_exit_min_samples : UAMH
//                             Innovation 2 (early exit) enable, tolerance and
//                             warm-up; tie early_exit_en to 0 for baseline behaviour.
//
// Architectural Outputs:
//   - busy, inference_done  : Inference status.
//   - dram_data_ready       : Ingress ready. During an IC replay layer it instead
//                             requests the next spilled line (U-Tagging).
//   - weight_full           : Weight buffer full.
//   - mean_prediction, uncertainty_score, reduction_done : Eq. 1 results.
//   - layer_features_out, layer_features_valid : Layer output egress to DRAM.
//   - layer_done            : 1-cycle strobe once a layer has fully drained.
//   - ic_lines_cached, ic_bytes_used : UMPS telemetry, IC lines (pixels) cached
//                             and bytes of IC storage they occupy.
//   - ic_spill_data, ic_spill_valid : Layer N-B lines not admitted on-chip
//                             (unmasked INT8), to be stored off-chip in order and
//                             returned on dram_data_in when dram_data_ready asks.
//   - high_u_cached_count, low_u_bypassed_count, current_line_u_tag :
//                             U-Tagging telemetry.
//   - early_exit_triggered, samples_executed : Early-exit telemetry; the
//                             inference stopped before S, after s_actual passes.
//   - eval_pending          : Early-exit decision pending for the sample that just
//                             finished; the host must not push the next layer's
//                             weights until it clears and busy is still high.
//
// Description:
//   Every layer runs as: ingress -> compute -> drain -> advance (Sec. III-A).
//   The input of a layer comes from DRAM, except for layer N-B+1 of samples
//   2..S, which is replayed from the on-chip IC buffer with a fresh MCD mask
//   (Sec. IV-B, Fig. 11(c)); layers 1..N-B are skipped for those samples.
//   Stage 2 runs in the background and fills its mask FIFO ahead of use
//   (Sec. IV-A). Weights for the PF filters are cached on-chip and reused for
//   every output pixel by recirculating the weight FIFO (Sec. III-B).
//   pool_step / pool_win_done are generated here from the PE output stream:
//   consecutive POOL_WIN_SIZE pixels form one pooling window.
//   IC_RAM_DEPTH / IC_ADDR_WIDTH size the Intermediate-layer Cache (default:
//   package macros); a smaller IC models a BRAM-constrained edge device.
//   Constraints: PF x PV == PC (a layer output word is the next layer's input
//   word) and the host must not push weights while a layer is computing.
//==============================================================================

`include "bcnn_pkg.vh"

module bcnn_top #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PC = `PC,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter ADDR_WIDTH = `ADDR_WIDTH,
    parameter DIM_WIDTH = `DIM_WIDTH,
    parameter TILE_CNT_WIDTH = `TILE_CNT_WIDTH,
    parameter KERNEL_DIM_WIDTH = `KERNEL_DIM_WIDTH,
    parameter STRIDE_WIDTH = `STRIDE_WIDTH,
    parameter LAYER_CNT_WIDTH = `LAYER_CNT_WIDTH,
    parameter SAMPLE_CNT_WIDTH = `SAMPLE_CNT_WIDTH,
    parameter QUANT_SCALE_WIDTH = `QUANT_SCALE_WIDTH,
    parameter QUANT_SHIFT_WIDTH = `QUANT_SHIFT_WIDTH,
    parameter QUANT_BIAS_WIDTH = `QUANT_BIAS_WIDTH,
    parameter POOL_MODE_WIDTH = `POOL_MODE_WIDTH,
    parameter POOL_CNT_WIDTH = `POOL_CNT_WIDTH,
    parameter VAR_OUT_WIDTH = 2*DATA_WIDTH,
    parameter IC_RAM_DEPTH = `IC_RAM_DEPTH,
    parameter IC_ADDR_WIDTH = `IC_ADDR_WIDTH
)(
    input wire clk, rst_n,

    // High-level control
    input wire start_inference,
    output wire busy,
    output wire inference_done,

    // Network & Bayesian hyperparameters
    input wire [LAYER_CNT_WIDTH-1:0] total_layers_N,
    input wire [LAYER_CNT_WIDTH-1:0] bayesian_layers_B,
    input wire [SAMPLE_CNT_WIDTH-1:0] total_samples_S,

    // Layer geometry & operator config (per layer, from host)
    input wire [DIM_WIDTH-1:0] H,
    input wire [DIM_WIDTH-1:0] W,
    input wire [TILE_CNT_WIDTH-1:0] C_tiles,
    input wire [TILE_CNT_WIDTH-1:0] W_tiles,
    input wire [DIM_WIDTH-1:0] L_frames,
    input wire [KERNEL_DIM_WIDTH-1:0] KH,
    input wire [KERNEL_DIM_WIDTH-1:0] KW,
    input wire [KERNEL_DIM_WIDTH-1:0] KL,
    input wire [STRIDE_WIDTH-1:0] stride,
    input wire mode_3d,
    input wire relu_en,
    input wire sc_en,
    input wire [POOL_MODE_WIDTH-1:0] pool_mode,
    input wire signed [QUANT_SCALE_WIDTH-1:0] quant_scale,
    input wire [QUANT_SHIFT_WIDTH-1:0] quant_shift,
    input wire signed [QUANT_BIAS_WIDTH-1:0] quant_bias,

    // DRAM data ingress (AXI-stream style)
    input wire dram_data_valid,
    output wire dram_data_ready,
    input wire [(PC*DATA_WIDTH)-1:0] dram_data_in,

    // DRAM weight ingress
    input wire weight_push,
    input wire [(PC*PF*DATA_WIDTH)-1:0] weight_din,
    output wire weight_full,

    // SC buffer input (ResNet skip connection)
    input wire [(PF*PV*DATA_WIDTH)-1:0] sc_features_in,

    // Stage 2 PRNG re-seeding
    input wire load_seed,
    input wire [`LFSR_WIDTH-1:0] seed_in,

    // UAMH Innovation 1: Uncertainty-Modulated Precision Storage
    input wire umps_en,
    input wire signed [`UMPS_THRESH_WIDTH-1:0] umps_thresh,

    // UAMH Innovation 3: Uncertainty-Tagged Cache Lines
    input wire utag_en,
    input wire signed [DATA_WIDTH-1:0] utag_zero_thresh,
    input wire [$clog2(PF*PV+1)-1:0] utag_high_count_th,

    // UAMH Innovation 2: Closed-Loop Early-Exit Sample Throttling
    input wire early_exit_en,
    input wire [`EARLY_EXIT_THRESH_WIDTH-1:0] early_exit_thresh,
    input wire [SAMPLE_CNT_WIDTH-1:0] early_exit_min_samples,

    // Bayesian reduction outputs (Stage 5)
    output wire [(PF*PV*DATA_WIDTH)-1:0] mean_prediction,
    output wire [(PF*PV*VAR_OUT_WIDTH)-1:0] uncertainty_score,
    output wire reduction_done,

    // Layer egress (to DRAM / inspection)
    output wire [(PF*PV*DATA_WIDTH)-1:0] layer_features_out,
    output wire layer_features_valid,
    output wire layer_done,

    // UMPS telemetry (compression ratio = ic_lines_cached x PF x PV / ic_bytes_used)
    output wire [IC_ADDR_WIDTH+1:0] ic_lines_cached,
    output wire [IC_ADDR_WIDTH+$clog2(PF*PV):0] ic_bytes_used,

    // U-Tagging spill egress and telemetry
    output wire [(PF*PV*DATA_WIDTH)-1:0] ic_spill_data,
    output wire ic_spill_valid,
    output wire [IC_ADDR_WIDTH+1:0] high_u_cached_count,
    output wire [IC_ADDR_WIDTH+1:0] low_u_bypassed_count,
    output wire [`UTAG_WIDTH-1:0] current_line_u_tag,

    // Early-exit telemetry
    output wire early_exit_triggered,
    output wire [SAMPLE_CNT_WIDTH-1:0] samples_executed,
    output wire eval_pending
);

    // Pipeline depth from a RAG read issue to the Stage 4 output:
    // BRAM(1) + MAC(1) + ACC(1) + QUANT(1) + RELU(1) + POOL(1) + DROPOUT(1)
    localparam PIPE_LATENCY = 7;
    localparam PIPE_DRAIN = PIPE_LATENCY + 1;
    localparam DRAIN_CNT_WIDTH = $clog2(PIPE_DRAIN + 1);
    localparam POOL_WIN_SIZE = 1 << POOL_CNT_WIDTH;

    // Controller states
    localparam ST_WIDTH = 3;
    localparam [ST_WIDTH-1:0] S_IDLE = 0;
    localparam [ST_WIDTH-1:0] S_LAYER = 1;
    localparam [ST_WIDTH-1:0] S_INGRESS = 2;
    localparam [ST_WIDTH-1:0] S_ARM = 3;
    localparam [ST_WIDTH-1:0] S_COMPUTE = 4;
    localparam [ST_WIDTH-1:0] S_DRAIN = 5;
    localparam [ST_WIDTH-1:0] S_ADVANCE = 6;

    reg [ST_WIDTH-1:0] state;
    reg [DRAIN_CNT_WIDTH-1:0] drain_cnt;
    reg start_ingress_r, start_compute_r, replay_load_r, layer_done_r;
    reg ping_pong_sel;
    reg src_replay;

    // Stage 1 wires
    wire s1_dram_ready, s1_ingress_done;
    wire s1_re_b_valid, s1_window_done, s1_layer_done;
    wire s1_read_issue, s1_last_window;
    wire [(PF*PV*PC*DATA_WIDTH)-1:0] pe_data_out;
    wire [(PV*PC*PF*DATA_WIDTH)-1:0] pe_weight_out;
    wire weight_empty;

    // Stage 2 wires
    wire [PF-1:0] sampler_mask_out;
    wire sampler_mask_valid;
    wire sampler_mask_pop;

    // Stage 3 wires
    wire [(PF*PV*DATA_WIDTH)-1:0] pe_features_out;
    wire pe_features_valid;

    // Stage 4 wires
    wire [(PF*PV*DATA_WIDTH)-1:0] stage4_features_out;
    wire stage4_valid_out;
    wire [(PF*PV*DATA_WIDTH)-1:0] premask_features;
    wire premask_valid;
    wire stage4_mask_pop;
    reg [POOL_CNT_WIDTH-1:0] pool_cnt;

    // Stage 5 wires
    wire s5_busy, s5_ic_write_en, s5_ic_read_en, s5_bypass, s5_mcd_en, s5_is_final;
    wire s5_eval_pending;
    wire [SAMPLE_CNT_WIDTH-1:0] s5_sample_idx;
    wire [LAYER_CNT_WIDTH-1:0] s5_layer_idx;
    wire [(PF*PV*DATA_WIDTH)-1:0] replay_features;
    wire replay_valid, replay_mask_pop;
    wire ic_full;

    //--------------------------------------------------------------------------
    // Layer-level Controller
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state <= S_IDLE;
            drain_cnt <= {DRAIN_CNT_WIDTH{1'b0}};
            start_ingress_r <= 1'b0;
            start_compute_r <= 1'b0;
            replay_load_r <= 1'b0;
            layer_done_r <= 1'b0;
            ping_pong_sel <= 1'b0;
            src_replay <= 1'b0;
        end else begin
            start_ingress_r <= 1'b0;
            start_compute_r <= 1'b0;
            replay_load_r <= 1'b0;
            layer_done_r <= 1'b0;

            case(state)
                S_IDLE: begin
                    // Stage 5 latches the configuration on the same edge
                    if(start_inference && !s5_busy) begin
                        state <= S_LAYER;
                    end
                end
                S_LAYER: begin
                    if(!s5_busy) begin
                        state <= S_IDLE;
                    end else if(!s5_eval_pending && (!s5_ic_read_en || sampler_mask_valid)) begin
                        // An early-exit decision pending blocks the next sample's layer
                        // Layer N-B+1 of samples 2..S is fed from the IC buffer
                        src_replay <= s5_ic_read_en;
                        replay_load_r <= s5_ic_read_en;
                        start_ingress_r <= 1'b1;
                        state <= S_INGRESS;
                    end
                end
                S_INGRESS: begin
                    if(s1_ingress_done) begin
                        state <= S_ARM;
                    end
                end
                S_ARM: begin
                    // Need a mask in the FIFO before an MCD layer starts computing
                    if(!s5_mcd_en || sampler_mask_valid) begin
                        ping_pong_sel <= ~ping_pong_sel; // read the bank just written
                        start_compute_r <= 1'b1;
                        state <= S_COMPUTE;
                    end
                end
                S_COMPUTE: begin
                    if(s1_layer_done) begin
                        drain_cnt <= {DRAIN_CNT_WIDTH{1'b0}};
                        state <= S_DRAIN;
                    end
                end
                S_DRAIN: begin
                    // Let the last window reach Stage 4/5 before advancing the layer
                    if(drain_cnt == PIPE_DRAIN-1) begin
                        layer_done_r <= 1'b1;
                        state <= S_ADVANCE;
                    end else begin
                        drain_cnt <= drain_cnt + 1'b1;
                    end
                end
                S_ADVANCE: begin
                    // Stage 5 updates its layer/sample on this edge
                    state <= S_LAYER;
                end
                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

    assign busy = (state != S_IDLE);
    assign eval_pending = s5_eval_pending;
    assign layer_done = layer_done_r;

    //--------------------------------------------------------------------------
    // Ingress source: DRAM, or the IC replay stream for layer N-B+1 (samples 2..S)
    //--------------------------------------------------------------------------
    wire ing_valid = src_replay ? replay_valid : dram_data_valid;
    wire [(PC*DATA_WIDTH)-1:0] ing_data = src_replay ? replay_features[(PC*DATA_WIDTH)-1:0] : dram_data_in;
    wire ic_rd_req = s1_dram_ready && src_replay;

    // In a replay layer the DRAM bus only carries spilled lines back to Stage 5
    wire s5_spill_req;
    assign dram_data_ready = src_replay ? s5_spill_req : s1_dram_ready;

    //--------------------------------------------------------------------------
    // Weight reuse: one window of PF-filter weights is cached in the FIFO and
    // recirculated for every output pixel, then drained by the last window.
    // A single-step window keeps its word in the registered FIFO output.
    //--------------------------------------------------------------------------
    wire single_step_window = (KH == 1) && (KW == 1) && (C_tiles == 1) && (!mode_3d || (KL == 1));
    reg weight_first, weight_pop_d1, last_window_d1;

    wire weight_pop = s1_read_issue && (!single_step_window || weight_first);
    wire recirc_push = weight_pop_d1 && !last_window_d1;
    wire wb_push = recirc_push || weight_push;
    wire [(PC*PF*DATA_WIDTH)-1:0] wb_din = recirc_push ? pe_weight_out[(PC*PF*DATA_WIDTH)-1:0] : weight_din;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            weight_first <= 1'b0;
            weight_pop_d1 <= 1'b0;
            last_window_d1 <= 1'b0;
        end else begin
            if(start_compute_r) begin
                weight_first <= 1'b1;
            end else if(s1_read_issue) begin
                weight_first <= 1'b0;
            end
            weight_pop_d1 <= weight_pop && !single_step_window;
            last_window_d1 <= s1_last_window;
        end
    end

    //--------------------------------------------------------------------------
    // Pooling window sequencer (consecutive PE output pixels of one layer)
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            pool_cnt <= {POOL_CNT_WIDTH{1'b0}};
        end else if(start_compute_r) begin
            pool_cnt <= {POOL_CNT_WIDTH{1'b0}};
        end else if(pe_features_valid) begin
            pool_cnt <= pool_cnt + 1'b1;
        end
    end

    wire pool_win_done = (pool_cnt == POOL_WIN_SIZE-1);

    // Stage 2 FIFO pops: replay dropout during replay ingress, Stage 4 otherwise
    assign sampler_mask_pop = (s5_ic_read_en && (state != S_COMPUTE)) ? replay_mask_pop : stage4_mask_pop;

    //--------------------------------------------------------------------------
    // STAGE 1: Smart Data Buffer & Smart Weight Buffer
    //--------------------------------------------------------------------------
    smart_data_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DIM_WIDTH(DIM_WIDTH),
        .TILE_CNT_WIDTH(TILE_CNT_WIDTH),
        .KERNEL_DIM_WIDTH(KERNEL_DIM_WIDTH),
        .STRIDE_WIDTH(STRIDE_WIDTH),
        .PC(PC),
        .PV(PV),
        .PF(PF)
    ) u_stage1_data (
        .clk(clk),
        .rst_n(rst_n),
        .ping_pong_sel(ping_pong_sel),
        .H(H),
        .W(W),
        .L_frames(L_frames),
        .C_tiles(C_tiles),
        .W_tiles(W_tiles),
        .KH(KH),
        .KW(KW),
        .KL(KL),
        .stride(stride),
        .mode_3d(mode_3d),
        .start_ingress(start_ingress_r),
        .dram_valid(ing_valid),
        .dram_data_in(ing_data),
        .dram_ready(s1_dram_ready),
        .ingress_done(s1_ingress_done),
        .start_compute(start_compute_r),
        .re_b_valid(s1_re_b_valid),
        .window_done(s1_window_done),
        .layer_done(s1_layer_done),
        .read_issue(s1_read_issue),
        .last_window(s1_last_window),
        .pe_data_out(pe_data_out)
    );

    smart_weight_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .PC(PC),
        .PF(PF),
        .PV(PV)
    ) u_stage1_weight (
        .clk(clk),
        .rst_n(rst_n),
        .weight_push(wb_push),
        .weight_din(wb_din),
        .weight_full(weight_full),
        .weight_pop(weight_pop),
        .pe_weight_out(pe_weight_out),
        .weight_empty(weight_empty)
    );

    //--------------------------------------------------------------------------
    // STAGE 2: Bernoulli Sampler (runs in the background, pauses when full)
    //--------------------------------------------------------------------------
    bernoulli_sampler #(
        .PF(PF),
        .LFSR_WIDTH(`LFSR_WIDTH),
        .N_LFSR(`N_LFSR),
        .FIFO_DEPTH(`MASK_FIFO_DEPTH),
        .FIFO_ADDR(`MASK_FIFO_ADDR)
    ) u_stage2 (
        .clk(clk),
        .rst_n(rst_n),
        .sampler_en(1'b1),
        .load_seed(load_seed),
        .seed_in(seed_in),
        .mask_pop(sampler_mask_pop),
        .mask_out(sampler_mask_out),
        .mask_valid(sampler_mask_valid),
        .mask_empty(),
        .mask_full(),
        .mask_count()
    );

    //--------------------------------------------------------------------------
    // STAGE 3: Processing Engine
    //--------------------------------------------------------------------------
    processing_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .PC(PC),
        .PF(PF),
        .PV(PV),
        .QUANT_SCALE_WIDTH(QUANT_SCALE_WIDTH),
        .QUANT_SHIFT_WIDTH(QUANT_SHIFT_WIDTH),
        .QUANT_BIAS_WIDTH(QUANT_BIAS_WIDTH)
    ) u_stage3 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(s1_re_b_valid),
        .window_done(s1_window_done),
        .relu_en(relu_en),
        .pe_data_in(pe_data_out),
        .pe_weight_in(pe_weight_out),
        .quant_scale(quant_scale),
        .quant_shift(quant_shift),
        .quant_bias(quant_bias),
        .pe_features_out(pe_features_out),
        .features_valid(pe_features_valid)
    );

    //--------------------------------------------------------------------------
    // STAGE 4: Functional Engine (SC -> Pool -> Dropout)
    //--------------------------------------------------------------------------
    functional_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .PC(PC),
        .PF(PF),
        .PV(PV),
        .POOL_MODE_WIDTH(POOL_MODE_WIDTH),
        .POOL_CNT_WIDTH(POOL_CNT_WIDTH)
    ) u_stage4 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(pe_features_valid),
        .sc_en(sc_en),
        .pool_mode(pool_mode),
        .pool_win_done(pool_win_done),
        .pool_step(pool_cnt),
        .mcd_en(s5_mcd_en),
        .conv_features_in(pe_features_out),
        .sc_features_in(sc_features_in),
        .mask_in(sampler_mask_out),
        .mask_valid(sampler_mask_valid),
        .mask_load(start_compute_r),
        .mask_pop(stage4_mask_pop),
        .stage4_features_out(stage4_features_out),
        .stage4_valid_out(stage4_valid_out),
        .premask_features_out(premask_features),
        .premask_valid_out(premask_valid)
    );

    assign layer_features_out = stage4_features_out;
    assign layer_features_valid = stage4_valid_out;

    //--------------------------------------------------------------------------
    // STAGE 5: Cache & Reduction Engine
    //--------------------------------------------------------------------------
    cache_reduction_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV),
        .LAYER_CNT_WIDTH(LAYER_CNT_WIDTH),
        .SAMPLE_CNT_WIDTH(SAMPLE_CNT_WIDTH),
        .IC_RAM_DEPTH(IC_RAM_DEPTH),
        .IC_ADDR_WIDTH(IC_ADDR_WIDTH),
        .VAR_OUT_WIDTH(VAR_OUT_WIDTH)
    ) u_stage5 (
        .clk(clk),
        .rst_n(rst_n),
        .start_inference(start_inference),
        .layer_done(layer_done_r),
        .total_layers_N(total_layers_N),
        .bayesian_layers_B(bayesian_layers_B),
        .total_samples_S(total_samples_S),
        .premask_features_in(premask_features),
        .premask_valid_in(premask_valid),
        .stage4_features_in(stage4_features_out),
        .stage4_valid_in(stage4_valid_out),
        .ic_rd_req(ic_rd_req),
        .replay_mask_load(replay_load_r),
        .mask_in(sampler_mask_out),
        .mask_valid(sampler_mask_valid),
        .replay_mask_pop(replay_mask_pop),
        .replay_features(replay_features),
        .replay_valid(replay_valid),
        .ic_word_count(ic_lines_cached),
        .ic_byte_count(ic_bytes_used),
        .ic_full(ic_full),
        .umps_en(umps_en),
        .umps_thresh(umps_thresh),
        .utag_en(utag_en),
        .utag_zero_thresh(utag_zero_thresh),
        .utag_high_count_th(utag_high_count_th),
        .spill_features(ic_spill_data),
        .spill_valid(ic_spill_valid),
        .spill_req(s5_spill_req),
        .spill_ret_features(dram_data_in[(PF*PV*DATA_WIDTH)-1:0]),
        .spill_ret_valid(src_replay && dram_data_valid),
        .high_u_cached_count(high_u_cached_count),
        .low_u_bypassed_count(low_u_bypassed_count),
        .current_line_u_tag(current_line_u_tag),
        .early_exit_en(early_exit_en),
        .early_exit_thresh(early_exit_thresh),
        .early_exit_min_samples(early_exit_min_samples),
        .eval_pending(s5_eval_pending),
        .early_exit_triggered(early_exit_triggered),
        .samples_executed(samples_executed),
        .sample_idx(s5_sample_idx),
        .layer_idx(s5_layer_idx),
        .busy(s5_busy),
        .ic_write_en(s5_ic_write_en),
        .ic_read_en(s5_ic_read_en),
        .bypass_feature_extractor(s5_bypass),
        .mcd_en(s5_mcd_en),
        .is_final_layer(s5_is_final),
        .inference_done(inference_done),
        .mean_prediction(mean_prediction),
        .uncertainty_score(uncertainty_score),
        .reduction_done(reduction_done)
    );

endmodule
