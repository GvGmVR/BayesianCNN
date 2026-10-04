//==============================================================================
// Module: cache_reduction_engine.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 5 - Intermediate Caching & Output Reduction
//------------------------------------------------------------------------------
// Purpose:
//   Top-level wrapper for Stage 5. Integrates the MC sample controller, the
//   Intermediate-layer Caching (IC) buffer with its replay path, and the
//   output reducer (Section IV-B, Fig. 11(c) and Eq. 1 in Fan et al.).
//
// Architectural Inputs:
//   - clk, rst_n          : Clock (220 MHz) and active-low reset.
//   - start_inference     : 1-cycle strobe to begin an S-sample inference.
//   - layer_done          : 1-cycle strobe when the current layer finishes.
//   - total_layers_N, bayesian_layers_B, total_samples_S : Configuration.
//   - premask_features_in : Stage 4 pre-dropout tap (functional_engine premask_features_out).
//   - premask_valid_in    : Valid strobe for premask_features_in.
//   - stage4_features_in  : Stage 4 final output (functional_engine stage4_features_out).
//   - stage4_valid_in     : Valid strobe for stage4_features_in.
//   - ic_rd_req           : Stage 1 ingress requests the next cached word.
//   - replay_mask_load    : 1-cycle strobe to latch a fresh mask for the replayed tile.
//   - mask_in, mask_valid : Bernoulli mask word and valid from the Stage 2 FIFO.
//   - umps_en             : UMPS enable (UAMH Innovation 1), latched at inference start.
//   - umps_thresh         : UMPS activity threshold tau.
//   - utag_en             : U-Tagging enable (UAMH Innovation 3), latched at inference start.
//   - utag_zero_thresh    : |val| at or below which a channel counts as inactive.
//   - utag_high_count_th  : INT8-precision channel count that makes a line UTAG_HIGH.
//   - spill_ret_features, spill_ret_valid : Spilled lines returned from DRAM
//                           during replay, in spill order.
//   - early_exit_en, early_exit_thresh, early_exit_min_samples : UAMH
//                           Innovation 2 enable, tolerance epsilon and warm-up S_min.
//
// Architectural Outputs:
//   - replay_mask_pop     : Pop strobe to the Stage 2 FIFO (replay path).
//   - replay_features     : Cached layer N-B output with this sample's MCD mask applied.
//   - replay_valid        : 1-cycle strobe indicating replay_features is valid.
//   - ic_word_count       : Number of lines (pixels) cached for layer N-B.
//   - ic_byte_count       : Bytes of IC data storage used (UMPS telemetry).
//   - ic_full             : IC buffer is full (baseline drops, U-Tagging spills).
//   - spill_features, spill_valid : Lines not admitted on-chip, unmasked INT8,
//                           to be stored off-chip in order (U-Tagging only).
//   - spill_req           : Replay needs the next spilled line from DRAM.
//   - high_u_cached_count, low_u_bypassed_count, current_line_u_tag :
//                           U-Tagging telemetry.
//   - eval_pending        : Early-exit decision pending; do not start the next layer.
//   - early_exit_triggered, samples_executed : Early-exit telemetry (s_actual).
//   - sample_idx, layer_idx, busy, ic_write_en, ic_read_en,
//     bypass_feature_extractor, mcd_en, is_final_layer, inference_done :
//                           Controller status and routing control.
//   - mean_prediction     : PF x PV INT8 predictive means.
//   - uncertainty_score   : PF x PV per-channel variances.
//   - reduction_done      : Reduction results are valid.
//
// Description:
//   Sample 1: the pre-dropout output of layer N-B is written sequentially into
//   the IC buffer while Stage 4 still applies sample 1's mask on its own path.
//   Samples 2..S: layers 1..N-B are skipped. Stage 1 pulls the cached words in
//   the same order they were written and a local dropout_engine applies a
//   freshly popped mask, i.e. "applying MCD on the cached data" (Sec. IV-B).
//   Caching before the mask is what keeps every replayed sample independent.
//   Layer N: every sample's Stage 4 output is accumulated by the reducer.
//   UMPS (umps_en = 1): the cached pixel passes variance_analyzer -> umps_packer
//   (2 cycles) so low-activity channel pairs are stored as two INT4 nibbles in
//   one byte, and umps_unpacker (1 cycle) restores INT8 before replay dropout.
//   With umps_en = 0 both are bypassed and timing is identical to the baseline.
//   U-Tagging (utag_en = 1): uncertainty_tagger tags each line in parallel with
//   the packer and u_tag_manager decides admission. A line that is not admitted
//   is spilled to DRAM unmasked, and a 1-bit-per-line directory records where
//   every line went. On replay the lines are walked in original order: resident
//   lines are read from the IC, spilled lines are requested back over the DRAM
//   bus, and both pass through the replay dropout so each sample is re-masked.
//   With utag_en = 0 every line is admitted while not full, nothing is spilled,
//   and the replay path is the baseline one.
//   Early exit (early_exit_en = 1): after layer N of each sample the controller
//   waits while output_reducer recomputes the running mean / variance and its
//   convergence_monitor decides whether the remaining samples can be skipped.
//==============================================================================

`include "bcnn_pkg.vh"

module cache_reduction_engine #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter LAYER_CNT_WIDTH = `LAYER_CNT_WIDTH,
    parameter SAMPLE_CNT_WIDTH = `SAMPLE_CNT_WIDTH,
    parameter IC_RAM_DEPTH = `IC_RAM_DEPTH,
    parameter IC_ADDR_WIDTH = `IC_ADDR_WIDTH,
    parameter REDUCER_ACCUM_WIDTH = `REDUCER_ACCUM_WIDTH,
    parameter VAR_ACCUM_WIDTH = `VAR_ACCUM_WIDTH,
    parameter VAR_OUT_WIDTH = 2*DATA_WIDTH
)(
    input wire clk, rst_n, start_inference, layer_done,

    // Configuration registers
    input wire [LAYER_CNT_WIDTH-1:0] total_layers_N,
    input wire [LAYER_CNT_WIDTH-1:0] bayesian_layers_B,
    input wire [SAMPLE_CNT_WIDTH-1:0] total_samples_S,

    // Stage 4 interface
    input wire [(PF*PV*DATA_WIDTH)-1:0] premask_features_in,
    input wire premask_valid_in,
    input wire [(PF*PV*DATA_WIDTH)-1:0] stage4_features_in,
    input wire stage4_valid_in,

    // IC replay interface (Stage 1 ingress / Stage 2 sampler)
    input wire ic_rd_req,
    input wire replay_mask_load,
    input wire [PF-1:0] mask_in,
    input wire mask_valid,
    output wire replay_mask_pop,
    output wire [(PF*PV*DATA_WIDTH)-1:0] replay_features,
    output wire replay_valid,
    output wire [IC_ADDR_WIDTH+1:0] ic_word_count,
    output wire [IC_ADDR_WIDTH+$clog2(PF*PV):0] ic_byte_count,
    output wire ic_full,

    // UMPS control
    input wire umps_en,
    input wire signed [`UMPS_THRESH_WIDTH-1:0] umps_thresh,

    // U-Tagging control, spill path and telemetry
    input wire utag_en,
    input wire signed [DATA_WIDTH-1:0] utag_zero_thresh,
    input wire [$clog2(PF*PV+1)-1:0] utag_high_count_th,
    output wire [(PF*PV*DATA_WIDTH)-1:0] spill_features,
    output wire spill_valid,
    output wire spill_req,
    input wire [(PF*PV*DATA_WIDTH)-1:0] spill_ret_features,
    input wire spill_ret_valid,
    output wire [IC_ADDR_WIDTH+1:0] high_u_cached_count,
    output wire [IC_ADDR_WIDTH+1:0] low_u_bypassed_count,
    output wire [`UTAG_WIDTH-1:0] current_line_u_tag,

    // Early-exit control and telemetry
    input wire early_exit_en,
    input wire [`EARLY_EXIT_THRESH_WIDTH-1:0] early_exit_thresh,
    input wire [SAMPLE_CNT_WIDTH-1:0] early_exit_min_samples,
    output wire eval_pending,
    output wire early_exit_triggered,
    output wire [SAMPLE_CNT_WIDTH-1:0] samples_executed,

    // Controller status / routing control
    output wire [SAMPLE_CNT_WIDTH-1:0] sample_idx,
    output wire [LAYER_CNT_WIDTH-1:0] layer_idx,
    output wire busy,
    output wire ic_write_en,
    output wire ic_read_en,
    output wire bypass_feature_extractor,
    output wire mcd_en,
    output wire is_final_layer,
    output wire inference_done,

    // Reduction results
    output wire [(PF*PV*DATA_WIDTH)-1:0] mean_prediction,
    output wire [(PF*PV*VAR_OUT_WIDTH)-1:0] uncertainty_score,
    output wire reduction_done
);

    wire run_start;
    wire [SAMPLE_CNT_WIDTH-1:0] num_samples;

    // Early exit: controller <-> reducer handshake
    wire ee_active;
    wire red_decided;
    wire red_exit;

    localparam LEN_WIDTH = $clog2(PF*PV+1);
    localparam [LEN_WIDTH-1:0] FULL_LEN = PF*PV;
    localparam OCC_WIDTH = IC_ADDR_WIDTH+$clog2(PF*PV)+1;
    localparam [OCC_WIDTH-1:0] IC_CAPACITY = IC_RAM_DEPTH*PF*PV;

    // Line directory: a replayed map must fit Stage 1's data buffer, so RAM_DEPTH lines
    localparam DIR_DEPTH = `RAM_DEPTH;
    localparam DIR_IDX_W = $clog2(DIR_DEPTH+1);
    localparam INFLIGHT_W = 2; // IC reads in flight: BRAM (1) + unpacker (1)

    // UMPS / U-Tagging modes are fixed for a whole inference so every line is
    // decoded and replayed the way it was written
    reg umps_on, utag_on;
    wire enhanced = umps_on || utag_on;

    // Write path: baseline tap, or analyzer -> packer / tagger when a UAMH mode is on
    wire cache_in = ic_write_en && premask_valid_in;
    wire an_valid;
    wire [(PF*PV)-1:0] an_low_var;
    wire [(PF*PV*DATA_WIDTH)-1:0] an_features;
    wire pk_valid;
    wire [(PF*PV*DATA_WIDTH)-1:0] pk_features;
    wire [(PF*PV)-1:0] pk_mask;
    wire [LEN_WIDTH-1:0] pk_len;
    wire tg_valid;
    wire [`UTAG_WIDTH-1:0] tg_tag;
    reg [(PF*PV*DATA_WIDTH)-1:0] spill_feat_q;

    wire buf_wr_en = enhanced ? pk_valid : cache_in;
    wire [(PF*PV*DATA_WIDTH)-1:0] buf_wr_data = enhanced ? pk_features : premask_features_in;
    wire [LEN_WIDTH-1:0] buf_wr_len = enhanced ? pk_len : FULL_LEN;
    wire [(PF*PV)-1:0] buf_wr_mask = enhanced ? pk_mask : {(PF*PV){1'b0}};
    wire [`UTAG_WIDTH-1:0] buf_wr_tag = enhanced ? tg_tag : `UTAG_ZERO;
    wire mgr_admit, mgr_spill;
    wire ic_wr = mgr_admit;

    assign spill_valid = mgr_spill;
    assign spill_features = spill_feat_q;

    // Line directory and replay sequencer (U-Tagging only)
    reg dir_mem [0:DIR_DEPTH-1];
    reg [DIR_IDX_W-1:0] src_lines;
    reg [DIR_IDX_W-1:0] rp_line;
    reg [INFLIGHT_W-1:0] ic_inflight;
    reg spill_pending;

    wire rp_more = rp_line < src_lines;
    wire rp_in_ic = dir_mem[rp_line[DIR_IDX_W-2:0]];
    wire seq_go = ic_read_en && ic_rd_req && rp_more && !spill_pending;
    wire seq_ic_rd = seq_go && rp_in_ic;
    wire seq_spill = seq_go && !rp_in_ic && (ic_inflight == {INFLIGHT_W{1'b0}});
    wire spill_ret_accept = spill_pending && spill_ret_valid;

    assign spill_req = spill_pending;

    // Read path: raw line, or unpacked line when UMPS is on
    wire ic_rd = utag_on ? seq_ic_rd : (ic_read_en && ic_rd_req);
    wire [(PF*PV*DATA_WIDTH)-1:0] buf_rd_data;
    wire [(PF*PV)-1:0] buf_rd_mask;
    wire buf_rd_valid;
    wire [(PF*PV*DATA_WIDTH)-1:0] unp_features;
    wire unp_valid;

    wire [(PF*PV*DATA_WIDTH)-1:0] ic_rd_data = umps_on ? unp_features : buf_rd_data;
    wire ic_rd_valid = umps_on ? unp_valid : buf_rd_valid;

    // Replay stream in original line order: IC line or line returned from DRAM
    wire rp_valid = ic_rd_valid || spill_ret_accept;
    wire [(PF*PV*DATA_WIDTH)-1:0] rp_features = spill_ret_accept ? spill_ret_features : ic_rd_data;

    // 1. MC Sample Controller
    mc_sample_controller #(
        .LAYER_CNT_WIDTH(LAYER_CNT_WIDTH),
        .SAMPLE_CNT_WIDTH(SAMPLE_CNT_WIDTH)
    ) u_mc_ctrl (
        .clk(clk),
        .rst_n(rst_n),
        .start_inference(start_inference),
        .layer_done(layer_done),
        .total_layers_N(total_layers_N),
        .bayesian_layers_B(bayesian_layers_B),
        .total_samples_S(total_samples_S),
        .sample_idx(sample_idx),
        .layer_idx(layer_idx),
        .num_samples(num_samples),
        .busy(busy),
        .run_start(run_start),
        .ic_write_en(ic_write_en),
        .ic_read_en(ic_read_en),
        .bypass_feature_extractor(bypass_feature_extractor),
        .mcd_en(mcd_en),
        .is_final_layer(is_final_layer),
        .inference_done(inference_done),
        .early_exit_en(early_exit_en),
        .eval_done(red_decided),
        .early_exit_trigger(red_exit),
        .early_exit_active(ee_active),
        .eval_pending(eval_pending),
        .samples_executed(samples_executed),
        .early_exit_triggered(early_exit_triggered)
    );

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            umps_on <= 1'b0;
            utag_on <= 1'b0;
        end else if(run_start) begin
            umps_on <= umps_en;
            utag_on <= utag_en;
        end
    end

    // 2. UMPS write path: channel classification and INT4 pair packing
    variance_analyzer #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV)
    ) u_variance_analyzer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(enhanced && cache_in),
        .features_in(premask_features_in),
        .thresh_in(umps_thresh),
        .is_low_var(an_low_var),
        .features_out(an_features),
        .valid_out(an_valid)
    );

    umps_packer #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV)
    ) u_umps_packer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(an_valid),
        .umps_en(umps_on),
        .is_low_var(an_low_var),
        .features_in(an_features),
        .packed_features_out(pk_features),
        .pack_mask_out(pk_mask),
        .packed_len_out(pk_len),
        .valid_out(pk_valid)
    );

    // 3. U-Tagging: line tag (aligned with the packer) and admission policy
    uncertainty_tagger #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV)
    ) u_uncertainty_tagger (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(an_valid),
        .features_in(an_features),
        .is_low_var(an_low_var),
        .zero_thresh(utag_zero_thresh),
        .high_count_thresh(utag_high_count_th),
        .u_tag_out(tg_tag),
        .valid_out(tg_valid)
    );

    // Unmasked INT8 copy of the line, aligned with the packer, for spilling
    always @(posedge clk) begin
        if(an_valid) begin
            spill_feat_q <= an_features;
        end
    end

    u_tag_manager #(
        .OCC_WIDTH(OCC_WIDTH),
        .CNT_WIDTH(IC_ADDR_WIDTH+2)
    ) u_tag_mgr (
        .clk(clk),
        .rst_n(rst_n),
        .clear(run_start),
        .utag_en(utag_on),
        .line_valid_in(buf_wr_en),
        .u_tag_in(buf_wr_tag),
        .ic_occupancy(ic_byte_count),
        .ic_capacity(IC_CAPACITY),
        .ic_full(ic_full),
        .admit_to_bram(mgr_admit),
        .spill_to_dram(mgr_spill),
        .high_u_cached_count(high_u_cached_count),
        .low_u_bypassed_count(low_u_bypassed_count)
    );

    // Directory write (where each line went) and replay sequencing
    always @(posedge clk) begin
        if(utag_on && (mgr_admit || mgr_spill) && (src_lines < DIR_DEPTH)) begin
            dir_mem[src_lines[DIR_IDX_W-2:0]] <= mgr_admit;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            src_lines <= {DIR_IDX_W{1'b0}};
            rp_line <= {DIR_IDX_W{1'b0}};
            ic_inflight <= {INFLIGHT_W{1'b0}};
            spill_pending <= 1'b0;
        end else begin
            if(run_start) begin
                src_lines <= {DIR_IDX_W{1'b0}};
            end else if(utag_on && (mgr_admit || mgr_spill) && (src_lines < DIR_DEPTH)) begin
                src_lines <= src_lines + 1'b1;
            end

            if(run_start || layer_done) begin
                rp_line <= {DIR_IDX_W{1'b0}};
            end else if(seq_ic_rd || seq_spill) begin
                rp_line <= rp_line + 1'b1;
            end

            // Spilled lines are only requested once every earlier IC read has landed
            if(run_start) begin
                ic_inflight <= {INFLIGHT_W{1'b0}};
            end else if(utag_on) begin
                ic_inflight <= ic_inflight + seq_ic_rd - ic_rd_valid;
            end

            if(run_start) begin
                spill_pending <= 1'b0;
            end else if(seq_spill) begin
                spill_pending <= 1'b1;
            end else if(spill_ret_accept) begin
                spill_pending <= 1'b0;
            end
        end
    end

    // 4. Intermediate-layer Cache (variable-length lines, replay restarts every layer)
    ic_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV),
        .IC_RAM_DEPTH(IC_RAM_DEPTH),
        .IC_ADDR_WIDTH(IC_ADDR_WIDTH)
    ) u_ic_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .clear(run_start),
        .rd_rewind(layer_done),
        .wr_en(ic_wr),
        .wr_data(buf_wr_data),
        .wr_len(buf_wr_len),
        .wr_mask(buf_wr_mask),
        .wr_u_tag(buf_wr_tag),
        .rd_en(ic_rd),
        .rd_data(buf_rd_data),
        .rd_mask(buf_rd_mask),
        .rd_u_tag(current_line_u_tag),
        .rd_valid(buf_rd_valid),
        .line_count(ic_word_count),
        .byte_count(ic_byte_count),
        .full(ic_full)
    );

    // 5. UMPS read path: INT4 nibbles sign-extended back to INT8
    umps_unpacker #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV)
    ) u_umps_unpacker (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(umps_on && buf_rd_valid),
        .pack_mask_in(buf_rd_mask),
        .packed_features_in(buf_rd_data),
        .unpacked_features_out(unp_features),
        .valid_out(unp_valid)
    );

    // 6. Replay dropout - fresh filter-wise MCD mask on the cached layer N-B output
    dropout_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV)
    ) u_replay_dropout (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(rp_valid),
        .mcd_en(ic_read_en),
        .features_in(rp_features),
        .mask_in(mask_in),
        .mask_valid(mask_valid),
        .mask_load(replay_mask_load),
        .mask_pop(replay_mask_pop),
        .masked_features(replay_features),
        .valid_out(replay_valid)
    );

    // 7. Output Reducer (layer N of every sample)
    output_reducer #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV),
        .SAMPLE_CNT_WIDTH(SAMPLE_CNT_WIDTH),
        .REDUCER_ACCUM_WIDTH(REDUCER_ACCUM_WIDTH),
        .VAR_ACCUM_WIDTH(VAR_ACCUM_WIDTH),
        .VAR_OUT_WIDTH(VAR_OUT_WIDTH)
    ) u_reducer (
        .clk(clk),
        .rst_n(rst_n),
        .clear(run_start),
        .sample_valid(is_final_layer && stage4_valid_in),
        .sample_in(stage4_features_in),
        .sample_idx(sample_idx),
        .total_samples_S(num_samples),
        .early_exit_en(ee_active),
        .early_exit_thresh(early_exit_thresh),
        .early_exit_min_samples(early_exit_min_samples),
        .mean_prediction(mean_prediction),
        .uncertainty_score(uncertainty_score),
        .reduction_done(reduction_done),
        .sample_decided(red_decided),
        .early_exit_trigger(red_exit),
        .variance_delta()
    );

endmodule
