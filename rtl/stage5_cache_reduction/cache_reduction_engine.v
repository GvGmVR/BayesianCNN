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
//
// Architectural Outputs:
//   - replay_mask_pop     : Pop strobe to the Stage 2 FIFO (replay path).
//   - replay_features     : Cached layer N-B output with this sample's MCD mask applied.
//   - replay_valid        : 1-cycle strobe indicating replay_features is valid.
//   - ic_word_count       : Number of words cached for layer N-B.
//   - ic_full             : IC buffer is full; further writes are dropped.
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
    output wire [IC_ADDR_WIDTH:0] ic_word_count,
    output wire ic_full,

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

    // Write pointer doubles as the cached word count; read pointer restarts every layer
    reg [IC_ADDR_WIDTH:0] wr_ptr;
    reg [IC_ADDR_WIDTH:0] rd_ptr;

    wire ic_wr = ic_write_en && premask_valid_in && !ic_full;
    wire ic_rd = ic_read_en && ic_rd_req && (rd_ptr < wr_ptr);

    wire [(PF*PV*DATA_WIDTH)-1:0] ic_rd_data;
    wire ic_rd_valid;

    assign ic_word_count = wr_ptr;
    assign ic_full = (wr_ptr == IC_RAM_DEPTH);

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
        .inference_done(inference_done)
    );

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            wr_ptr <= {(IC_ADDR_WIDTH+1){1'b0}};
            rd_ptr <= {(IC_ADDR_WIDTH+1){1'b0}};
        end else begin
            if(run_start) begin
                wr_ptr <= {(IC_ADDR_WIDTH+1){1'b0}};
            end else if(ic_wr) begin
                wr_ptr <= wr_ptr + 1'b1;
            end

            if(run_start || layer_done) begin
                rd_ptr <= {(IC_ADDR_WIDTH+1){1'b0}};
            end else if(ic_rd) begin
                rd_ptr <= rd_ptr + 1'b1;
            end
        end
    end

    // 2. Intermediate-layer Cache
    ic_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV),
        .IC_RAM_DEPTH(IC_RAM_DEPTH),
        .IC_ADDR_WIDTH(IC_ADDR_WIDTH)
    ) u_ic_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(ic_wr),
        .wr_addr(wr_ptr[IC_ADDR_WIDTH-1:0]),
        .wr_data(premask_features_in),
        .rd_en(ic_rd),
        .rd_addr(rd_ptr[IC_ADDR_WIDTH-1:0]),
        .rd_data(ic_rd_data),
        .rd_valid(ic_rd_valid)
    );

    // 3. Replay dropout - fresh filter-wise MCD mask on the cached layer N-B output
    dropout_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .PF(PF),
        .PV(PV)
    ) u_replay_dropout (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(ic_rd_valid),
        .mcd_en(ic_read_en),
        .features_in(ic_rd_data),
        .mask_in(mask_in),
        .mask_valid(mask_valid),
        .mask_load(replay_mask_load),
        .mask_pop(replay_mask_pop),
        .masked_features(replay_features),
        .valid_out(replay_valid)
    );

    // 4. Output Reducer (layer N of every sample)
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
        .mean_prediction(mean_prediction),
        .uncertainty_score(uncertainty_score),
        .reduction_done(reduction_done)
    );

endmodule
