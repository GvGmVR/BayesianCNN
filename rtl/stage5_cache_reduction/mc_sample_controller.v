//==============================================================================
// Module: mc_sample_controller.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 5 - Intermediate Caching & Output Reduction
//------------------------------------------------------------------------------
// Purpose:
//   Sequences the layer-by-layer, sample-by-sample execution of a partially
//   Bayesian CNN with N layers, of which the last B are Bayesian, over S Monte
//   Carlo samples using Intermediate-layer Caching (Section IV-B in Fan et al.).
//
// Architectural Inputs:
//   - clk, rst_n         : Clock (220 MHz) and active-low reset.
//   - start_inference    : 1-cycle strobe to begin a new S-sample inference.
//   - layer_done         : 1-cycle strobe from Stage 1/4 when current layer finishes.
//   - total_layers_N     : Number of layers N (must be >= 1).
//   - bayesian_layers_B  : Number of Bayesian layers B (clamped to N).
//   - total_samples_S    : Number of MC samples S (0 is treated as 1).
//   - early_exit_en      : UAMH Innovation 2 enable, latched at inference start.
//   - eval_done          : output_reducer's convergence decision for the current
//                          sample is ready.
//   - early_exit_trigger : That decision is "converged" (valid with eval_done).
//
// Architectural Outputs:
//   - sample_idx         : Current MC sample, 1..S (0 when idle).
//   - layer_idx          : Current layer, 1..N (0 when idle).
//   - num_samples        : Effective S latched at start (1 when B = 0).
//   - busy               : High while an inference is running.
//   - run_start          : 1-cycle strobe when start_inference is accepted.
//   - ic_write_en        : Cache phase - layer N-B of sample 1.
//   - ic_read_en         : Replay phase - layer N-B+1 of samples 2..S.
//   - bypass_feature_extractor : High for samples 2..S (layers 1..N-B skipped).
//   - mcd_en             : Stage 4 MCD enable, outputs of layers N-B..N-1.
//   - is_final_layer     : High while layer N is executing.
//   - inference_done     : 1-cycle strobe when layer N of sample S finishes, or
//                          when an early exit is taken.
//   - early_exit_active  : Latched early_exit_en for this inference.
//   - eval_pending       : Waiting for the convergence decision; the system must
//                          not start the next layer while this is high.
//   - samples_executed   : Monte Carlo passes completed so far (final s_actual).
//   - early_exit_triggered : The inference stopped before sample S.
//
// Description:
//   Sample 1 runs layers 1..N. Every later sample resumes directly at layer
//   N-B+1 and is fed from the IC buffer, so the deterministic part runs only
//   once, saving (N-B) x (S-1) layer evaluations. With B = N there is nothing
//   to cache and every sample restarts at layer 1. With B = 0 all passes are
//   identical, so a single sample is run.
//   Early exit: when layer N of a sample s < S finishes, the controller holds
//   the current layer/sample (eval_pending) until output_reducer has updated the
//   running statistics and decided. "Converged" ends the inference at s_actual = s;
//   otherwise sample s+1 starts. With early_exit_en = 0 nothing waits and the
//   sequencing is the baseline one.
//==============================================================================

`include "bcnn_pkg.vh"

module mc_sample_controller #(
    parameter LAYER_CNT_WIDTH = `LAYER_CNT_WIDTH,
    parameter SAMPLE_CNT_WIDTH = `SAMPLE_CNT_WIDTH
)(
    input wire clk, rst_n, start_inference, layer_done,

    // Configuration registers
    input wire [LAYER_CNT_WIDTH-1:0] total_layers_N,
    input wire [LAYER_CNT_WIDTH-1:0] bayesian_layers_B,
    input wire [SAMPLE_CNT_WIDTH-1:0] total_samples_S,

    // Early exit (UAMH Innovation 2)
    input wire early_exit_en,
    input wire eval_done,
    input wire early_exit_trigger,

    // Progress
    output reg [SAMPLE_CNT_WIDTH-1:0] sample_idx,
    output reg [LAYER_CNT_WIDTH-1:0] layer_idx,
    output reg [SAMPLE_CNT_WIDTH-1:0] num_samples,
    output reg busy,
    output wire run_start,

    // Routing control
    output wire ic_write_en,
    output wire ic_read_en,
    output wire bypass_feature_extractor,
    output wire mcd_en,
    output wire is_final_layer,
    output reg inference_done,

    // Early-exit status
    output reg early_exit_active,
    output reg eval_pending,
    output reg [SAMPLE_CNT_WIDTH-1:0] samples_executed,
    output reg early_exit_triggered
);

    localparam [SAMPLE_CNT_WIDTH-1:0] FIRST_SAMPLE = 1;
    localparam [LAYER_CNT_WIDTH-1:0] FIRST_LAYER = 1;

    // Configuration is latched at start so it stays stable for the whole run
    reg [LAYER_CNT_WIDTH-1:0] cfg_N, cfg_B;

    // Number of deterministic layers; layer N-B is the one that gets cached
    wire [LAYER_CNT_WIDTH-1:0] det_layers = cfg_N - cfg_B;

    assign run_start = start_inference && !busy && (total_layers_N != {LAYER_CNT_WIDTH{1'b0}});

    assign is_final_layer = busy && (layer_idx == cfg_N);
    assign bypass_feature_extractor = busy && (sample_idx > FIRST_SAMPLE) && (det_layers != {LAYER_CNT_WIDTH{1'b0}});
    assign ic_write_en = busy && (layer_idx == det_layers) && (sample_idx == FIRST_SAMPLE);
    assign ic_read_en = bypass_feature_extractor && (layer_idx == det_layers + FIRST_LAYER);

    // MCD sits on the outputs of layers N-B..N-1, i.e. the inputs of the B Bayesian layers
    assign mcd_en = busy && (cfg_B != {LAYER_CNT_WIDTH{1'b0}}) && (layer_idx >= det_layers) && (layer_idx < cfg_N);

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            cfg_N <= {LAYER_CNT_WIDTH{1'b0}};
            cfg_B <= {LAYER_CNT_WIDTH{1'b0}};
            num_samples <= {SAMPLE_CNT_WIDTH{1'b0}};
            sample_idx <= {SAMPLE_CNT_WIDTH{1'b0}};
            layer_idx <= {LAYER_CNT_WIDTH{1'b0}};
            busy <= 1'b0;
            inference_done <= 1'b0;
            early_exit_active <= 1'b0;
            eval_pending <= 1'b0;
            samples_executed <= {SAMPLE_CNT_WIDTH{1'b0}};
            early_exit_triggered <= 1'b0;
        end else begin
            inference_done <= 1'b0;

            if(run_start) begin
                cfg_N <= total_layers_N;
                cfg_B <= (bayesian_layers_B > total_layers_N) ? total_layers_N : bayesian_layers_B;
                num_samples <= ((total_samples_S == {SAMPLE_CNT_WIDTH{1'b0}}) || (bayesian_layers_B == {LAYER_CNT_WIDTH{1'b0}})) ? FIRST_SAMPLE : total_samples_S;
                sample_idx <= FIRST_SAMPLE;
                layer_idx <= FIRST_LAYER;
                busy <= 1'b1;
                early_exit_active <= early_exit_en;
                eval_pending <= 1'b0;
                samples_executed <= {SAMPLE_CNT_WIDTH{1'b0}};
                early_exit_triggered <= 1'b0;
            end else if(busy && eval_pending) begin
                // Convergence decision for the sample that just finished
                if(eval_done) begin
                    eval_pending <= 1'b0;
                    if(early_exit_trigger) begin
                        sample_idx <= {SAMPLE_CNT_WIDTH{1'b0}};
                        layer_idx <= {LAYER_CNT_WIDTH{1'b0}};
                        busy <= 1'b0;
                        inference_done <= 1'b1;
                        early_exit_triggered <= 1'b1;
                    end else begin
                        sample_idx <= sample_idx + 1'b1;
                        layer_idx <= det_layers + FIRST_LAYER;
                    end
                end
            end else if(busy && layer_done) begin
                if(layer_idx == cfg_N) begin
                    samples_executed <= sample_idx;
                end

                if(layer_idx != cfg_N) begin
                    layer_idx <= layer_idx + 1'b1;
                end else if(sample_idx != num_samples) begin
                    if(early_exit_active) begin
                        // Hold here until output_reducer decides whether to stop
                        eval_pending <= 1'b1;
                    end else begin
                        // Next MC sample skips layers 1..N-B and is fed from the IC buffer
                        sample_idx <= sample_idx + 1'b1;
                        layer_idx <= det_layers + FIRST_LAYER;
                    end
                end else begin
                    sample_idx <= {SAMPLE_CNT_WIDTH{1'b0}};
                    layer_idx <= {LAYER_CNT_WIDTH{1'b0}};
                    busy <= 1'b0;
                    inference_done <= 1'b1;
                end
            end
        end
    end

endmodule
