//==============================================================================
// Module: convergence_monitor.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: UAMH Innovation 2 - Closed-Loop Early-Exit Sample Throttling
//------------------------------------------------------------------------------
// Purpose:
//   Watches how much the running per-channel variance changes from one Monte
//   Carlo pass to the next and signals when the predictive uncertainty has
//   stabilised, so the remaining passes can be skipped.
//
// Architectural Inputs:
//   - clk, rst_n        : Clock (220 MHz) and active-low reset.
//   - clear             : Inference start; forgets the previous pass.
//   - sample_valid      : Running statistics of pass sample_idx are valid.
//   - sample_idx        : Monte Carlo pass s these statistics include.
//   - current_variance  : Running variance vector after pass s (PF x PV channels).
//   - early_exit_en     : 1 = monitor active, 0 = never trigger.
//   - early_exit_thresh : Stability tolerance epsilon.
//   - min_samples       : Warm-up S_min; no exit is considered before pass S_min.
//
// Architectural Outputs:
//   - early_exit_trigger: 1-cycle pulse, one cycle after sample_valid, when the
//                         last CONV_STABILITY_COUNT passes were all stable.
//   - variance_delta_out: max_f |var_s[f] - var_(s-1)[f]| of the last pass
//                         (saturated to EARLY_EXIT_THRESH_WIDTH bits).
//
// Description:
//   delta_s = max over channels of |var_s[f] - var_(s-1)[f]|. A pass is stable
//   when s >= min_samples and delta_s <= epsilon; the first pass after clear has
//   no predecessor and is never stable. stable_count counts consecutive stable
//   passes and resets on any unstable one; the trigger fires on the pass that
//   makes it reach CONV_STABILITY_COUNT.
//==============================================================================

`include "bcnn_pkg.vh"

module convergence_monitor #(
    parameter PF = `PF,
    parameter PV = `PV,
    parameter SAMPLE_CNT_WIDTH = `SAMPLE_CNT_WIDTH,
    parameter VAR_OUT_WIDTH = 2*`DATA_WIDTH,
    parameter EARLY_EXIT_THRESH_WIDTH = `EARLY_EXIT_THRESH_WIDTH,
    parameter CONV_STABILITY_COUNT = `CONV_STABILITY_COUNT
)(
    input wire clk, rst_n, clear,

    input wire sample_valid,
    input wire [SAMPLE_CNT_WIDTH-1:0] sample_idx,
    input wire [(PF*PV*VAR_OUT_WIDTH)-1:0] current_variance,

    input wire early_exit_en,
    input wire [EARLY_EXIT_THRESH_WIDTH-1:0] early_exit_thresh,
    input wire [SAMPLE_CNT_WIDTH-1:0] min_samples,

    output reg early_exit_trigger,
    output reg [EARLY_EXIT_THRESH_WIDTH-1:0] variance_delta_out
);

    localparam STAB_WIDTH = $clog2(CONV_STABILITY_COUNT+1);

    reg [(PF*PV*VAR_OUT_WIDTH)-1:0] prev_variance;
    reg has_prev;
    reg [STAB_WIDTH-1:0] stable_count;

    // Largest per-channel absolute change against the previous pass
    reg [VAR_OUT_WIDTH-1:0] cur_v, prev_v, diff, max_delta;
    integer f;

    always @(*) begin
        max_delta = {VAR_OUT_WIDTH{1'b0}};
        for(f=0; f<(PF*PV); f=f+1) begin
            cur_v = current_variance[f*VAR_OUT_WIDTH +: VAR_OUT_WIDTH];
            prev_v = prev_variance[f*VAR_OUT_WIDTH +: VAR_OUT_WIDTH];
            diff = (cur_v > prev_v) ? (cur_v - prev_v) : (prev_v - cur_v);
            if(diff > max_delta) begin
                max_delta = diff;
            end
        end
    end

    wire [EARLY_EXIT_THRESH_WIDTH-1:0] delta_sat = (max_delta > {EARLY_EXIT_THRESH_WIDTH{1'b1}}) ? {EARLY_EXIT_THRESH_WIDTH{1'b1}} : max_delta;
    wire stable = early_exit_en && has_prev && (sample_idx >= min_samples) && (max_delta <= early_exit_thresh);

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            prev_variance <= {(PF*PV*VAR_OUT_WIDTH){1'b0}};
            has_prev <= 1'b0;
            stable_count <= {STAB_WIDTH{1'b0}};
            early_exit_trigger <= 1'b0;
            variance_delta_out <= {EARLY_EXIT_THRESH_WIDTH{1'b0}};
        end else if(clear) begin
            has_prev <= 1'b0;
            stable_count <= {STAB_WIDTH{1'b0}};
            early_exit_trigger <= 1'b0;
            variance_delta_out <= {EARLY_EXIT_THRESH_WIDTH{1'b0}};
        end else begin
            early_exit_trigger <= 1'b0;
            if(sample_valid) begin
                prev_variance <= current_variance;
                has_prev <= 1'b1;
                variance_delta_out <= has_prev ? delta_sat : {EARLY_EXIT_THRESH_WIDTH{1'b0}};
                if(stable) begin
                    if(stable_count != CONV_STABILITY_COUNT) begin
                        stable_count <= stable_count + 1'b1;
                    end
                    early_exit_trigger <= (stable_count >= CONV_STABILITY_COUNT-1);
                end else begin
                    stable_count <= {STAB_WIDTH{1'b0}};
                end
            end
        end
    end

endmodule
