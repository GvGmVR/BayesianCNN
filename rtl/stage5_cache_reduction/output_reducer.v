//==============================================================================
// Module: output_reducer.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 5 - Intermediate Caching & Output Reduction
//------------------------------------------------------------------------------
// Purpose:
//   Reduces the final-layer outputs of the S Monte Carlo samples into the
//   predictive mean (Eq. 1 in Fan et al.) and a per-channel variance that
//   serves as the uncertainty estimate.
//
// Architectural Inputs:
//   - clk, rst_n        : Clock (220 MHz) and active-low reset.
//   - clear             : 1-cycle strobe at inference start; clears accumulators.
//   - sample_valid      : Final-layer output vector of one sample is valid.
//   - sample_in         : PF x PV parallel INT8 final-layer outputs.
//   - sample_idx        : Current MC sample index (1..S).
//   - total_samples_S   : Number of samples S (>= 1, from mc_sample_controller).
//
// Architectural Outputs:
//   - mean_prediction   : PF x PV INT8 means, Mean[f] = Sum(x[f]) / S.
//   - uncertainty_score : PF x PV unsigned variances (VAR_OUT_WIDTH bits each),
//                         Var[f] = Sum(x[f]^2) / S - Mean[f]^2.
//   - reduction_done    : High once results are valid; held until next clear.
//
// Description:
//   Each sample contributes one PF x PV vector. Per channel, a running sum and
//   a running sum of squares are accumulated. When the S-th vector arrives the
//   two totals are divided by S with bit-serial restoring dividers (one pair
//   per channel, sharing the divisor), which takes VAR_ACCUM_WIDTH cycles and
//   avoids PF x PV wide combinational dividers. The mean is truncated toward
//   zero, which guarantees Mean^2 <= Sum(x^2)/S and hence a non-negative
//   variance. Requires VAR_ACCUM_WIDTH > REDUCER_ACCUM_WIDTH.
//==============================================================================

`include "bcnn_pkg.vh"

module output_reducer #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter SAMPLE_CNT_WIDTH = `SAMPLE_CNT_WIDTH,
    parameter REDUCER_ACCUM_WIDTH = `REDUCER_ACCUM_WIDTH,
    parameter VAR_ACCUM_WIDTH = `VAR_ACCUM_WIDTH,
    parameter VAR_OUT_WIDTH = 2*DATA_WIDTH // Var of INT8 data <= 2^(2*DATA_WIDTH-2)
)(
    input wire clk, rst_n, clear,

    input wire sample_valid,
    input wire [(PF*PV*DATA_WIDTH)-1:0] sample_in,
    input wire [SAMPLE_CNT_WIDTH-1:0] sample_idx,
    input wire [SAMPLE_CNT_WIDTH-1:0] total_samples_S,

    output wire [(PF*PV*DATA_WIDTH)-1:0] mean_prediction,
    output wire [(PF*PV*VAR_OUT_WIDTH)-1:0] uncertainty_score,
    output reg reduction_done
);

    localparam ST_WIDTH = 3;
    localparam [ST_WIDTH-1:0] RED_ACCUM = 0;
    localparam [ST_WIDTH-1:0] RED_LOAD = 1;
    localparam [ST_WIDTH-1:0] RED_DIV = 2;
    localparam [ST_WIDTH-1:0] RED_FINAL = 3;
    localparam [ST_WIDTH-1:0] RED_DONE = 4;
    localparam DIV_CNT_WIDTH = $clog2(VAR_ACCUM_WIDTH);

    reg [ST_WIDTH-1:0] state;
    reg [DIV_CNT_WIDTH-1:0] div_cnt;
    reg [SAMPLE_CNT_WIDTH-1:0] divisor;

    // Shared control
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state <= RED_ACCUM;
            div_cnt <= {DIV_CNT_WIDTH{1'b0}};
            divisor <= {SAMPLE_CNT_WIDTH{1'b0}};
            reduction_done <= 1'b0;
        end else if(clear) begin
            state <= RED_ACCUM;
            reduction_done <= 1'b0;
        end else begin
            case(state)
                RED_ACCUM: begin
                    // Last MC sample accumulated this cycle - start the 1/S normalisation
                    if(sample_valid && (sample_idx == total_samples_S)) begin
                        divisor <= total_samples_S;
                        state <= RED_LOAD;
                    end
                end
                RED_LOAD: begin
                    div_cnt <= {DIV_CNT_WIDTH{1'b0}};
                    state <= RED_DIV;
                end
                RED_DIV: begin
                    div_cnt <= div_cnt + 1'b1;
                    if(div_cnt == VAR_ACCUM_WIDTH-1) begin
                        state <= RED_FINAL;
                    end
                end
                RED_FINAL: begin
                    reduction_done <= 1'b1;
                    state <= RED_DONE;
                end
                default: begin
                    // RED_DONE: hold results until the next clear
                    state <= state;
                end
            endcase
        end
    end

    // Per-channel datapath
    genvar f;
    generate
        for(f=0; f<(PF*PV); f=f+1) begin : GEN_REDUCE
            wire signed [DATA_WIDTH-1:0] x = sample_in[f*DATA_WIDTH +: DATA_WIDTH];
            wire signed [VAR_OUT_WIDTH-1:0] x_sq = x * x;

            reg signed [REDUCER_ACCUM_WIDTH-1:0] sum;
            reg [VAR_ACCUM_WIDTH-1:0] sum_sq;

            // Restoring dividers: dividend shifts out MSB-first, quotient shifts in at LSB
            reg [VAR_ACCUM_WIDTH-1:0] q_mean, q_sq;
            reg [SAMPLE_CNT_WIDTH-1:0] r_mean, r_sq;
            reg neg;

            wire [REDUCER_ACCUM_WIDTH-1:0] sum_mag = sum[REDUCER_ACCUM_WIDTH-1] ? -sum : sum;
            wire [SAMPLE_CNT_WIDTH:0] r_mean_sh = {r_mean, q_mean[VAR_ACCUM_WIDTH-1]};
            wire [SAMPLE_CNT_WIDTH:0] r_sq_sh = {r_sq, q_sq[VAR_ACCUM_WIDTH-1]};
            wire mean_ge = (r_mean_sh >= divisor);
            wire sq_ge = (r_sq_sh >= divisor);

            // |Mean| <= 2^(DATA_WIDTH-1), so the quotient fits back into DATA_WIDTH bits
            wire signed [DATA_WIDTH-1:0] mean_q = neg ? -q_mean[DATA_WIDTH-1:0] : q_mean[DATA_WIDTH-1:0];
            wire signed [VAR_OUT_WIDTH-1:0] mean_sq = mean_q * mean_q;
            wire [VAR_ACCUM_WIDTH-1:0] var_full = q_sq - mean_sq;

            reg [DATA_WIDTH-1:0] mean_r;
            reg [VAR_OUT_WIDTH-1:0] var_r;

            assign mean_prediction[f*DATA_WIDTH +: DATA_WIDTH] = mean_r;
            assign uncertainty_score[f*VAR_OUT_WIDTH +: VAR_OUT_WIDTH] = var_r;

            always @(posedge clk or negedge rst_n) begin
                if(!rst_n) begin
                    sum <= {REDUCER_ACCUM_WIDTH{1'b0}};
                    sum_sq <= {VAR_ACCUM_WIDTH{1'b0}};
                    q_mean <= {VAR_ACCUM_WIDTH{1'b0}};
                    q_sq <= {VAR_ACCUM_WIDTH{1'b0}};
                    r_mean <= {SAMPLE_CNT_WIDTH{1'b0}};
                    r_sq <= {SAMPLE_CNT_WIDTH{1'b0}};
                    neg <= 1'b0;
                    mean_r <= {DATA_WIDTH{1'b0}};
                    var_r <= {VAR_OUT_WIDTH{1'b0}};
                end else if(clear) begin
                    sum <= {REDUCER_ACCUM_WIDTH{1'b0}};
                    sum_sq <= {VAR_ACCUM_WIDTH{1'b0}};
                end else begin
                    case(state)
                        RED_ACCUM: begin
                            if(sample_valid) begin
                                sum <= sum + x;
                                sum_sq <= sum_sq + x_sq;
                            end
                        end
                        RED_LOAD: begin
                            // Divide magnitudes, restore the sign of the mean afterwards
                            q_mean <= {{(VAR_ACCUM_WIDTH-REDUCER_ACCUM_WIDTH){1'b0}}, sum_mag};
                            q_sq <= sum_sq;
                            r_mean <= {SAMPLE_CNT_WIDTH{1'b0}};
                            r_sq <= {SAMPLE_CNT_WIDTH{1'b0}};
                            neg <= sum[REDUCER_ACCUM_WIDTH-1];
                        end
                        RED_DIV: begin
                            r_mean <= mean_ge ? (r_mean_sh - divisor) : r_mean_sh[SAMPLE_CNT_WIDTH-1:0];
                            q_mean <= {q_mean[VAR_ACCUM_WIDTH-2:0], mean_ge};
                            r_sq <= sq_ge ? (r_sq_sh - divisor) : r_sq_sh[SAMPLE_CNT_WIDTH-1:0];
                            q_sq <= {q_sq[VAR_ACCUM_WIDTH-2:0], sq_ge};
                        end
                        RED_FINAL: begin
                            mean_r <= mean_q;
                            var_r <= var_full[VAR_OUT_WIDTH-1:0];
                        end
                        default: begin
                            mean_r <= mean_r;
                        end
                    endcase
                end
            end
        end
    endgenerate

endmodule
