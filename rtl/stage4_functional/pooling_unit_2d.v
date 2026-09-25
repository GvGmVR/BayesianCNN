//==============================================================================
// Module: pooling_unit_2d.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 4 - 2D Spatial Dimension Downsampling
//------------------------------------------------------------------------------
// Purpose:
//   Performs 2x2 spatial Max Pooling or Average Pooling across all PF channels.
//
// Architectural Inputs:
//   - clk, rst_n        : Clock (220 MHz) and active-low reset.
//   - valid_in          : Valid strobe from SC addition stage.
//   - pool_mode         : 2-bit mode (00=Bypass, 01=Max Pool, 10=Avg Pool).
//   - pool_win_done     : 1-cycle strobe indicating 4th pixel of 2x2 patch arrived.
//   - pool_step         : 2-bit counter (0 to 3) indicating window step index.
//   - features_in       : PF parallel 8-bit channels from SC unit (PF x DATA_WIDTH bits).
//
// Architectural Outputs:
//   - pooled_features   : PF parallel 8-bit downsampled channels (PF x DATA_WIDTH bits).
//   - valid_out         : 1-cycle strobe indicating pooled_features is valid.
//
// Description:
//   In Max Pool mode, tracks running maximum over 4 pixels. In Avg Pool mode,
//   accumulates 4 pixels and divides by 4 (arithmetic shift right by 2).
//==============================================================================

`include "bcnn_pkg.vh"

module pooling_unit_2d #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PF = `PF,
    parameter PV = `PV,
    parameter POOL_MODE_WIDTH = `POOL_MODE_WIDTH,
    parameter POOL_CNT_WIDTH = `POOL_CNT_WIDTH
)(
    input wire clk, rst_n,valid_in,

    //Control
    input wire [POOL_MODE_WIDTH-1:0] pool_mode,
    input wire pool_win_done,
    input wire [POOL_CNT_WIDTH-1:0] pool_step,

    input wire [(PF*PV*DATA_WIDTH)-1:0] features_in,
    output reg [(PF*PV*DATA_WIDTH)-1:0] pooled_features,
    output reg valid_out    
);

    // Saturation constant for initialization
    localparam signed [DATA_WIDTH-1:0] INT_MIN = {1'b1, {(DATA_WIDTH-1){1'b0}}};

     // Intermediate Registers for multi-step pooling (parallel PF channels)
     reg signed [DATA_WIDTH-1:0] max_regs [(PF*PV)-1:0];
     reg signed [(DATA_WIDTH + POOL_CNT_WIDTH)-1:0] sum_regs [(PF*PV)-1:0];

    genvar i;
    always @(posedge clk or negedge clk) begin 
        if(!rst_n) begin 
            pooled_features <= {(PF*PV*DATA_WIDTH){1'b0}};
            valid_out <= 1'b0;
            for (i=0;i<(PF*PV); i=i+1) begin 
                max_regs[i] <= INT_MIN;
                sum_regs <= {(DATA_WIDTH + POOL_CNT_WIDTH){1'b0}};
            end
        end else if(valid_in) begin 
            case(pool_mode) 
                // Mode 00: Bypass (Pass through immediately)
                `POOL_MODE_BYPASS: begin 
                    pooled_features <= features_in;
                    valid_out <= 1'b1;
                end

                // Mode 01: 2x2 Max Pooling
                `POOL_MODE_MAX: begin 
                    for(i=0; i<(PF*PV);i=i+1) begin : POOL_MODE_MAX
                        wire signed [DATA_WIDTH-1:0] in_val;
                        assign in_val = features_in[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH];

                        if(pool_step == {POOL_CNT_WIDTH{1'b0}}) begin 
                            // Initialize with first pixel
                            max_regs[i] <= in_val;
                        end else begin 
                            // Maximum is tracked
                            if(in_val > max_regs[i]) begin 
                                max_regs[i] <= in_val;
                            end
                        end

                        if(pool_win_done) begin 
                            pooled_features[(i+1)*DATA_WIDTH-1: I*DATA_WIDTH] <= (in_val > max_regs[i]) ? in_val : max_regs[i];
                        end
                    end
                    valid_out <= pool_win_done;
                end

                // Mode 10: 2x2 Average Pooling
                `POOL_MODE_AVG: begin 
                    for (i = 0; i < (PF * PV); i = i + 1) begin : POOL_MODE_AVG
                        wire signed [DATA_WIDTH-1:0] in_val;
                        assign in_val = features_in[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH];
                        wire signed [(DATA_WIDTH + POOL_CNT_WIDTH)-1:0] in_val_ext;
                        assign in_val_ext = {{POOL_CNT_WIDTH{in_val[DATA_WIDTH-1]}}, in_val};
                        wire signed [(DATA_WIDTH + POOL_CNT_WIDTH)-1:0] current_sum;
                        assign current_sum = (pool_step == {POOL_CNT_WIDTH{1'b0}}) ? in_val_ext : (sum_regs[i] + in_val_ext);

                        sum_regs[i] <= current_sum;

                        if(pool_win_done) begin 
                            // Divide total sum by 2^POOL_CNT_WIDTH via arithmetic right shift
                            pooled_features[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH] <= current_sum >>> POOL_CNT_WIDTH;
                        end
                    end
                    valid_out <= pool_win_done;
                end

                default: begin 
                    pooled_features <= features_in;
                    valid_out <= 1'b1;
                end
            endcase
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule