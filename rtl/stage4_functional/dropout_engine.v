//==============================================================================
// Module: dropout_engine.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 4 - Monte Carlo Dropout Masking (DE)
//------------------------------------------------------------------------------
// Purpose:
//   Applies the 64-bit random Bernoulli mask vector popped from Stage 2 
//   to the 64 output feature channels (Equation 2 in Fan et al.: O = Y ⊙ M).
//
// Architectural Inputs:
//   - clk, rst_n        : Clock (220 MHz) and active-low reset.
//   - valid_in          : Valid strobe from Pooling stage.
//   - mcd_en            : 1-bit control flag (1 = Bayesian MCD active, 0 = Non-Bayesian).
//   - features_in       : PF parallel 8-bit channels from Pooling unit (PF x DATA_WIDTH bits).
//   - mask_in           : PF-bit (64-bit) Bernoulli mask from Stage 2 FIFO (mask_out).
//   - mask_valid        : Valid handshake from Stage 2 FIFO.
//
// Architectural Outputs:
//   - mask_pop          : 1-cycle strobe sent to Stage 2 to pop the next mask word.
//   - masked_features   : PF parallel 8-bit masked feature output (PF x DATA_WIDTH bits).
//   - valid_out         : 1-cycle strobe indicating masked_features is valid.
//
// Description:
//   If mcd_en is active, pops a 64-bit mask from Stage 2 and zeroes out any 
//   channel where mask_in[f] == 0. If mcd_en is 0, features pass through unaltered.
//==============================================================================

`include "../bcnn_pkg.vh"

module dropout_engine #(
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter PF = `PF,
    parameter PV = `PV
)(
    input wire clk, rst_n, valid_in,

    //Bayesian control flag
    input wire mcd_en,

    input wire [(PF*PV*DATA_WIDTH)-1:0] features_in,

    // Interface with bernoulli sampler
    input wire [PF-1:0] mask_in,
    input wire mask_valid,
    output reg mask_pop,

    //Masked features for Stage 5 DRAM write
    output reg [(PF*PV*DATA_WIDTH)-1:0] masked_features,
    output reg valid_out
);

    integer f;
    always @(posedge clk or negedge rst_n) begin 
        if(!rst_n) begin 
            masked_features <= {(PF*PV*DATA_WIDTH){1'b0}};
            valid_out <= 1'b0;
            mask_pop <= 1'b0;
        end else if (valid_in) begin 
            valid_out <= 1'b1;

            if(mcd_en) begin 
                // Bayesian Mode: Pop 1 mask word from Stage 2 FIFO
                mask_pop <= 1'b1;

                // Channel-wise Bernoulli Masking: O[f] = Y[f] * M[f]
                for (f=0;f<(PF*PV);f=f+1) begin 
                    if(mask_in[f % PF] == 1'b1) begin 
                        masked_features[(f+1)*DATA_WIDTH-1 : f*DATA_WIDTH] <= features_in[(f+1)*DATA_WIDTH-1 : f*DATA_WIDTH];
                    end else begin
                        // Drop feature channel
                        masked_features[(f+1)*DATA_WIDTH-1 : f*DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                    end
                end
            end else begin 
                // Non-bayesian - bypass dropout
                mask_pop <= 1'b0;
                masked_features <= features_in;
            end
        end else begin 
            valid_out <= 1'b0;
            mask_pop <= 1'b0
        end
    end


endmodule