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
//   - mask_valid        : Valid handshake from Stage 2 FIFO (FIFO not empty).
//   - mask_load         : 1-cycle strobe from controller at the start of a new
//                         PF-filter output tile / new MC sample. Pops one mask
//                         word and latches it for the whole output feature map.
//
// Architectural Outputs:
//   - mask_pop          : Pop strobe to Stage 2 FIFO (same cycle as mask_load,
//                         FIFO is first-word-fall-through so mask_in is the head).
//   - masked_features   : PF parallel 8-bit masked feature output (PF x DATA_WIDTH bits).
//   - valid_out         : 1-cycle strobe indicating masked_features is valid.
//
// Description:
//   MCD applies a filter-wise mask M_i (one Bernoulli bit per output filter) to
//   the output feature map Y_i (Fan et al., Sec. II-B1, Eq. 2). The same mask
//   therefore holds for every spatial position of a PF-filter tile within one
//   MC sample: a mask is popped only on mask_load, not per output pixel.
//   If mcd_en is active, channel f is zeroed when mask[f] == 0. If mcd_en is 0,
//   features pass through unaltered and no masks are consumed.
//==============================================================================

`include "bcnn_pkg.vh"

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
    input wire mask_load,
    output wire mask_pop,

    //Masked features for Stage 5 DRAM write
    output reg [(PF*PV*DATA_WIDTH)-1:0] masked_features,
    output reg valid_out
);

    // Filter-wise mask held for the current PF-filter tile (reset: keep all filters)
    reg [PF-1:0] mask_reg;

    // Pop only when a new mask is requested in Bayesian mode and one is available
    assign mask_pop = mask_load && mcd_en && mask_valid;

    // A mask loaded in the same cycle as valid_in applies to that pixel
    wire [PF-1:0] active_mask = mask_pop ? mask_in : mask_reg;

    always @(posedge clk or negedge rst_n) begin 
        if(!rst_n) begin 
            mask_reg <= {PF{1'b1}};
        end else if (mask_pop) begin 
            mask_reg <= mask_in;
        end
    end

    integer f;
    always @(posedge clk or negedge rst_n) begin 
        if(!rst_n) begin 
            masked_features <= {(PF*PV*DATA_WIDTH){1'b0}};
            valid_out <= 1'b0;
        end else if (valid_in) begin 
            valid_out <= 1'b1;

            if(mcd_en) begin 
                // Channel-wise Bernoulli Masking: O[f] = Y[f] * M[f]
                for (f=0;f<(PF*PV);f=f+1) begin 
                    if(active_mask[f % PF] == 1'b1) begin 
                        masked_features[f*DATA_WIDTH +: DATA_WIDTH] <= features_in[f*DATA_WIDTH +: DATA_WIDTH];
                    end else begin
                        // Drop feature channel
                        masked_features[f*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                    end
                end
            end else begin 
                // Non-bayesian - bypass dropout
                masked_features <= features_in;
            end
        end else begin 
            valid_out <= 1'b0;
        end
    end

endmodule