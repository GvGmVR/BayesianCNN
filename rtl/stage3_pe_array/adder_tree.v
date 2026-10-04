//==============================================================================
// Module: adder_tree.v
// Project: Bayesian CNN (BayesCNN) Hardware Accelerator
// Stage: 3 - Fully Parameterized Log2-Level Binary Adder Tree
//------------------------------------------------------------------------------
// Purpose:
//   Sums PC parallel multiplication products into a single scalar channel dot-product.
//
// Architectural Inputs:
//   - prod_vec        : PC parallel signed products from Multiplier Array (PC x IN_WIDTH bits).
//
// Architectural Outputs:
//   - sum_out         : Signed sum of width (IN_WIDTH + LOG2_PC bits).
//
// Description:
//   Generates a balanced binary adder tree with LOG2_PC stages. At each level L,
//   the bit-width grows by +1 to guarantee zero arithmetic overflow.
//==============================================================================

`include "bcnn_pkg.vh"

module adder_tree #(
    parameter PC = `PC,
    parameter LOG2_PC = `LOG2_PC,
    parameter IN_WIDTH = `MULT_OUT_WIDTH,
    parameter OUT_WIDTH = `ADDR_TREE_WIDTH
)(
    input wire [(PC*IN_WIDTH-1):0] prod_vec,
    output wire signed [OUT_WIDTH-1:0] sum_out
);

    // Wire arrays for each stage of the binary tree
    // Level 0 (Inputs): 64 nodes of 16-bit
    // Level 1: 32 nodes of 17-bit
    // Level 2: 16 nodes of 18-bit
    // Level 3: 8 nodes of 19-bit
    // Level 4: 4 nodes of 20-bit
    // Level 5: 2 nodes of 21-bit
    // Level 6: 1 node of 22-bit (Output)

    wire signed [IN_WIDTH:0] lvl1_nodes [ (PC/2)-1 : 0 ];
    wire signed [IN_WIDTH+1:0] lvl2_nodes [ (PC/4)-1 : 0 ];
    wire signed [IN_WIDTH+2:0] lvl3_nodes [ (PC/8)-1 : 0 ];
    wire signed [IN_WIDTH+3:0] lvl4_nodes [ (PC/16)-1 : 0 ];
    wire signed [IN_WIDTH+4:0] lvl5_nodes [ (PC/32)-1 : 0 ];
    wire signed [IN_WIDTH+5:0] lvl6_nodes [ (PC/64)-1 : 0 ];

    // Level 1 Adders (PC -> PC/2)
    genvar i1;
    generate 
        for (i1=0;i1<(PC/2);i1=i1+1) begin: GEN_LVL1
            wire signed [IN_WIDTH-1:0] opA =  prod_vec[(2*i1+1)*IN_WIDTH-1:(2*i1)*IN_WIDTH];
            wire signed [IN_WIDTH-1:0] opB =  prod_vec[(2*i1+2)*IN_WIDTH-1:(2*i1+1)*IN_WIDTH];
            assign lvl1_nodes[i1] = opA + opB;
        end
    endgenerate

    // Level 2 Adders (PC/2 -> PC/4)
    genvar i2;
    generate
        for (i2 = 0; i2 < (PC/4); i2 = i2 + 1) begin : GEN_LVL2
            assign lvl2_nodes[i2] = lvl1_nodes[2*i2] + lvl1_nodes[2*i2 + 1];
        end
    endgenerate

    // Level 3 Adders (PC/4 -> PC/8)
    genvar i3;
    generate
        for (i3 = 0; i3 < (PC/8); i3 = i3 + 1) begin : GEN_LVL3
            assign lvl3_nodes[i3] = lvl2_nodes[2*i3] + lvl2_nodes[2*i3 + 1];
        end
    endgenerate

    // Level 4 Adders (PC/8 -> PC/16)
    genvar i4;
    generate
        for (i4 = 0; i4 < (PC/16); i4 = i4 + 1) begin : GEN_LVL4
            assign lvl4_nodes[i4] = lvl3_nodes[2*i4] + lvl3_nodes[2*i4 + 1];
        end
    endgenerate

    // Level 5 Adders (PC/16 -> PC/32)
    genvar i5;
    generate
        for (i5 = 0; i5 < (PC/32); i5 = i5 + 1) begin : GEN_LVL5
            assign lvl5_nodes[i5] = lvl4_nodes[2*i5] + lvl4_nodes[2*i5 + 1];
        end
    endgenerate

    // Level 6 Adders (PC/32 -> 1 final sum)
    genvar i6;
    generate
        for (i6 = 0; i6 < (PC/64); i6 = i6 + 1) begin : GEN_LVL6
            assign lvl6_nodes[i6] = lvl5_nodes[2*i6] + lvl5_nodes[2*i6 + 1];
        end
    endgenerate

    // Final tree output
    assign sum_out = lvl6_nodes[0];
    
endmodule