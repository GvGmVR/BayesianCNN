'''
Run ` python scripts/gen_test_vectors.py

'''

import os
import numpy as np

os.makedirs("../data",exist_ok=True)

# Parameters matching testbench
H,W,C = 4,4,64
PF = 64

print(f"Generating test vectors for H={H}, W={W}, C={C}, PF={PF}...")

#Generate Input Feature Map (INT8: 0 to 255)
# Shape: (H, W, C)

input_feature_map = np.zeros((H,W,C),dtype=np.uint8)
for h in range(H):
    for w in range(W):
        p=h*W+w
        for c in range(C):
            input_feature_map[h,w,c] = (p*10+c) & 0xFF


# Write to Hex file (1 line = 512-bit word = 64 bytes)
with open("../data/input_feature_map.hex", "w") as f:
    for h in range(H):
        for w in range(W):
            # Write channels 63 down to 0 (Big-Endian hex representation per line)
            line_bytes = [f"{input_feature_map[h, w, c]:02X}" for c in reversed(range(C))]
            f.write("".join(line_bytes) + "\n")

print("Generated: ../data/input_feature_map.hex")

# Generate Filter Weights
# Shape: (PF, C)
weights = np.zeros((PF,C),dtype=np.uint8)
for pf in range(PF):
    for c in range(C):
        weights[pf,c] = (pf*10+c) & 0xFF

with open("../data/filter_weights.hex", "w") as f:
    for pf in range(PF):
        line_bytes = [f"{weights[pf, c]:02X}" for c in reversed(range(C))]
        f.write("".join(line_bytes) + "\n")

print("Generated: ../data/filter_weights.hex")
print("Vector generation complete!")    