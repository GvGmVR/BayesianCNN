** Compile with Icarus Verilog (iverilog)

iverilog -I rtl -g2005-sv -o sim/stage1/stage1_sim.out rtl/stage1_buffers/ram_bank.v rtl/stage1_buffers/tree_fanout.v rtl/stage1_buffers/crossbar_switch.v rtl/stage1_buffers/data_ingress_engine.v rtl/stage1_buffers/read_addr_gen.v rtl/stage1_buffers/weight_fifo.v rtl/stage1_buffers/smart_data_buffer.v rtl/stage1_buffers/smart_weight_buffer.v sim/tb_stage1_top.v



** Execute simulation(vvp)

vvp sim/stage1_sim.out

** View Signals in GTKWave

gtkwave sim/stage1_simulation.vcd