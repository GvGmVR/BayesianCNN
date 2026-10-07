"""
Architectural block diagrams of the baseline BayesCNN accelerator (Fan et al.).

Renders six publication-grade PNGs into images/ from the module hierarchy, instance
names, signals and bus widths of the baseline RTL (rtl/stage1_buffers .. rtl/stage5_
cache_reduction and rtl/bcnn_top.v), using the default configuration of rtl/bcnn_pkg.vh:
PF = PC = 64, PV = 1, DATA_WIDTH = 8. The UAMH extensions (rtl/UAMH/) are intentionally
not shown.

Usage (from the repository root):
    venv/Scripts/python.exe scripts/generate_architecture_diagrams.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon
from matplotlib.path import Path

# ----------------------------------------------------------------------------
# Canvas and palette
# ----------------------------------------------------------------------------
FIG_W, FIG_H, DPI = 24, 16, 150          # 3600 x 2400 px
XMAX, YMAX = 240, 160                    # drawing units (10 units per inch)

PALETTE = {
    "mem": ("#E8F0FE", "#1A73E8"),       # memory / buffers
    "alu": ("#FEF7E0", "#F29900"),       # arithmetic / compute
    "ctl": ("#F3E8FD", "#9334E6"),       # control / FSM / registers
    "rte": ("#F1F3F4", "#5F6368"),       # routing / muxes / fan-out
}
INK = "#202124"
SUB = "#5F6368"
DATA_C = "#3C4043"
CTRL_C = "#9334E6"
LOOP_C = "#188038"
MONO = "DejaVu Sans Mono"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")


def canvas(title, subtitle):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    ax.set_facecolor("white")
    ax.text(5, 154, title, fontsize=27, fontweight="bold", color=INK, va="center")
    ax.text(235, 154, "Baseline · PF = PC = 64, PV = 1, DATA_WIDTH = 8 (rtl/bcnn_pkg.vh)", ha="right", va="center",
            fontsize=12, color=SUB, style="italic")
    ax.text(5, 148.6, subtitle, fontsize=14.5, color=SUB, va="center")
    ax.plot([5, 235], [145.2, 145.2], color="#DADCE0", lw=1.2)
    return fig, ax


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------
def block(ax, x, y, w, h, title, inst=None, lines=(), kind="mem", stacked=0,
          title_size=14.5, body_size=11.5, align="left", line_step=2.75, body_top=None):
    """Rounded module block; `stacked` draws shadow copies to denote an instance array."""
    fill, edge = PALETTE[kind]
    for k in range(stacked, 0, -1):
        ax.add_patch(FancyBboxPatch((x + 1.1 * k, y - 1.1 * k), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    fc=fill, ec=edge, lw=1.3, alpha=0.55, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=fill, ec=edge, lw=2.2, zorder=3))
    ty = y + h - 2.9
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=INK, zorder=4)
    if inst:
        ty -= 2.8
        ax.text(x + w / 2, ty, inst, ha="center", va="center", fontsize=body_size, family=MONO, color=edge, zorder=4)
    ly = (ty - 3.3) if body_top is None else body_top
    for ln in lines:
        if align == "left":
            ax.text(x + 1.8, ly, ln, ha="left", va="center", fontsize=body_size, color=INK, zorder=4)
        else:
            ax.text(x + w / 2, ly, ln, ha="center", va="center", fontsize=body_size, color=INK, zorder=4)
        ly -= line_step


def container(ax, x, y, w, h, label, inst=None, color="#80868B", fs=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2",
                                fc="none", ec=color, lw=1.8, ls=(0, (6, 4)), zorder=1))
    ax.text(x + 2.0, y + h - 2.6, label, ha="left", va="center", fontsize=fs, fontweight="bold", color=INK, zorder=4)
    if inst:
        ax.text(x + w - 2.0, y + h - 2.6, inst, ha="right", va="center", fontsize=12.5, family=MONO, color=SUB, zorder=4)


def wire(ax, pts, kind="data", label=None, lpos=None, arrow=True, lw=None, fs=11, both=False, lalign="center"):
    """Polyline with arrow head; data = solid dark bus, ctrl = dashed violet strobe, loop = dashed green."""
    color = {"data": DATA_C, "ctrl": CTRL_C, "loop": LOOP_C}[kind]
    lw = lw or (2.6 if kind == "data" else 1.7)
    ls = "-" if kind == "data" else (0, (5, 3))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
    style = "<|-|>" if both else ("-|>" if arrow else "-")
    ax.add_patch(FancyArrowPatch(path=Path(pts, codes), arrowstyle=style, mutation_scale=20 if kind == "data" else 15,
                                 color=color, lw=lw, linestyle=ls, zorder=5, shrinkA=0, shrinkB=0, joinstyle="miter"))
    if label:
        if lpos is None:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            lpos = ((x0 + x1) / 2, (y0 + y1) / 2 + 1.7)
        ax.text(lpos[0], lpos[1], label, ha=lalign, va="center", fontsize=fs, family=MONO, color=color, zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.0))


def lbl(ax, x, y, text, kind="data", fs=10.5, ha="center", mono=True):
    color = {"data": DATA_C, "ctrl": CTRL_C, "loop": LOOP_C, "sub": SUB, "ink": INK}[kind]
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, family=MONO if mono else None, color=color, zorder=6)


def port(ax, x, y, text, kind="data", w=None, fs=12):
    """Stage-boundary port tag (pill) with its left edge at x."""
    w = w or (len(text) * 0.98 + 4)
    edge = DATA_C if kind == "data" else CTRL_C
    ax.add_patch(FancyBboxPatch((x, y - 2.3), w, 4.6, boxstyle="round,pad=0,rounding_size=2.3",
                                fc="white", ec=edge, lw=1.6, zorder=6))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=edge, zorder=7)


def note(ax, x, y, w, h, lines, fs=11, title=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#FFFFFF", ec="#BDC1C6", lw=1.3, ls=(0, (3, 2)), zorder=2))
    ly = y + h - 2.6
    if title:
        ax.text(x + 1.6, ly, title, ha="left", va="center", fontsize=fs + 0.5, fontweight="bold", color=INK, zorder=4)
        ly -= 2.8
    for ln in lines:
        ax.text(x + 1.6, ly, ln, ha="left", va="center", fontsize=fs, color=SUB, zorder=4)
        ly -= 2.6


def badge(ax, x, y, num, text):
    ax.add_patch(Circle((x, y), 1.9, fc=LOOP_C, ec="none", zorder=8))
    ax.text(x, y, num, ha="center", va="center", fontsize=12, fontweight="bold", color="white", zorder=9)
    ax.text(x + 3.0, y, text, ha="left", va="center", fontsize=11, color=LOOP_C, zorder=9)


def legend(ax, y=4.5, extra=None):
    x = 5
    for kind, text in [("mem", "Memory / buffers"), ("alu", "Arithmetic / compute"),
                       ("ctl", "Control / FSM / registers"), ("rte", "Routing / mux / fan-out")]:
        fill, edge = PALETTE[kind]
        ax.add_patch(FancyBboxPatch((x, y - 1.6), 5.5, 3.2, boxstyle="round,pad=0,rounding_size=0.6", fc=fill, ec=edge, lw=1.8))
        ax.text(x + 7, y, text, va="center", fontsize=12.5, color=INK)
        x += 7 + len(text) * 0.95 + 4
    for color, style, text in [(DATA_C, "-", "Data bus [msb:lsb]"), (CTRL_C, (0, (5, 3)), "Control / handshake")] + \
            ([(LOOP_C, (0, (5, 3)), extra)] if extra else []):
        ax.add_patch(FancyArrowPatch((x, y), (x + 9, y), arrowstyle="-|>", mutation_scale=17, color=color,
                                     lw=2.4 if style == "-" else 1.7, linestyle=style))
        ax.text(x + 10.5, y, text, va="center", fontsize=12.5, color=INK)
        x += 10.5 + len(text) * 0.95 + 5


def mux(ax, x, y, w, h, label, sel=None):
    """Trapezoid multiplexer symbol."""
    fill, edge = PALETTE["rte"]
    ax.add_patch(Polygon([(x, y), (x + w, y + h * 0.18), (x + w, y + h * 0.82), (x, y + h)], closed=True,
                         fc=fill, ec=edge, lw=2.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, fontweight="bold", color=INK,
            rotation=90, zorder=4)
    if sel:
        lbl(ax, x + w / 2, y - 2.0, sel, "ctrl", fs=10.5)


# ----------------------------------------------------------------------------
# 1. Stage 1 - Smart buffers
# ----------------------------------------------------------------------------
def stage1():
    fig, ax = canvas("Stage 1 — Smart Data & Weight Buffers",
                     "rtl/stage1_buffers/  ·  Fan et al., Sec. III-A4, Fig. 6 (smart data buffer), Fig. 7 (data layout), "
                     "Fig. 8 (smart weight buffer), Algorithm 1 (RAG)")

    # ---------------- smart data buffer ----------------
    container(ax, 22, 58, 186, 84, "smart_data_buffer.v  —  Ping-Pong double-buffered feature memory", "u_stage1_data")
    block(ax, 26, 106, 50, 30, "Data Ingress Engine", "data_ingress_engine.v · u_engine", [
        "FSM: IDLE → WRITE → DONE",
        "Nested counters  c_tile → w_tile → h → l",
        "Address map (Fig. 7, channel-first):",
        "  M = ((l·H + h)·W_tiles + w)·C_tiles + c",
        "Accept beat on dram_valid & dram_ready",
    ], kind="ctl", body_size=11)
    lbl(ax, 51, 102.2, "Port A: din_a [511:0] · addr_a [9:0] · we_a [63:0]", fs=10.3)
    block(ax, 26, 64, 50, 34, "Read Address Generator", "read_addr_gen.v · u_rag", [
        "Algorithm 1 loop nest (innermost first):",
        "  c_tile → kw → kh → kl(3D) → w_tile → h",
        "h_in = h·stride + kh",
        "w_in = w_tile·PV·stride + kw",
        "One BRAM read per cycle; flags describe",
        "the same read: window_done, last_window",
    ], kind="ctl", body_size=11)

    mux(ax, 84, 106, 7, 30, "Port-A steer", sel="ping_pong_sel")
    block(ax, 102, 113, 40, 24, "PING RAM banks", "ram_bank.v · u_ping_bank ×64", [
        "GEN_PING_PONG_RAMS: PV×PC = 64",
        "each 1024 × 8 b  (RAM_DEPTH)",
        "1 channel / bank, 1-cycle read",
    ], kind="mem", stacked=2, body_size=11)
    block(ax, 102, 80, 40, 24, "PONG RAM banks", "ram_bank.v · u_pong_bank ×64", [
        "same geometry as Ping",
        "Port A ← ingress, Port B ← RAG",
        "sync. read: dout_b valid next cycle",
    ], kind="mem", stacked=2, body_size=11)
    mux(ax, 152, 84, 7, 50, "raw_ram_out select", sel="ping_pong_sel")
    block(ax, 170, 100, 14, 24, "Crossbar", "u_crossbar", ["enable =", "re_b_d1", "(PV = 1:", "identity)"],
          kind="rte", align="center", body_size=10.3, title_size=12.5)
    block(ax, 190, 100, 14, 24, "Tree", "u_fanout", ["tree_fanout", "duplicate", "× PF = 64"],
          kind="rte", align="center", body_size=10.3, title_size=12.5)
    block(ax, 102, 62, 40, 11, "BRAM-latency align", "rag_re_b_d1 · rag_window_done_d1", [],
          kind="ctl", title_size=12.5, body_size=10.5)

    # ingress
    port(ax, 3, 128, "Off-chip DRAM", w=17)
    lbl(ax, 3, 122.6, "dram_data_in [511:0]", fs=10.3, ha="left")
    lbl(ax, 3, 119.9, "dram_valid /", "ctrl", fs=10.3, ha="left")
    lbl(ax, 3, 117.4, "dram_ready", "ctrl", fs=10.3, ha="left")
    wire(ax, [(20, 128), (26, 128)])
    wire(ax, [(76, 121), (84, 121)])
    wire(ax, [(91, 128), (102, 128)])
    lbl(ax, 96.5, 130.2, "sel=0", "sub", fs=10)
    wire(ax, [(91, 112), (95, 112), (95, 100), (102, 100)])
    lbl(ax, 97.0, 102.0, "sel=1", "sub", fs=9)

    # port B (read) from RAG to both sets, strobes to the latency register
    wire(ax, [(76, 86), (102, 86)], "ctrl", "addr_b [9:0], re_b", lpos=(87.5, 88.4), fs=10.3)
    wire(ax, [(99, 86), (99, 118), (102, 118)], "ctrl")
    wire(ax, [(76, 70), (102, 70)], "ctrl", "re_b, window_done", lpos=(89, 72.3), fs=10.3)

    # bank outputs, select, crossbar, fan-out
    wire(ax, [(142, 125), (152, 125)])
    wire(ax, [(142, 92), (152, 92)])
    lbl(ax, 147.3, 127.6, "dout_b", fs=10)
    lbl(ax, 147.3, 94.6, "dout_b", fs=10)
    wire(ax, [(159, 112), (170, 112)])
    lbl(ax, 164.5, 116.4, "raw_ram_out", fs=10)
    lbl(ax, 164.5, 108.6, "[511:0]", fs=10)
    wire(ax, [(184, 112), (190, 112)])
    wire(ax, [(142, 70.5), (164, 70.5), (164, 104), (170, 104)], "ctrl", "enable", lpos=(153, 72.9), fs=10.3)

    # outputs to Stage 3
    wire(ax, [(204, 112), (214, 112)])
    port(ax, 214, 112, "→ Stage 3", w=20)
    lbl(ax, 224, 106.3, "pe_data_out", fs=10.5)
    lbl(ax, 224, 103.6, "[32767:0] (4 KiB)", fs=10.5)
    wire(ax, [(142, 64.5), (214, 64.5)], "ctrl", "re_b_valid, window_done  (aligned with pe_data_out)",
         lpos=(178, 61.9), fs=10.3)
    port(ax, 214, 64.5, "→ Stage 3", "ctrl", w=20)

    note(ax, 167, 78, 39, 19, [
        "sel=0: ingress → PING, RAG ← PONG",
        "sel=1: ingress → PONG, RAG ← PING",
        "bcnn_top toggles sel in S_ARM, so",
        "compute reads the bank just written",
    ], fs=10.5, title="Ping-Pong double buffering")

    # ---------------- smart weight buffer ----------------
    container(ax, 22, 8, 186, 44, "smart_weight_buffer.v  —  PC × PF weight FIFOs + PV fan-out", "u_stage1_weight")
    wire(ax, [(51, 64), (51, 55), (104, 55), (104, 45)], "ctrl")
    lbl(ax, 106, 55, "weight_pop ← read_issue   ·   recirculation stops in last_window", "ctrl", fs=10.3, ha="left")

    port(ax, 3, 32, "Off-chip DRAM", w=17)
    lbl(ax, 3, 26.6, "weight_din", fs=10.3, ha="left")
    lbl(ax, 3, 24.0, "[32767:0]", fs=10.3, ha="left")
    lbl(ax, 3, 21.3, "weight_push", "ctrl", fs=10.3, ha="left")
    wire(ax, [(20, 32), (28, 32)])
    mux(ax, 28, 19, 7, 26, "wb_din mux")
    lbl(ax, 33.5, 15.3, "sel = recirc_push", "ctrl", fs=10, ha="left")
    wire(ax, [(35, 32), (48, 32)])
    block(ax, 48, 18, 60, 27, "Weight FIFO array", "weight_fifo.v · GEN_WEIGHT_FIFOS u_weight_fifo ×4096", [
        "PC × PF = 4096 FIFOs, each 512 × 8 b",
        "dout registered on pop, held between pops",
        "weight_full = |full,  weight_empty = |empty",
    ], kind="mem", stacked=2, body_size=11)
    wire(ax, [(110.2, 32), (136, 32)])
    lbl(ax, 123, 36.0, "raw_weight_bus", fs=10.3)
    lbl(ax, 123, 28.4, "[32767:0]", fs=10.3)
    block(ax, 136, 22, 24, 20, "PV Fan-out", "GEN_WEIGHT_PV_FANOUT", ["replicate × PV = 1"], kind="rte",
          align="center", body_size=10.5, title_size=13)
    wire(ax, [(160, 32), (214, 32)], "data", "pe_weight_out [32767:0] (4 KiB)", lpos=(187, 34.8), fs=10.5)
    port(ax, 214, 32, "→ Stage 3", w=20)
    wire(ax, [(128, 32), (128, 13), (31.5, 13), (31.5, 19)], "loop")
    lbl(ax, 115, 10.6, "weight recirculation (bcnn_top): recirc_push = weight_pop_d1 & !last_window_d1  —  "
                       "the PF filters' weights are reused for every output pixel (Fig. 8)", "loop", fs=10.3, mono=False)
    legend(ax, extra="Weight reuse loop")
    return save(fig, "stage1_smart_buffers.png")


# ----------------------------------------------------------------------------
# 2. Stage 2 - Bernoulli sampler
# ----------------------------------------------------------------------------
def lfsr_chain(ax, x0, y0, cell_w, cell_h, labels):
    fill, edge = PALETTE["ctl"]
    xs = []
    for i, lab in enumerate(labels):
        x = x0 + i * (cell_w + 1.2)
        xs.append(x)
        if lab == "…":
            ax.text(x + cell_w / 2, y0 + cell_h / 2, "· · ·", ha="center", va="center", fontsize=15, color=SUB)
            continue
        ax.add_patch(FancyBboxPatch((x, y0), cell_w, cell_h, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc=fill, ec=edge, lw=1.8, zorder=3))
        ax.text(x + cell_w / 2, y0 + cell_h / 2, lab, ha="center", va="center", fontsize=11.5, family=MONO, color=INK, zorder=4)
    return xs


def xor_gate(ax, x, y, r=2.2):
    ax.add_patch(Circle((x, y), r, fc="#FEF7E0", ec="#F29900", lw=2.0, zorder=6))
    ax.plot([x - r * 0.65, x + r * 0.65], [y, y], color="#F29900", lw=2.0, zorder=7)
    ax.plot([x, x], [y - r * 0.65, y + r * 0.65], color="#F29900", lw=2.0, zorder=7)


def stage2():
    fig, ax = canvas("Stage 2 — Bernoulli Sampler",
                     "rtl/stage2_sampler/  ·  Fan et al., Sec. III-A3, Fig. 5  ·  runs in the background, "
                     "overlapping mask generation with computation (Sec. IV-A, Fig. 10)")
    container(ax, 24, 18, 182, 122, "bernoulli_sampler.v", "u_stage2")

    # LFSR
    container(ax, 30, 84, 126, 50, "lfsr_128bit.v  —  128-bit 4-tap Fibonacci LFSR", "GEN_LFSRS[i].u_lfsr",
              color="#9334E6", fs=13.5)
    lbl(ax, 93, 125.8, "reset → DEFAULT_SEED   ·   load_seed → seed_in ^ i  (all-zero seed replaced)", "sub",
        fs=10.8, mono=False)
    lbl(ax, 34, 119.6, "r_state[127:0]  ←  {r_state[126:0], feedback}   (shift left every enabled cycle)",
        "ink", fs=11.3, ha="left")
    labels = ["R0", "R1", "R2", "…", "R98", "R99", "R100", "…", "R125", "R126", "R127"]
    xs = lfsr_chain(ax, 34, 108, 8.6, 7, labels)
    for i in range(len(xs) - 1):
        if labels[i] != "…" and labels[i + 1] != "…":
            wire(ax, [(xs[i] + 8.6, 111.5), (xs[i + 1], 111.5)], lw=1.6)
    xg, yg = 92, 96
    xor_gate(ax, xg, yg)
    for idx in (4, 6, 8, 10):
        cx = xs[idx] + 4.3
        side = 1 if cx > xg else -1
        wire(ax, [(cx, 108), (cx, 101), (xg + side * 2.4, 101), (xg + side * 1.6, yg + 1.6)], "ctrl")
    lbl(ax, xg, 91.4, "feedback = R127 ^ R125 ^ R100 ^ R98", fs=11)
    lbl(ax, xg, 88.6, "polynomial x^128 + x^126 + x^101 + x^99 + 1", fs=11)
    wire(ax, [(xg - 2.2, yg), (38.3, yg), (38.3, 108)], lw=1.8)
    lbl(ax, 48, 97.8, "feedback → R0", fs=10.3)
    wire(ax, [(xs[10] + 8.6, 111.5), (166, 111.5)], "data", "lfsr_bit_out", lpos=(158, 114.2), lw=2.0, fs=10.5)

    ax.add_patch(FancyBboxPatch((32, 74), 122, 7.5, boxstyle="round,pad=0,rounding_size=1", fc="#F3E8FD", ec="#9334E6",
                                lw=1.4, ls=(0, (4, 3)), alpha=0.7, zorder=2))
    ax.text(93, 77.7, "LFSR #2 … #N_lfsr  (N_LFSR ≤ 5, independent seeds seed_in ^ i)", ha="center", va="center",
            fontsize=11.5, color=INK, zorder=4)
    wire(ax, [(154, 77.7), (172, 77.7), (172, 100)], lw=1.8)

    block(ax, 166, 100, 30, 26, "Bit-wise AND", None, [
        "bernoulli_bit =",
        "  &lfsr_bits[N_LFSR-1:0]",
        "P(keep = 1) = 1 / 2^N_lfsr",
        "N_LFSR = 1 → p = 0.5",
    ], kind="alu", body_size=11, body_top=119.5)

    # SIPO
    wire(ax, [(181, 100), (181, 70), (133, 70), (133, 63)], "data", "bernoulli_bit (1-bit, serial)", lpos=(160, 72.4),
         lw=2.0, fs=10.5)
    block(ax, 104, 36, 58, 27, "Serial-In Parallel-Out", "sipo_shift_reg.v · u_sipo", [
        "shift_reg[63:0] ← {shift_reg[62:0], bit_in}",
        "bit_cnt[5:0] counts PF = 64 shifts",
        "64th shift: parallel_mask ← word,",
        "  word_valid pulses for 1 cycle",
    ], kind="ctl", body_size=11)
    wire(ax, [(104, 52), (82.2, 52)], "data")
    lbl(ax, 93, 54.4, "parallel_mask", fs=10)
    lbl(ax, 93, 49.6, "[63:0]", fs=10)
    wire(ax, [(104, 42), (82.2, 42)], "ctrl")
    lbl(ax, 93, 39.6, "word_valid → push", "ctrl", fs=10)

    # FIFO
    block(ax, 30, 36, 50, 32, "Mask FIFO (FWFT)", "mask_fifo.v · u_mask_fifo", [
        "MASK_FIFO_DEPTH = 64 × 64 b",
        "wr_ptr[5:0], rd_ptr[5:0], count[6:0]",
        "dout = empty ? 0 : mem[rd_ptr]",
        "  (head visible before the pop)",
        "full = (count == 64)",
        "empty = (count == 0)",
        "mask_valid = !mask_empty",
    ], kind="mem", stacked=2, body_size=10.8)

    # back-pressure
    wire(ax, [(55, 68), (55, 72), (27.8, 72), (27.8, 104), (30, 104)], "ctrl")
    ax.text(26.0, 88, "mask_full → en = sampler_en & !mask_full", rotation=90, va="center", ha="center",
            fontsize=9.8, color=CTRL_C, zorder=6)

    # host inputs
    port(ax, 3, 128, "Host", "ctrl", w=12)
    wire(ax, [(15, 128), (30, 128)], "ctrl", "seed_in [127:0]", lpos=(22, 131), fs=10.3)
    lbl(ax, 3, 122.6, "load_seed", "ctrl", fs=10.3, ha="left")
    lbl(ax, 3, 120.0, "sampler_en = 1", "ctrl", fs=10.3, ha="left")
    lbl(ax, 3, 112, "clk (220 MHz)", "ink", fs=11, ha="left")
    lbl(ax, 3, 109.2, "rst_n", "ink", fs=11, ha="left")

    # outputs and pop handshake
    wire(ax, [(30, 46), (16, 46), (16, 12), (214, 12)])
    ax.text(110, 14.4, "mask_out [63:0]   ·   mask_valid   ·   mask_count [6:0]", ha="center", fontsize=11.5,
            family=MONO, color=DATA_C, bbox=dict(fc="white", ec="none", pad=1), zorder=6)
    port(ax, 214, 12, "→ Stage 4 / 5", w=22)
    port(ax, 214, 30, "Stage 4 / 5 →", "ctrl", w=22)
    wire(ax, [(214, 30), (208, 30), (208, 22), (55, 22), (55, 33.8)], "ctrl", "mask_pop (arbitrated in bcnn_top)",
         lpos=(150, 24.4), fs=10.8)
    note(ax, 167, 36, 36, 22, [
        "1 Bernoulli bit / enabled cycle",
        "64-bit mask every 64 cycles",
        "Dropout engines pop one mask",
        "per PF-filter tile (mask_load);",
        "the FIFO keeps masks ahead.",
    ], fs=10.8, title="Timing")
    legend(ax)
    return save(fig, "stage2_bernoulli_sampler.png")


# ----------------------------------------------------------------------------
# 3. Stage 3 - Processing engine
# ----------------------------------------------------------------------------
def adder_tree_art(ax, x, y, w, h):
    levels = [("64×16b", 64), ("32×17b", 32), ("16×18b", 16), ("8×19b", 8), ("4×20b", 4), ("2×21b", 2), ("1×22b", 1)]
    step = w / (len(levels) - 1 + 0.6)
    for i, (lab, n) in enumerate(levels):
        cx = x + 1 + i * step
        hh = max(2.0, h * n / 64)
        ax.add_patch(FancyBboxPatch((cx, y + (h - hh) / 2), step * 0.45, hh, boxstyle="round,pad=0,rounding_size=0.3",
                                    fc="#FDE293", ec="#F29900", lw=1.3, zorder=5))
        lbl(ax, cx + step * 0.22, y - (2.0 if i % 2 == 0 else 4.6), lab, "ink", fs=9.5)
        if i < len(levels) - 1:
            ax.annotate("", xy=(cx + step, y + h / 2), xytext=(cx + step * 0.45, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#B06000", lw=1.2), zorder=6)


def stage3():
    fig, ax = canvas("Stage 3 — Processing Engine (NNE compute core)",
                     "rtl/stage3_pe_array/  ·  Fan et al., Sec. III-A2, Fig. 4  ·  PF = 64 processing units in lock-step, "
                     "each a PC = 64-wide MAC → accumulator → quantizer → ReLU")
    container(ax, 22, 22, 196, 118, "processing_engine.v  —  GEN_PUS: PF = 64 parallel processing units", "u_stage3")
    for k in range(3, 0, -1):
        ax.add_patch(FancyBboxPatch((28 + 1.6 * k, 28 - 1.6 * k), 184, 98, boxstyle="round,pad=0,rounding_size=2",
                                    fc="white", ec="#C4C7C5", lw=1.2, zorder=1))
    ax.add_patch(FancyBboxPatch((28, 28), 184, 98, boxstyle="round,pad=0,rounding_size=2", fc="white", ec="#80868B",
                                lw=1.8, zorder=1))
    ax.text(30, 122.6, "processing_unit.v  —  u_pu[f]  (one output filter, shown for filter f; ×64)", fontsize=14,
            fontweight="bold", color=INK)

    block(ax, 31, 74, 22, 40, "Slice f", None, [
        "pu_data_slice", "[511:0]", "pu_weight_slice", "[511:0]", "= bits", "f·512 +: 512", "(64 × INT8)",
    ], kind="rte", align="center", body_size=10.5, body_top=105)

    container(ax, 58, 58, 92, 60, "mac_unit.v  —  u_mac", None, color="#F29900", fs=13.5)
    block(ax, 61, 82, 30, 30, "Multiplier array", "u_mult_array", [
        "multiplier_array.v",
        "64 × signed 8b × 8b",
        "→ prod_vec [1023:0]",
        "(64 × 16 b)",
        "DSP mapping by synthesis;",
        "paper packs 2 INT8/DSP",
    ], kind="alu", body_size=10.3, title_size=13)
    block(ax, 96, 82, 51, 30, "Log2(PC) = 6-level adder tree", "u_adder_tree · adder_tree.v", [], kind="alu",
          title_size=12.5, body_size=10.5)
    adder_tree_art(ax, 98, 92.5, 44, 10)
    block(ax, 96, 62, 51, 15, "Output register", None, ["sum_out [21:0] (signed), valid_out = valid_in"],
          kind="ctl", body_size=10.5, title_size=12.5, body_top=67.4)
    wire(ax, [(91, 97), (96, 97)])
    wire(ax, [(121.5, 82), (121.5, 77)], "data", "tree_sum_comb [21:0]", lpos=(135, 79.5), fs=10)
    wire(ax, [(53, 97), (61, 97)])

    block(ax, 156, 86, 52, 30, "Window accumulator", "accumulator_32bit.v · u_accum", [
        "valid & !window_done:",
        "  running_sum += sum_in",
        "valid & window_done:",
        "  accum_out = running_sum + sum_in",
        "  running_sum ← 0, accum_valid = 1",
    ], kind="alu", body_size=10.5, title_size=13)
    wire(ax, [(147, 69.5), (152, 69.5), (152, 101), (156, 101)], "data", "sum_out [21:0]", lpos=(163, 82), fs=10)
    block(ax, 156, 50, 52, 30, "Linear quantizer", "linear_quantizer.v · u_quant", [
        "48-bit: acc × scale",
        "  + (bias <<< shift), then >>> shift",
        "saturate to [-128, +127]",
        "quant_scale[15:0], quant_shift[4:0],",
        "quant_bias[31:0]",
    ], kind="alu", body_size=10.5, title_size=13)
    wire(ax, [(182, 86), (182, 80)], "data", "accum_out [31:0]", lpos=(196, 83), fs=10)
    block(ax, 156, 31, 52, 15, "ReLU (bypassable)", "relu_unit.v · u_relu", [], kind="rte", title_size=13, body_size=10.5)
    ax.text(182, 34.2, "relu_en & sign ? 0 : data_in  (registered)", ha="center", fontsize=10.5, color=INK, zorder=5)
    wire(ax, [(182, 50), (182, 46)])
    lbl(ax, 184, 48, "quant_out [7:0]", fs=10, ha="left")

    block(ax, 61, 31, 86, 22, "Valid / window alignment", None, [
        "window_done_d1 ← window_done   (aligns with the mac sum_out register)",
        "relu_valid ← quant_valid   (feature_valid aligned with the registered ReLU)",
        "Latency: MAC 1 + ACC 1 + QUANT 1 + RELU 1 = 4 cycles",
    ], kind="ctl", body_size=10.5, title_size=12.5, body_top=44.5)

    ax.text(31, 66, "Bit-width progression", fontsize=12, fontweight="bold", color=INK)
    for i, t in enumerate(["8b × 8b → 16b", "→ 22b (tree)", "→ 32b (accum)", "→ 48b (scale)", "→ 8b (clamp)"]):
        lbl(ax, 31, 62.4 - i * 2.8, t, fs=10.8, ha="left")
        ax.texts[-1].set_color("#B06000")

    # inputs from Stage 1
    port(ax, 2, 112, "Stage 1 →", w=16)
    lbl(ax, 2, 106.4, "pe_data_in", fs=10.5, ha="left")
    lbl(ax, 2, 103.6, "[32767:0]", fs=10.5, ha="left")
    port(ax, 2, 92, "Stage 1 →", w=16)
    lbl(ax, 2, 86.4, "pe_weight_in", fs=10.5, ha="left")
    lbl(ax, 2, 83.6, "[32767:0]", fs=10.5, ha="left")
    wire(ax, [(18, 112), (31, 112)])
    wire(ax, [(18, 92), (31, 92)])
    port(ax, 2, 70, "Stage 1 →", "ctrl", w=16)
    lbl(ax, 2, 64.4, "valid_in", "ctrl", fs=10.3, ha="left")
    lbl(ax, 2, 61.8, "(re_b_valid)", "ctrl", fs=10.3, ha="left")
    lbl(ax, 2, 59.2, "window_done", "ctrl", fs=10.3, ha="left")
    wire(ax, [(18, 70), (24.5, 70), (24.5, 42), (61, 42)], "ctrl")

    # host configuration (right)
    port(ax, 220, 58, "Host cfg", "ctrl", w=16)
    wire(ax, [(220, 58), (215, 58), (215, 66), (208, 66)], "ctrl")
    wire(ax, [(215, 58), (215, 43), (208, 43)], "ctrl")
    for i, t in enumerate(["relu_en", "quant_scale", "quant_shift", "quant_bias"]):
        lbl(ax, 228, 52.4 - i * 2.6, t, "ctrl", fs=10)

    # output to Stage 4
    wire(ax, [(208, 38.5), (213, 38.5), (213, 16), (222, 16)])
    lbl(ax, 196, 18.6, "feature_out [7:0] ×64", fs=10)
    port(ax, 222, 16, "→ Stage 4", w=14)
    lbl(ax, 236, 10.9, "pe_features_out [511:0]", fs=10.5, ha="right")
    lbl(ax, 236, 8.3, "features_valid (= u_pu[0])", "ctrl", fs=10.3, ha="right")
    legend(ax, y=3.6)
    return save(fig, "stage3_processing_engine.png")


# ----------------------------------------------------------------------------
# 4. Stage 4 - Functional & dropout engine
# ----------------------------------------------------------------------------
def stage4():
    fig, ax = canvas("Stage 4 — Functional & Dropout Engine",
                     "rtl/stage4_functional/  ·  Fan et al., Sec. III-A2 (FE / DE), Sec. III-B2 Fig. 9 (shortcut), "
                     "Sec. III-B3 Fig. 3 (pooling), Sec. II-B1 Eq. 2 (MCD:  O = Y ⊙ M)")
    container(ax, 26, 52, 184, 74, "functional_engine.v", "u_stage4")

    block(ax, 32, 78, 48, 32, "Shortcut addition", "sc_addition_unit.v · u_sc_add", [
        "GEN_SC_ADDERS × 64, combinational",
        "raw_sum[8:0] = conv_val + sc_val",
        "clamped = sat(raw_sum, −128, +127)",
        "out = sc_en ? clamped : conv_val",
        "ResNet:  Out = Conv(X) + X",
        "valid_out = valid_in  (0 cycles)",
    ], kind="alu", body_size=10.8)
    block(ax, 94, 78, 48, 32, "2D spatial pooling", "pooling_unit_2d.v · u_pool", [
        "pool_mode[1:0]:",
        "  00 BYPASS → register, forward",
        "  01 MAX → max_regs[i] (signed)",
        "  10 AVG → sum_regs[i][9:0] >>> 2",
        "pool_step == 0 loads 1st pixel",
        "valid_out = pool_win_done (1 cycle)",
    ], kind="alu", body_size=10.8)
    block(ax, 160, 78, 46, 32, "Dropout engine (MCD)", "dropout_engine.v · u_dropout", [
        "mask_pop = mask_load & mcd_en",
        "           & mask_valid",
        "mask_reg[63:0] ← mask_in (per tile)",
        "active = mask_pop ? mask_in : mask_reg",
        "O[f] = active[f] ? Y[f] : 0",
        "mcd_en = 0 → pass-through (1 cycle)",
    ], kind="alu", body_size=10.6)

    wire(ax, [(80, 91), (94, 91)])
    lbl(ax, 87, 95.6, "sc_out_bus", fs=10)
    lbl(ax, 87, 93.2, "[511:0]", fs=10)
    wire(ax, [(142, 91), (160, 91)])
    lbl(ax, 151, 95.6, "pool_out", fs=10)
    lbl(ax, 151, 93.2, "[511:0]", fs=10)
    ax.text(90, 56.5, "SC addition (0 cycles) → 2D pooling (1 cycle) → dropout (1 cycle):  2-cycle pipeline",
            ha="center", fontsize=12, color=SUB, style="italic")

    # inputs
    port(ax, 2, 98, "Stage 3 →", w=17)
    lbl(ax, 2, 92.3, "conv_features_in", fs=10.3, ha="left")
    lbl(ax, 2, 89.6, "[511:0], valid_in", fs=10.3, ha="left")
    wire(ax, [(19, 98), (32, 98)])
    port(ax, 2, 83, "SC buffer →", w=17)
    lbl(ax, 2, 77.3, "sc_features_in", fs=10.3, ha="left")
    lbl(ax, 2, 74.6, "[511:0]", fs=10.3, ha="left")
    wire(ax, [(19, 83), (32, 83)])

    # controls from the top
    port(ax, 40, 141, "bcnn_top controller", "ctrl", w=34)
    wire(ax, [(57, 138.7), (57, 110)], "ctrl", "sc_en", lpos=(61, 117), fs=10.3)
    wire(ax, [(57, 132), (112, 132), (112, 110)], "ctrl")
    lbl(ax, 113.5, 119.5, "pool_mode[1:0], pool_step[1:0],", "ctrl", fs=10.3, ha="left")
    lbl(ax, 113.5, 116.9, "pool_win_done", "ctrl", fs=10.3, ha="left")
    wire(ax, [(112, 132), (166, 132), (166, 110)], "ctrl")
    lbl(ax, 167.5, 119.5, "mask_load", "ctrl", fs=10.3, ha="left")
    lbl(ax, 167.5, 116.9, "(= start_compute)", "ctrl", fs=10.3, ha="left")
    port(ax, 182, 141, "Stage 5 ctrl", "ctrl", w=22)
    wire(ax, [(193, 138.7), (193, 110)], "ctrl")
    lbl(ax, 194.5, 119.5, "mcd_en", "ctrl", fs=10.3, ha="left")

    # Stage 2 interface
    port(ax, 213, 104, "Stage 2 →", w=19)
    wire(ax, [(213, 104), (206, 104)])
    lbl(ax, 222.5, 98.6, "mask_in [63:0]", fs=10.3)
    lbl(ax, 222.5, 96.0, "mask_valid", "ctrl", fs=10.3)
    port(ax, 213, 84, "→ Stage 2", "ctrl", w=19)
    wire(ax, [(206, 84), (213, 84)], "ctrl")
    lbl(ax, 222.5, 78.6, "mask_pop", "ctrl", fs=10.3)

    # outputs
    wire(ax, [(183, 78), (183, 38), (213, 38)])
    port(ax, 213, 38, "→ DRAM / Stage 5", w=25)
    lbl(ax, 197, 32.4, "stage4_features_out [511:0], stage4_valid_out", fs=10.8)
    lbl(ax, 197, 29.6, "(egress to DRAM; layer-N output to output_reducer)", "sub", fs=10.3, mono=False)
    wire(ax, [(156, 91), (156, 38), (110, 38)])
    port(ax, 85, 38, "→ Stage 5 IC", w=25)
    lbl(ax, 97.5, 32.4, "premask_features_out [511:0], premask_valid_out", fs=10.8)
    lbl(ax, 97.5, 29.6, "pre-dropout tap: layer N-B is cached unmasked so every sample can be re-masked", "sub",
        fs=10.3, mono=False)
    legend(ax)
    return save(fig, "stage4_functional_engine.png")


# ----------------------------------------------------------------------------
# 5. Stage 5 - Cache & reduction (baseline)
# ----------------------------------------------------------------------------
def stage5():
    fig, ax = canvas("Stage 5 — Intermediate-layer Cache & Output Reduction",
                     "rtl/stage5_cache_reduction/  ·  Fan et al., Sec. IV-B, Fig. 11(c) (IC), Sec. II-B2 (partial Bayesian), "
                     "Sec. II-B Eq. 1 (predictive mean over S samples)")
    container(ax, 24, 14, 188, 126, "cache_reduction_engine.v", "u_stage5")

    block(ax, 30, 100, 62, 34, "MC sample controller", "mc_sample_controller.v · u_mc_ctrl", [
        "latches N, B (≤ N), S at run_start",
        "layer < N → layer + 1 on layer_done",
        "layer = N, s < S → s + 1, jump to N-B+1",
        "layer = N, s = S → inference_done",
        "ic_write_en = (layer == N-B) & (s == 1)",
        "ic_read_en  = (s > 1) & (layer == N-B+1)",
        "mcd_en = (N-B ≤ layer < N)",
        "is_final_layer = (layer == N)",
    ], kind="ctl", body_size=10.6, line_step=2.6)
    block(ax, 104, 96, 46, 30, "IC buffer", "ic_buffer.v · u_ic_buffer", [
        "on-chip dual-port BRAM",
        "IC_RAM_DEPTH = 1024 × 512 b",
        "1 pixel (64 × INT8) per row",
        "wr_en = ic_write_en & premask_valid",
        "rd ptr reset on every layer_done",
        "registered read (1 cycle)",
    ], kind="mem", stacked=2, body_size=10.5, line_step=2.6)
    block(ax, 164, 96, 44, 30, "Replay dropout", "dropout_engine.v · u_replay_dropout", [
        "mcd_en = ic_read_en",
        "mask_load = replay_mask_load",
        "fresh filter-wise mask M_s",
        "  for each sample s = 2..S",
        "O = cached Y ⊙ M_s",
        "latency: BRAM 1 + DE 1",
    ], kind="alu", body_size=10.5, line_step=2.6)
    wire(ax, [(152.2, 111), (164, 111)])
    lbl(ax, 158, 114.6, "rd_data", fs=10)
    lbl(ax, 158, 107.6, "[511:0]", fs=10)
    block(ax, 104, 44, 102, 32, "Output reducer", "output_reducer.v · u_reducer", [
        "Per channel f (GEN_REDUCE ×64):  sum[23:0] += x,  sum_sq[31:0] += x²  for each layer-N vector",
        "After sample S: bit-serial restoring dividers, 32 cycles, divisor S = num_samples",
        "  Mean[f] = trunc(Σx / S)",
        "  Var[f]  = floor(Σx² / S) − Mean[f]²   (never negative)",
        "FSM: ACCUM → LOAD → DIV → FINAL → DONE",
    ], kind="alu", body_size=10.8)

    # sample-1 write path
    port(ax, 2, 106, "Stage 4 →", w=16)
    lbl(ax, 2, 100.4, "premask_features_in", fs=9.8, ha="left")
    lbl(ax, 2, 97.8, "[511:0], premask_valid", fs=9.8, ha="left")
    wire(ax, [(18, 106), (22, 106), (22, 88), (98, 88), (98, 104), (104, 104)])
    badge(ax, 44, 91.2, "1", "Sample 1, layer N-B: cache pre-dropout output")
    wire(ax, [(92, 122), (104, 122)], "ctrl", "ic_write_en", lpos=(98, 124.6), fs=9.8)

    # replay path
    port(ax, 214, 130, "Stage 1 →", "ctrl", w=19)
    wire(ax, [(214, 130), (127, 130), (127, 126)], "ctrl", "ic_rd_req  (Stage 1 ingress dram_ready)",
         lpos=(170, 132.5), fs=10.3)
    wire(ax, [(208, 111), (214, 111)])
    port(ax, 214, 111, "→ Stage 1 ingress", w=24)
    lbl(ax, 226, 105.4, "replay_features [511:0]", fs=10)
    lbl(ax, 226, 102.8, "replay_valid", "ctrl", fs=10)
    badge(ax, 108, 89.6, "2", "Samples 2..S: replayed with a fresh mask")
    port(ax, 214, 84, "Stage 2 ↔", "ctrl", w=19)
    wire(ax, [(214, 84), (204, 84), (204, 96)], "ctrl")
    lbl(ax, 202.5, 86.6, "mask_in, replay_mask_pop", "ctrl", fs=10, ha="right")

    # reduction path
    port(ax, 2, 54, "Stage 4 →", w=16)
    lbl(ax, 2, 48.4, "stage4_features_in", fs=9.8, ha="left")
    lbl(ax, 2, 45.8, "[511:0], stage4_valid", fs=9.8, ha="left")
    wire(ax, [(18, 54), (104, 54)], "data", "sample_valid = is_final_layer & stage4_valid_in", lpos=(63, 57), fs=10)
    badge(ax, 40, 63.5, "3", "Layer N of every sample: accumulate, then reduce")
    wire(ax, [(206, 58), (214, 58)])
    port(ax, 214, 58, "→ Host", w=12)
    lbl(ax, 225, 52.4, "mean_prediction [511:0]", fs=10)
    lbl(ax, 225, 49.8, "uncertainty_score [1023:0]", fs=10)
    lbl(ax, 225, 47.2, "reduction_done", "ctrl", fs=10)

    # controller IO
    port(ax, 2, 130, "bcnn_top ↔", "ctrl", w=16)
    for i, t in enumerate(["start_inference ↓", "layer_done ↓", "N, B, S ↓", "busy ↑", "inference_done ↑"]):
        lbl(ax, 2, 124.6 - i * 2.5, t, "ctrl", fs=9.5, ha="left")
    wire(ax, [(18, 130), (30, 130)], "ctrl", both=True)
    wire(ax, [(34, 100), (34, 18), (214, 18)], "ctrl")
    port(ax, 214, 18, "→ Stage 4 / top", "ctrl", w=24)
    ax.text(150, 20.6, "mcd_en → Stage 4 dropout   ·   ic_read_en, bypass_feature_extractor → bcnn_top ingress mux",
            fontsize=10.3, family=MONO, color=CTRL_C, ha="center", bbox=dict(fc="white", ec="none", pad=1), zorder=6)
    note(ax, 40, 23, 58, 21, [
        "Layer executions: (N − B) + B·S  instead of  N·S",
        "Off-chip input traffic for samples 2..S: none",
        "Cached data is unmasked, so each sample",
        "  draws an independent mask (Sec. IV-B)",
    ], fs=10.5, title="Intermediate-layer Caching")
    legend(ax)
    return save(fig, "stage5_cache_reduction.png")


# ----------------------------------------------------------------------------
# 6. Full system
# ----------------------------------------------------------------------------
def stage6():
    fig, ax = canvas("BayesCNN Accelerator — System Integration",
                     "rtl/bcnn_top.v  ·  Fan et al., Fig. 4 (NNE + Bernoulli sampler + off-chip interface), "
                     "Sec. III-A2 (layer-by-layer), Sec. IV-A (overlapped sampling), Sec. IV-B (IC)")

    block(ax, 4, 26, 26, 110, "Host CPU &", None, [], kind="mem", title_size=13.5)
    ax.text(17, 130.2, "Off-chip DRAM", ha="center", fontsize=13.5, fontweight="bold", color=INK, zorder=4)
    for i, ln in enumerate(["layer inputs", "layer weights", "layer egress", "", "per-layer config:", "H, W, C_tiles,",
                            "W_tiles, KH, KW,", "KL, stride,", "relu_en, sc_en,", "pool_mode,", "quant_scale /",
                            "shift / bias", "", "network: N, B, S"]):
        ax.text(17, 124 - i * 2.9, ln, ha="center", fontsize=10.5, color=INK, zorder=4)

    container(ax, 38, 14, 198, 126, "bcnn_top.v", "dut")

    block(ax, 46, 108, 56, 26, "Layer controller FSM", None, [
        "S_IDLE → S_LAYER → S_INGRESS → S_ARM",
        "→ S_COMPUTE → S_DRAIN (8) → S_ADVANCE",
        "start_ingress / start_compute pulses,",
        "ping_pong_sel toggle, layer_done",
    ], kind="ctl", body_size=10.3, body_top=125.4)
    block(ax, 108, 108, 38, 26, "Glue logic", None, [
        "mask-pop arbitration",
        "weight recirculation",
        "pool_step / pool_win_done",
        "  sequencer (pool_cnt)",
    ], kind="rte", body_size=10.3, body_top=125.4)
    block(ax, 152, 108, 34, 26, "Stage 2", "Bernoulli sampler", [
        "LFSR → AND → SIPO",
        "→ mask FIFO",
        "(background)",
    ], kind="ctl", body_size=10.3)

    mux(ax, 44, 66, 6, 30, "ingress src")
    lbl(ax, 51.5, 63.6, "src_replay", "ctrl", fs=10, ha="left")
    block(ax, 56, 64, 36, 34, "Stage 1", "Smart buffers", [
        "smart_data_buffer",
        "  Ping-Pong BRAM, RAG",
        "smart_weight_buffer",
        "  4096 weight FIFOs",
    ], kind="mem", body_size=10.3)
    block(ax, 104, 64, 34, 34, "Stage 3", "Processing engine", [
        "64 PUs × 64 MACs",
        "→ accumulate",
        "→ quantize → ReLU",
    ], kind="alu", body_size=10.3)
    block(ax, 150, 64, 34, 34, "Stage 4", "Functional engine", [
        "SC add → pool",
        "→ dropout",
        "O = Y ⊙ M",
    ], kind="alu", body_size=10.3)
    block(ax, 150, 20, 80, 28, "Stage 5", "Cache & reduction", [
        "mc_sample_controller  ·  ic_buffer (layer N-B)",
        "u_replay_dropout  ·  output_reducer",
    ], kind="mem", body_size=10.5, body_top=33.5)

    # host <-> controller
    wire(ax, [(30, 128), (46, 128)], "ctrl", "start_inference", lpos=(38, 130.4), fs=9.3)
    wire(ax, [(46, 118), (30, 118)], "ctrl")
    lbl(ax, 38, 115.4, "busy,", "ctrl", fs=9.3)
    lbl(ax, 38, 113.0, "inference_done", "ctrl", fs=9.3)

    # input stream and weights
    wire(ax, [(30, 86), (44, 86)])
    lbl(ax, 37, 90.4, "dram_data_in", fs=9.3)
    lbl(ax, 37, 88.2, "[511:0]", fs=9.3)
    wire(ax, [(50, 81), (56, 81)])
    wire(ax, [(30, 58), (74, 58), (74, 64)])
    lbl(ax, 75.5, 60.6, "weight_din [32767:0], weight_push", fs=9.8, ha="left")

    # main dataflow
    wire(ax, [(92, 81), (104, 81)])
    lbl(ax, 98, 85.4, "pe_data /", fs=9.3)
    lbl(ax, 98, 83.2, "pe_weight", fs=9.3)
    lbl(ax, 98, 78.6, "[32767:0]", fs=9.3)
    wire(ax, [(138, 81), (150, 81)])
    lbl(ax, 144, 84.4, "[511:0]", fs=9.8)
    wire(ax, [(169, 108), (169, 98)], "data")
    lbl(ax, 170.5, 103, "mask_out [63:0]", fs=9.8, ha="left")
    wire(ax, [(160, 64), (160, 48)])
    lbl(ax, 158.5, 56, "premask [511:0]", fs=9.8, ha="right")
    wire(ax, [(176, 64), (176, 48)])
    lbl(ax, 177.5, 56, "stage4_out [511:0]", fs=9.8, ha="left")

    # egress to DRAM over the top
    wire(ax, [(184, 90), (212, 90), (212, 142.6), (17, 142.6), (17, 136)])
    ax.text(115, 142.6, "layer_features_out [511:0], layer_features_valid  →  DRAM egress (next layer's input)",
            fontsize=10.3, family=MONO, color=DATA_C, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1), zorder=6)

    # replay loop
    wire(ax, [(150, 30), (47, 30), (47, 66)], "loop")
    ax.text(98, 32.4, "replay_features [511:0]  (samples 2..S, layer N-B+1; layers 1..N-B bypassed)",
            fontsize=10.3, family=MONO, color=LOOP_C, ha="center", bbox=dict(fc="white", ec="none", pad=1), zorder=6)
    wire(ax, [(186, 122), (198, 122), (198, 48)], "ctrl")
    lbl(ax, 199.5, 72, "mask_out →", "ctrl", fs=9.3, ha="left")
    lbl(ax, 199.5, 69.6, "replay dropout", "ctrl", fs=9.3, ha="left")

    # results
    wire(ax, [(190, 20), (190, 9.5), (17, 9.5), (17, 26)])
    ax.text(105, 11.8, "mean_prediction [511:0]  ·  uncertainty_score [1023:0]  ·  reduction_done",
            fontsize=10.8, family=MONO, color=DATA_C, ha="center", bbox=dict(fc="white", ec="none", pad=1), zorder=6)

    # control fabric
    wire(ax, [(74, 108), (74, 98)], "ctrl")
    lbl(ax, 75.5, 101.0, "start_ingress / start_compute", "ctrl", fs=9.3, ha="left")
    wire(ax, [(146, 121), (152, 121)], "ctrl")
    lbl(ax, 149, 124.0, "pop", "ctrl", fs=9.3)
    wire(ax, [(112, 108), (112, 98)], "ctrl")
    lbl(ax, 113.5, 101.0, "pool_step", "ctrl", fs=9.3, ha="left")
    wire(ax, [(96, 108), (96, 104.6), (144, 104.6), (144, 40), (150, 40)], "ctrl", both=True)
    lbl(ax, 142.5, 50.4, "layer_done ↓", "ctrl", fs=9.3, ha="right")
    lbl(ax, 142.5, 48.0, "ic_read_en, mcd_en,", "ctrl", fs=9.3, ha="right")
    lbl(ax, 142.5, 45.6, "inference_done ↑", "ctrl", fs=9.3, ha="right")

    note(ax, 44, 17, 100, 9, [
        "Per layer: ingress (DRAM or IC replay) → compute → drain → advance; Stage 2 refills its FIFO in the background.",
    ], fs=10.3)
    legend(ax, y=3.6, extra="IC replay loop")
    return save(fig, "stage6_bcnn_top_system.png")


if __name__ == "__main__":
    for fn in (stage1, stage2, stage3, stage4, stage5, stage6):
        print("wrote", fn())
