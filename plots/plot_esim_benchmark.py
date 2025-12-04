#!/usr/bin/env python3
"""
Benchmark visualization for event camera simulator backends.

This script generates a comprehensive figure showing:
1. A fast ESIM implementation is necessary to avoid simulation bottlenecks
2. Neurosim ESIM is significantly faster than alternatives
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up publication-quality styling
plt.rcParams.update(
    {
        "font.size": 16,
        "font.family": "sans-serif",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# =============================================================================
# Data from benchmark results
# =============================================================================

backends = ["Neurosim\nESIM", "VID2E\nESIM", "Torch\nESIM", "AirSim\nESIM"]
backend_details = ["(CUDA)", "(CUDA)", "(GPU)", "(CPU)"]

# VGA resolution data (640x480)
vga_fps = [2269.14, 1252.88, 1101.67, 459.73]
vga_esim_avg_ms = [0.0325, 0.3205, 0.4231, 1.7895]
vga_esim_std_ms = [0.0064, 0.1931, 0.3331, 0.1681]

# HD resolution data (1280x720)
hd_fps = [1976.41, 1073.99, 931.63, 175.63]
hd_esim_avg_ms = [0.0438, 0.3661, 0.4968, 5.2808]
hd_esim_std_ms = [0.0030, 0.1964, 0.3324, 0.5339]

# Full HD resolution data (1920x1080)
fhd_fps = [1461.63, 910.17, 730.12, 81.98]
fhd_esim_avg_ms = [0.0734, 0.4282, 0.6619, 11.6464]
fhd_esim_std_ms = [0.0075, 0.1948, 0.3366, 1.2035]

# Colors
colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]  # Green, Blue, Purple, Red

# =============================================================================
# Create figure with 3 subplots
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax1, ax2, ax3 = axes

x = np.arange(len(backends))
width = 0.35

# -----------------------------------------------------------------------------
# Plot 1: ESIM latency comparison with slowdown annotations
# -----------------------------------------------------------------------------
bars1 = ax1.bar(
    x - width / 2,
    vga_esim_avg_ms,
    width,
    label="VGA (640×480)",
    color=colors,
    alpha=0.7,
    edgecolor="black",
    linewidth=1,
)
bars2 = ax1.bar(
    x + width / 2,
    hd_esim_avg_ms,
    width,
    label="HD (1280×720)",
    color=colors,
    alpha=1.0,
    edgecolor="black",
    linewidth=1,
    hatch="///",
)

# Add error bars
ax1.errorbar(
    x - width / 2,
    vga_esim_avg_ms,
    yerr=vga_esim_std_ms,
    fmt="none",
    color="black",
    capsize=3,
    capthick=1,
)
ax1.errorbar(
    x + width / 2,
    hd_esim_avg_ms,
    yerr=hd_esim_std_ms,
    fmt="none",
    color="black",
    capsize=3,
    capthick=1,
)

ax1.set_ylabel("ESIM Latency per call (ms)")
ax1.set_title("Event Simulator Latency per Call", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{b}\n{d}" for b, d in zip(backends, backend_details)])
ax1.legend(loc="upper left")
ax1.set_yscale("log")
ax1.set_ylim(0.01, 15)
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# Add slowdown annotations (relative to Neurosim ESIM)
for i in range(1, len(backends)):
    vga_slowdown = vga_esim_avg_ms[i] / vga_esim_avg_ms[0]
    hd_slowdown = hd_esim_avg_ms[i] / hd_esim_avg_ms[0]
    # Position annotations above the bars
    ax1.text(
        i - width / 2,
        vga_esim_std_ms[i] + vga_esim_avg_ms[i] + 0.1,
        f"{vga_slowdown:.0f}×",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
    )
    ax1.text(
        i + width / 2,
        hd_esim_std_ms[i] + hd_esim_avg_ms[i] + 0.1,
        f"{hd_slowdown:.0f}×",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

# Add "1×" baseline annotation for Neurosim
ax1.text(
    0 - width / 2,
    vga_esim_std_ms[0] + vga_esim_avg_ms[0] + 0.005,
    "1×",
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="bold",
    color="green",
)
ax1.text(
    0 + width / 2,
    hd_esim_std_ms[0] + hd_esim_avg_ms[0] + 0.005,
    "1×",
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="bold",
    color="green",
)

# -----------------------------------------------------------------------------
# Plot 2: Simulation FPS
# -----------------------------------------------------------------------------
bars1 = ax2.bar(
    x - width / 2,
    vga_fps,
    width,
    label="VGA",
    color=colors,
    alpha=0.7,
    edgecolor="black",
    linewidth=1,
)
bars2 = ax2.bar(
    x + width / 2,
    hd_fps,
    width,
    label="HD",
    color=colors,
    alpha=1.0,
    edgecolor="black",
    linewidth=1,
    hatch="///",
)

ax2.set_ylabel("End-to-End Simulation FPS")
ax2.set_title("End-to-End Simulation Speed", fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{b}\n{d}" for b, d in zip(backends, backend_details)])
ax2.legend(loc="upper right")
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.set_ylim(0, 2500)

# Add FPS values on bars
for i, (v, h) in enumerate(zip(vga_fps, hd_fps)):
    color = "green" if i == 0 else "black"
    ax2.text(i - width / 2, v + 50, f"{v:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=color)
    ax2.text(i + width / 2 + 0.05, h + 50, f"{h:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=color)

# -----------------------------------------------------------------------------
# Plot 3: Bottleneck visualization - ESIM as % of frame time
# -----------------------------------------------------------------------------

# Calculate ESIM time as percentage of total frame time
# Frame time = 1/FPS * 1000 (in ms)
vga_frame_time = [1000 / fps for fps in vga_fps]
hd_frame_time = [1000 / fps for fps in hd_fps]

vga_esim_pct = [
    esim / frame * 100 for esim, frame in zip(vga_esim_avg_ms, vga_frame_time)
]
hd_esim_pct = [esim / frame * 100 for esim, frame in zip(hd_esim_avg_ms, hd_frame_time)]

# Plot percentage bars
bars1 = ax3.bar(
    x - width / 2,
    vga_esim_pct,
    width,
    label="VGA",
    color=colors,
    alpha=0.7,
    edgecolor="black",
    linewidth=1,
)
bars2 = ax3.bar(
    x + width / 2,
    hd_esim_pct,
    width,
    label="HD",
    color=colors,
    alpha=1.0,
    edgecolor="black",
    linewidth=1,
    hatch="///",
)

ax3.set_ylabel("ESIM as % of Simulation Step Time")
ax3.set_title(
    "ESIM Contribution to Simulation Latency",
    fontweight="bold",
)
ax3.set_xticks(x)
ax3.set_xticklabels([f"{b}\n{d}" for b, d in zip(backends, backend_details)])
ax3.legend(loc="upper left")
ax3.grid(axis="y", alpha=0.3, linestyle="--")
ax3.set_ylim(0, 100)
ax3.set_yticks(np.arange(0, 100, 20))

# Add percentage values and color-coded annotations
for i, (v, h) in enumerate(zip(vga_esim_pct, hd_esim_pct)):
    # color_v = color_h = "green" if i == 0 else "black"
    ax3.text(
        i,
        max(v,h) + 1,
        f"~{(v+h)/2:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        color="green" if i == 0 else "black",
        fontweight="bold",
    )

# =============================================================================
# save
# =============================================================================

# Add subplot labels (A), (B), (C)
ax1.text(-0.1, 1.05, '(A)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')
ax2.text(-0.1, 1.05, '(B)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')
ax3.text(-0.1, 1.05, '(C)', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')

plt.tight_layout()

# Print key findings
textstr = "\n".join(
    [
        "Key Findings:",
        f"• Neurosim ESIM is {vga_esim_avg_ms[1] / vga_esim_avg_ms[0]:.0f}-{hd_esim_avg_ms[3] / hd_esim_avg_ms[0]:.0f}× faster than alternatives",
        f"• At HD, AirSim ESIM takes {hd_esim_pct[3]:.0f}% of frame time (severe bottleneck)",
        f"• Neurosim ESIM uses only {max(vga_esim_pct[0], hd_esim_pct[0]):.1f}% of frame time",
    ]
)
print(textstr)

plt.savefig(
    "esim_benchmark_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.savefig(
    "esim_benchmark_comparison.pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)

print("\nSaved: esim_benchmark_comparison.png")
print("Saved: esim_benchmark_comparison.pdf")
