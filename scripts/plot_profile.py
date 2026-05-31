"""Plot a simulation-loop profiling breakdown produced by ``Profiler.save``.

Usage:
    python scripts/plot_profile.py profiling/profile_apartment_1_<ts>.json
    python scripts/plot_profile.py <json> --out-dir profiling/plots

Produces four figures (saved next to the JSON, or in ``--out-dir``):
    1. <stem>_breakdown.png    — top-level mean-ms bar + total-time share pie
    2. <stem>_composition.png  — stacked mean-step bar + render_sensors-by-uuid
    3. <stem>_timeline.png     — per-call time series (raw + rolling mean)
    4. <stem>_hist.png         — per-section histograms

Sections are a hierarchy encoded as dotted paths (see Profiler): the first
component is a top-level loop bucket; deeper components decompose it, e.g.
``render_sensors.event_camera_1.event_kernel``. ``event_kernel`` (the event-sim
kernel proper) is highlighted throughout.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
HIGHLIGHT = "#E63946"  # the event-sim kernel — the section of interest

# Stable colors for top-level buckets and for known leaf kinds, so a thing
# looks the same in every figure. Per-uuid bars fall back to a palette.
TOP_COLORS = {
    "render_sensors": "#2C5F8A",
    "dynamics_step": "#C44E52",
    "viz_rerun": "#8C8C8C",
    "control_update": "#55A868",
    "log_h5": "#CCB974",
    "callback": "#8172B3",
}
LEAF_COLORS = {
    "habitat_render": "#4C92C3",
    "color2intensity": "#76C7E0",
    "event_kernel": HIGHLIGHT,
    "to_numpy": "#B7B7B7",
}
_FALLBACK = plt.cm.tab20(np.linspace(0, 1, 20))

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.4,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.edgecolor": "#333333",
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)


def _color(name, i=0):
    leaf = name.rsplit(".", 1)[-1]
    if name in TOP_COLORS:
        return TOP_COLORS[name]
    if leaf in LEAF_COLORS:
        return LEAF_COLORS[leaf]
    return _FALLBACK[i % len(_FALLBACK)]


def _leaf(name):
    return name.rsplit(".", 1)[-1]


def _style(ax):
    ax.tick_params(width=1.3, length=5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def _load(path):
    with open(path) as f:
        data = json.load(f)
    return data["sections"]


def _children(sections, parent):
    """Direct children of ``parent`` (``""`` = top-level), ordered by total."""
    if parent == "":
        kids = {n: s for n, s in sections.items() if "." not in n}
    else:
        depth = parent.count(".") + 1
        kids = {
            n: s
            for n, s in sections.items()
            if n.startswith(parent + ".") and n.count(".") == depth
        }
    return dict(
        sorted(kids.items(), key=lambda kv: kv[1]["total_time_ms"], reverse=True)
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_breakdown(sections, out):
    """Headline: where loop time goes (mean per call + share of total)."""
    top = _children(sections, "")
    names = list(top)
    means = [top[n]["avg_time_ms"] for n in names]
    totals = [top[n]["total_time_ms"] for n in names]
    colors = [_color(n, i) for i, n in enumerate(names)]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(16, 6.5))

    bars = ax_bar.barh(
        names, means, color=colors, edgecolor="white", linewidth=1.5, height=0.7
    )
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("mean time per call (ms)")
    ax_bar.set_title("Per-section mean time", pad=12)
    ax_bar.set_xlim(0, max(means) * 1.18 if means else 1)
    for bar, v in zip(bars, means):
        ax_bar.text(
            v + max(means) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )
    _style(ax_bar)

    _, _, autotexts = ax_pie.pie(
        totals,
        labels=[_leaf(n) for n in names],
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12, "fontweight": "bold"},
        pctdistance=0.78,
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(11)
    ax_pie.set_title("Share of total loop time", pad=12)

    fig.suptitle("Simulation loop breakdown", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)


def plot_composition(sections, out):
    """Stacked mean-step composition + render_sensors decomposed by uuid."""
    top = _children(sections, "")
    fig, (ax_step, ax_rs) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: stacked mean-step composition (top-level buckets).
    bottom = 0.0
    step_total = sum(s["avg_time_ms"] for s in top.values()) or 1.0
    for i, (name, s) in enumerate(top.items()):
        v = s["avg_time_ms"]
        ax_step.bar(
            "mean step",
            v,
            bottom=bottom,
            label=_leaf(name),
            color=_color(name, i),
            edgecolor="white",
            linewidth=1.5,
            width=0.5,
        )
        if v > 0.03 * step_total:
            ax_step.text(
                0,
                bottom + v / 2,
                f"{_leaf(name)}\n{v:.3f} ms",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )
        bottom += v
    ax_step.set_ylabel("time (ms)")
    ax_step.set_title("Mean step composition", pad=12)
    ax_step.set_xlim(-0.6, 0.6)
    ax_step.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _style(ax_step)
    ax_step.grid(False, axis="x")

    # Right: render_sensors total time, one stacked bar per uuid. Event cameras
    # are split into their kernels (event_kernel highlighted); other sensors are
    # a single segment. Uses total_time_ms = true contribution to the bucket.
    rs = "render_sensors"
    uuids = _children(sections, rs)
    legend_seen = {}
    if uuids:
        x = np.arange(len(uuids))
        for xi, (uuid_full, s) in enumerate(uuids.items()):
            subs = _children(sections, uuid_full)
            bottom = 0.0
            if subs:  # event camera: stack its kernels
                for j, (sub_full, ss) in enumerate(subs.items()):
                    leaf = _leaf(sub_full)
                    v = ss["total_time_ms"]
                    c = _color(sub_full, j)
                    ax_rs.bar(
                        xi,
                        v,
                        bottom=bottom,
                        color=c,
                        edgecolor="white",
                        linewidth=1.2,
                        width=0.72,
                        label=leaf if leaf not in legend_seen else None,
                    )
                    legend_seen[leaf] = True
                    bottom += v
            else:  # color / depth / imu: single segment
                v = s["total_time_ms"]
                ax_rs.bar(
                    xi,
                    v,
                    color=_color(uuid_full, xi + 5),
                    edgecolor="white",
                    linewidth=1.2,
                    width=0.72,
                )
                bottom = v
            ax_rs.text(
                xi,
                bottom,
                f"{s['total_time_ms']:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax_rs.set_xticks(x)
        ax_rs.set_xticklabels([_leaf(u) for u in uuids], rotation=20, ha="right")
        ax_rs.set_ylabel("total time (ms)")
        ax_rs.set_title(
            "render_sensors by sensor (uuid)  —  event cameras split into kernels",
            pad=12,
        )
        if legend_seen:
            ax_rs.legend(loc="upper right")
        _style(ax_rs)
    else:
        ax_rs.text(0.5, 0.5, "no render_sensors children", ha="center", va="center")
        ax_rs.axis("off")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _rolling(y, win):
    if win <= 1 or y.size < win:
        return y
    return np.convolve(y, np.ones(win) / win, mode="same")


def plot_timeline(sections, out, max_points=4000):
    """Per-call time series: raw (faint) + rolling mean (bold) per section."""
    have = {n: s["samples_ms"] for n, s in sections.items() if s.get("samples_ms")}
    if not have:
        return
    have = dict(sorted(have.items(), key=lambda kv: sum(kv[1]), reverse=True))

    fig, ax = plt.subplots(figsize=(15, 7.5))
    for i, name in enumerate(have):
        s = np.asarray(have[name])
        if s.size > max_points:
            idx = np.linspace(0, s.size - 1, max_points).astype(int)
            x, y = idx, s[idx]
        else:
            x, y = np.arange(s.size), s
        c = _color(name, i)
        ax.plot(x, y, color=c, linewidth=0.8, alpha=0.20)
        ax.plot(
            x,
            _rolling(y, max(1, y.size // 50)),
            color=c,
            linewidth=2.4,
            label=name,
            solid_capstyle="round",
        )
    ax.set_xlabel("call index")
    ax.set_ylabel("time (ms)")
    ax.set_title(
        "Per-call timing over the run  (faint = raw, bold = rolling mean)", pad=12
    )
    ax.set_yscale("log")
    ax.legend(ncol=2, loc="upper right", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_hist(sections, out):
    """Per-section histograms of call times."""
    have = {n: s["samples_ms"] for n, s in sections.items() if s.get("samples_ms")}
    if not have:
        return
    have = dict(sorted(have.items(), key=lambda kv: sum(kv[1]), reverse=True))
    names = list(have)
    n = len(names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.2 * cols, 3.4 * rows), squeeze=False
    )
    for i, (ax, name) in enumerate(zip(axes.ravel(), names)):
        s = np.asarray(have[name])
        ax.hist(
            s,
            bins=60,
            color=_color(name, i),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvline(
            s.mean(),
            color="#333333",
            linestyle="--",
            linewidth=1.8,
            label=f"mean {s.mean():.3f} ms",
        )
        ax.axvline(
            np.percentile(s, 99),
            color=HIGHLIGHT,
            linestyle=":",
            linewidth=1.8,
            label=f"p99 {np.percentile(s, 99):.3f} ms",
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("ms")
        ax.legend(fontsize=9)
        _style(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Per-section timing distributions", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot a profiling breakdown JSON.")
    ap.add_argument("json", type=str, help="Profiling JSON from Profiler.save")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PNGs (default: alongside the JSON)",
    )
    args = ap.parse_args()

    json_path = Path(args.json)
    sections = _load(json_path)

    out_dir = Path(args.out_dir) if args.out_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = json_path.stem

    plot_breakdown(sections, out_dir / f"{stem}_breakdown.png")
    plot_composition(sections, out_dir / f"{stem}_composition.png")
    plot_timeline(sections, out_dir / f"{stem}_timeline.png")
    plot_hist(sections, out_dir / f"{stem}_hist.png")

    print(f"Wrote plots to {out_dir}/{stem}_*.png")


if __name__ == "__main__":
    main()
