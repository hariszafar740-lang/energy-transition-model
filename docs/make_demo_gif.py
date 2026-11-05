#!/usr/bin/env python3
"""Render docs/demo.gif from the committed workbench dataset.

Every number in the animation is read from ``workbench/data/etm_workbench_data.json``,
which ``etm-workbench build`` regenerates from solved PyPSA/HiGHS models. This script
only replays that file, it never invents a number:

* the dispatch panel draws ``flexibility.series[*].net_load`` and ``bess_discharge``
  together with ``reserve_requirement``;
* the price panel draws ``flexibility.series[*].lmp``;
* the right-hand panel compares ``flexibility.regimes[*].system_lcoe_per_mwh``
  for the four regimes and highlights the one on screen.

Usage
-----
    python3 docs/make_demo_gif.py                     # writes docs/demo.gif
    python3 docs/make_demo_gif.py --fps 10 --seconds-per-regime 10 --out /tmp/x.gif

Requires matplotlib (``pip install matplotlib``), nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "workbench" / "data" / "etm_workbench_data.json"

INK = "#101418"
MUTED = "#5b6770"
GRID = "#dde3e8"
BESS = "#2f9e6b"
LOAD = "#1d4ed8"
LMP = "#c2410c"
RES = "#7c3aed"

LABELS = {
    "1_Relaxed_NoReserves": "relaxed fleet, no reserves",
    "2_MustRun_And_Ramping": "must-run floors and ramping limits",
    "3_Inflexible_With_Reserves": "inflexible fleet plus spinning reserves",
    "4_Flexible_BESS_CoOptimized": "flexible fleet, BESS co-optimised",
}


def load():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    flex = data["flexibility"]
    regimes = {r["regime"]: r for r in flex["regimes"]}
    rows = []
    for s in flex["series"]:
        name = s["regime"]
        r = regimes.get(name, {})
        rows.append(
            {
                "name": name,
                "label": LABELS.get(name, name.split("_", 1)[-1].replace("_", " ").lower()),
                "net_load": list(s["net_load"]),
                "lmp": list(s["lmp"]),
                "reserve": list(s["reserve_price"]),
                "bess": list(s["bess_discharge"]),
                "lcoe": r.get("system_lcoe_per_mwh"),
                "max_lmp": r.get("max_lmp_per_mwh"),
                "curtail": r.get("vre_curtailment_pct"),
            }
        )
    order = [r["regime"] for r in flex["regimes"]]
    lcoes = [regimes[n]["system_lcoe_per_mwh"] for n in order if n in regimes]
    return data, rows, [regimes[n].get("system_lcoe_per_mwh") for n in order], lcoes


def draw(out, fps, per_regime, hold):
    data, rows, bar_values, _ = load()
    n = len(rows)
    hours = list(range(len(rows[0]["net_load"])))
    per_frames = max(10, int(fps * per_regime))
    hold_frames = max(2, int(fps * hold))
    total = n * per_frames + hold_frames

    ymax = max(max(r["net_load"]) for r in rows) * 1.14
    lmpmax = max(max(r["lmp"]) for r in rows) * 1.16
    lo = min(v for v in bar_values if v is not None) - 1.6
    hi = max(v for v in bar_values if v is not None) + 1.1

    fig = plt.figure(figsize=(8.0, 4.5), dpi=120)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.35, 1.0],
        height_ratios=[2.05, 1.0],
        wspace=0.27,
        hspace=0.42,
        left=0.075,
        right=0.977,
        top=0.775,
        bottom=0.105,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax3 = fig.add_subplot(gs[:, 1])
    for a in (ax, ax2, ax3):
        a.set_facecolor("white")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=7.2, length=3, width=0.7)
        a.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
        a.set_axisbelow(True)

    ax.set_xlim(-0.6, hours[-1] + 0.6)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("MW", color=MUTED, fontsize=7.4, labelpad=2)
    ax.set_xticklabels([])
    ax2.set_ylim(0, lmpmax)
    ax2.set_ylabel("LMP  /MWh", color=LMP, fontsize=7.4, labelpad=2)
    ax2.set_xlabel("hour of the representative day", color=MUTED, fontsize=7.4, labelpad=2)

    res_fill = ax.fill_between(
        hours, [0] * len(hours), [0] * len(hours), color=RES, alpha=0.14, linewidth=0
    )
    bess_fill = ax.fill_between(
        hours, [0] * len(hours), [0] * len(hours), color=BESS, alpha=0.34, linewidth=0
    )
    (load_line,) = ax.plot([], [], color=LOAD, linewidth=1.6, zorder=4)
    (res_line,) = ax.plot([], [], color=RES, linewidth=0.9, linestyle=(0, (4, 2.4)), zorder=3)
    (lmp_line,) = ax2.plot([], [], color=LMP, linewidth=1.3, zorder=3)
    sweep = [
        ax.axvline(0, color=INK, linewidth=0.9, alpha=0.5, zorder=6),
        ax2.axvline(0, color=INK, linewidth=0.9, alpha=0.5, zorder=6),
    ]

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LOAD),
        plt.Rectangle((0, 0), 1, 1, color=BESS, alpha=0.55),
        plt.Rectangle((0, 0), 1, 1, color=RES, alpha=0.35),
        plt.Rectangle((0, 0), 1, 1, color=LMP),
    ]
    ax.legend(
        handles,
        ["net load (MW)", "BESS discharge", "reserve requirement", "LMP"],
        loc="upper left",
        bbox_to_anchor=(-0.005, 1.005),
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handlelength=1.0,
        columnspacing=0.9,
        labelcolor=MUTED,
        borderpad=0.15,
        labelspacing=0.35,
    )

    fig.text(
        0.075,
        0.945,
        "Energy Transition Model",
        fontsize=13.0,
        color=INK,
        fontweight="bold",
        va="top",
    )
    fig.text(0.075, 0.895, "", fontsize=8.4, color=MUTED, va="top")
    sub = fig.text(0.075, 0.895, "", fontsize=8.4, color=MUTED, va="top")
    chip = fig.text(0.977, 0.945, "", fontsize=8.0, color=INK, va="top", ha="right")
    chip2 = fig.text(0.977, 0.902, "", fontsize=7.4, color=MUTED, va="top", ha="right")
    solves = data["meta"]["n_solves"]
    fig.text(
        0.977,
        0.028,
        f"flexibility regimes solved with PyPSA + HiGHS   ·   {solves} solves   ·   "
        "workbench/data/etm_workbench_data.json",
        fontsize=6.2,
        color=MUTED,
        ha="right",
    )

    names = []
    for r in rows:
        names.append(r["name"])
    ys = list(range(len(bar_values)))[::-1]
    ax3.set_xlim(0, 1)
    ax3.set_ylim(min(ys) - 0.65, max(ys) + 0.65)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.grid(False)
    for side in ("left", "bottom"):
        ax3.spines[side].set_visible(False)
    ax3.set_title(
        f"system LCOE by regime\n/MWh, axis starts at {lo:.0f}", fontsize=7.6, color=MUTED, pad=8
    )
    bars, dots, labels_txt = [], [], []
    for y in ys:
        b = ax3.barh(y, 0.0, height=0.34, color="#c9d3da")
        bars.append(b)
        (d,) = ax3.plot([], [], "o", markersize=5.2, color=MUTED)
        dots.append(d)
        labels_txt.append(ax3.text(0.02, y, "", fontsize=7.2, color=MUTED, va="center"))
    ax3.set_xlim(0, 1.45)
    base_left = 0.06

    def place(value):
        return base_left + (value - lo) / (hi - lo) * 1.1

    state = {"i": 0}

    def frame(_):
        idx = state["i"]
        state["i"] = (idx + 1) % total
        if idx >= n * per_frames:
            regime_i, prog = n - 1, 1.0
        else:
            regime_i = idx // per_frames
            prog = (idx % per_frames) / float(per_frames - 1)
        r = rows[regime_i]
        pos = prog * (len(hours) - 1)
        upto = int(pos) + 1
        load_line.set_data(hours[:upto], r["net_load"][:upto])
        lmp_line.set_data(hours[:upto], r["lmp"][:upto])
        res_line.set_data(hours[:upto], r["reserve"][:upto])
        bess_fill.set_data(
            hours, [0] * len(hours), [b if h < upto else 0 for h, b in enumerate(r["bess"])]
        )
        res_fill.set_data(
            hours, [0] * len(hours), [q if h < upto else 0 for h, q in enumerate(r["reserve"])]
        )
        for s in sweep:
            s.set_xdata([pos, pos])

        sub.set_text(f"regime {regime_i + 1} of {n}  ·  {r['label']}")
        bits = []
        if r["lcoe"] is not None:
            bits.append(f"LCOE {r['lcoe']:.2f} /MWh")
        if r["max_lmp"] is not None:
            bits.append(f"peak LMP {r['max_lmp']:.1f}")
        chip.set_text("   ·   ".join(bits))
        if r["curtail"] is not None:
            chip2.set_text(f"curtailment {r['curtail']:.2f} % of available VRE")

        for i, (bar, dot, txt, v) in enumerate(
            zip(bars, dots, labels_txt, bar_values, strict=True)
        ):
            active = i == regime_i
            left = place(v)
            bar[0].set_y(ys[i] - 0.17)
            bar[0].set_height(0.34)
            bar[0].set_x(base_left)
            bar[0].set_width(left - base_left)
            bar[0].set_color(BESS if active else "#cdd7de")
            bar[0].set_alpha(0.55 if active else 0.55)
            dot.set_data([left], [ys[i]])
            dot.set_color(BESS if active else MUTED)
            dot.set_markersize(5.6 if active else 3.8)
            txt.set_position((left + 0.035, ys[i]))
            txt.set_text(f"{v:.2f}")
            txt.set_color(INK if active else MUTED)
            txt.set_fontweight("bold" if active else "normal")
        return load_line, lmp_line, res_line

    writer = PillowWriter(fps=fps)
    anim = matplotlib.animation.FuncAnimation(
        fig, frame, frames=total, interval=1000.0 / fps, blit=False, repeat=False
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=writer)
    plt.close(fig)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo.gif"))
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seconds-per-regime", type=float, default=12.0)
    ap.add_argument("--hold", type=float, default=3.0)
    args = ap.parse_args()
    out = Path(args.out)
    frames = draw(out, args.fps, args.seconds_per_regime, args.hold)
    size_mb = out.stat().st_size / 1048576.0
    seconds = frames / float(args.fps)
    print(
        f"wrote {out}  ·  {frames} frames  ·  {seconds:.1f} s at {args.fps} fps  ·  "
        f"{size_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()
