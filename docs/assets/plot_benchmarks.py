"""Draw the README result figures from docs/assets/benchmarks.json.

Usage: uv run --with matplotlib python docs/assets/plot_benchmarks.py
Writes one light and one dark SVG per figure next to this file. Each panel plots
round-trip latency against tokens per Attention GPU on log axes.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "benchmarks.json").read_text())

AFDEP, MASKED = "afdEP-comm", "Masked DeepEP"
# Across NVLink domains 1.0.0 ran the dense directional protocol; 1.1.0 runs it sparse.
DENSE, SPARSE = "Dense directional (1.0.0)", "Sparse directional"
MIXED = "Cross-node: mixed-role directional"
# Validated categorical slots (light, dark); a colour follows its entity in every
# figure.
SERIES = {
    AFDEP: ("#2a78d6", "#3987e5"),
    DENSE: ("#2a78d6", "#3987e5"),
    MASKED: ("#eb6834", "#d95926"),
    SPARSE: ("#1baf7a", "#199e70"),
    "Sparse dispatch only": ("#eda100", "#c98500"),
    "Stagger only": ("#e87ba4", "#d55181"),
    MIXED: ("#6250d6", "#9085e9"),
}
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}
FONT = ["system-ui", "-apple-system", "Segoe UI", "Helvetica", "Arial", "sans-serif"]
Y_TICKS = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000]


def colour(label, theme):
    return SERIES[label][0 if theme == "light" else 1]


def panel(ax, name, dataset, order, theme, headline):
    t = THEMES[theme]
    tokens = dataset["tokens"]
    for label in order:
        ax.plot(
            tokens,
            dataset["series"][label],
            color=colour(label, theme),
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            marker="o",
            markersize=8,
            markerfacecolor=colour(label, theme),
            markeredgecolor=t["surface"],
            markeredgewidth=2,
            zorder=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(tokens))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{x:,}" for x in tokens])
    lo = min(min(dataset["series"][label]) for label in order)
    hi = max(max(dataset["series"][label]) for label in order)
    ticks = [y for y in Y_TICKS if lo / 1.6 <= y <= hi * 1.6]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{y:,}" for y in ticks])
    ax.set_ylim(lo / (1.7 if headline and len(headline) > 2 else 1.35), hi * 1.25)
    ax.set_xlim(tokens[0] / 1.4, tokens[-1] * 1.4)
    ax.grid(True, which="major", color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0, pad=4)
    ax.set_title(
        name, loc="left", fontsize=10.5, color=t["ink"], fontweight="semibold", pad=8
    )
    ax.set_facecolor(t["surface"])
    if headline and all(label in dataset["series"] for label in headline[:2]):
        # One direct label per panel: the first series' change against the reference at
        # one token, above the upper of the two points (the series are lowest there).
        # The change is the paired geometric mean where the dataset has one.
        new, ref = headline[:2]
        below = len(headline) > 2 and headline[2] == "below"
        a, b = dataset["series"][new][0], dataset["series"][ref][0]
        paired = dataset.get("paired_ratio", {})
        if new in paired and ref in paired:
            change = (paired[new][0] / paired[ref][0] - 1) * 100
        else:
            change = (a / b - 1) * 100
        sign = "−" if change < 0 else "+"
        ax.annotate(
            f"{sign}{abs(change):.0f}% at 1 token",
            xy=(tokens[0], min(a, b) if below else max(a, b)),
            xytext=(-6, -17 if below else 11),
            textcoords="offset points",
            fontsize=9,
            color=t["ink2"],
        )


def figure(stem, title, subtitle, panels, order, cols, headline=None):
    rows = (len(panels) + cols - 1) // cols
    height = 2.9 * rows + 1.0
    for theme, t in THEMES.items():
        plt.rcParams.update(
            {"font.family": FONT, "svg.fonttype": "none", "svg.hashsalt": "afdep"}
        )
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(8.6 if cols == 1 else 3.7 * cols + 0.4, height),
            squeeze=False,
        )
        fig.patch.set_facecolor(t["surface"])
        for ax, (name, dataset) in zip(axes.flat, panels, strict=False):
            panel(
                ax,
                name,
                dataset,
                [label for label in order if label in dataset["series"]],
                theme,
                headline,
            )
        for ax in list(axes.flat)[len(panels) :]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("round trip, µs (log)", color=t["muted"], fontsize=9)
        for ax in axes[-1, :]:
            ax.set_xlabel(
                "tokens per Attention GPU (log)", color=t["muted"], fontsize=9
            )
        fig.text(
            0.012,
            1 - 0.12 / height,
            title,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="semibold",
            color=t["ink"],
        )
        fig.text(
            0.012,
            1 - 0.42 / height,
            subtitle,
            ha="left",
            va="top",
            fontsize=9.5,
            color=t["ink2"],
        )
        handles = [
            plt.Line2D(
                [],
                [],
                color=colour(label, theme),
                linewidth=2,
                marker="o",
                markersize=7,
                markeredgecolor=t["surface"],
                markeredgewidth=2,
            )
            for label in order
        ]
        fig.legend(
            handles,
            order,
            loc="upper left",
            bbox_to_anchor=(0.008, 1 - 0.62 / height),
            ncol=min(len(order), 3),
            frameon=False,
            fontsize=9,
            labelcolor=t["ink"],
            handlelength=1.6,
            # Browsers draw the labels in a wider UI font than matplotlib measures.
            columnspacing=3.0 if max(len(label) for label in order) > 30 else 1.4,
        )
        fig.tight_layout(
            rect=(0, 0, 1, 1 - (0.95 + 0.22 * ((len(order) - 1) // 3)) / height)
        )
        suffix = "" if theme == "light" else "-dark"
        fig.savefig(
            HERE / f"{stem}{suffix}.svg",
            facecolor=t["surface"],
            metadata={"Date": None},
        )
        plt.close(fig)


figure(
    "results-in-node-h100",
    "One NVLink domain, 4 × H100",
    "afdEP-comm against Masked DeepEP (DeepEP V2, role_aware=False); top-8 of 128"
    " experts (96 for 1A:3F); BF16 unless noted",
    list(DATA["in_node_h100"]["layouts"].items()),
    [AFDEP, MASKED],
    3,
    (AFDEP, MASKED),
)
figure(
    "results-in-node-h200",
    "One NVLink domain, 8 × H200",
    "afdEP-comm against Masked DeepEP; BF16, hidden 2048, top-8 of 128 experts",
    list(DATA["in_node_h200"]["layouts"].items()),
    [AFDEP, MASKED],
    3,
    (AFDEP, MASKED),
)
figure(
    "results-cross-node",
    "Across NVLink domains, H200 over InfiniBand (NCCL GIN)",
    "afdEP-comm 1.0.0 (dense directional) against Masked DeepEP; FP8, hidden 2048,"
    " top-8 of 128 experts",
    list(DATA["cross_node"]["layouts"].items()),
    [AFDEP, MASKED],
    2,
    (AFDEP, MASKED),
)

if "cross_node_proposal" in DATA:
    P = DATA["cross_node_proposal"]
    final = P["sparse_11to1"]["all_three"]
    parts = P["sparse_11to1"]["parts"]["stagger"]
    n = len(parts["tokens"])
    combined = {
        "tokens": parts["tokens"],
        "series": {
            DENSE: final["series"][DENSE][:n],
            MASKED: P["sparse_11to1"]["masked_unpaired"]["series"][MASKED],
            "Stagger only": parts["series"]["Stagger only"],
            "Sparse dispatch only": parts["series"]["Sparse dispatch only"],
            SPARSE: final["series"][SPARSE][:n],
        },
        "paired_ratio": {
            label: final["paired_ratio"][label][:n] for label in (DENSE, SPARSE)
        },
    }
    figure(
        "results-xn-sparse",
        "Sparse directional hybrid, 11:1 across 12 single-role domains (12 × 2 H200)",
        "Before the done witness (200944ce); each part paired against the dense"
        " protocol in its own screen, Masked from an earlier unpaired one",
        [("2 FFN + 22 Attention ranks, FP8, hidden 2048", combined)],
        [DENSE, MASKED, SPARSE, "Sparse dispatch only", "Stagger only"],
        1,
        (SPARSE, DENSE, "below"),
    )
    figure(
        "results-xn-scaling",
        "Sparse directional hybrid at other layouts",
        "Against the dense protocol of 1.0.0, paired, before the done witness; FP8,"
        " hidden 2048, top-8 of 128 experts",
        list(P["sparse_scaling"].items()),
        [DENSE, MASKED, SPARSE],
        3,
        (SPARSE, DENSE, "below"),
    )
    figure(
        "results-xn-mixed",
        "Mixed-role directional hybrid: the FFN domain also holds Attention GPUs",
        "Against afdEP-comm, which runs the union hybrid here and matches Masked within"
        " 0.5%; 6 × 4 H200, FP8, hidden 2048",
        list(P["mixed_role"].items()),
        [AFDEP, MASKED, MIXED],
        2,
        (MIXED, AFDEP),
    )
