from __future__ import annotations
import matplotlib as mpl


# Signature palette 

PALETTE = {
    "ink":      "#2C3E50",   # deep slate — primary lines, titles, axes
    "density":  "#5B8FB9",   # mid-blue — sequential heatmaps, primary distributions
    "tail":     "#E67E22",   # warm amber — percentile overlays, tail events
    "alert":    "#C0392B",   # restrained red — burst annotations, threshold breaches
    "paper":    "#FAFAF7",   # off-white background — reduces stark contrast
    "grid":     "#D5D8DC",   # cool light grey — recessive gridlines
    "muted":    "#7F8C8D",   # neutral mid-grey — secondary annotations
}

# Sequential colormap for density plots — single-hue blue ramp, no rainbow
DENSITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "jane_density",
    ["#FAFAF7", "#D6E4F0", "#A2C5E0", "#5B8FB9", "#2C5F8D", "#1A3A5C"]
)

# Stage-specific line colors — for multi-stage CCDF overlays
STAGE_COLORS = {
    "parse":      "#5B8FB9",   # density blue
    "queue":      "#E67E22",   # tail amber
    "end_to_end": "#2C3E50",   # ink — emphasizes the composite
}


def apply_style() -> None:
    """
    Install the Jane Street-style matplotlib defaults globally.
    Call once at the top of any analysis notebook.
    """
    mpl.rcParams.update({
        # Typography hierarchy
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Inter", "Helvetica Neue", "Helvetica",
                                  "Arial", "DejaVu Sans"],
        "font.size":            10,
        "axes.titlesize":       13,
        "axes.titleweight":     "semibold",
        "axes.titlepad":        12,
        "axes.titlelocation":   "left",     # left-aligned, like editorial captions
        "axes.labelsize":       10,
        "axes.labelweight":     "regular",
        "axes.labelpad":        6,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,
        "figure.titlesize":     15,
        "figure.titleweight":   "semibold",

        # Color application
        "text.color":           PALETTE["ink"],
        "axes.edgecolor":       PALETTE["muted"],
        "axes.labelcolor":      PALETTE["ink"],
        "axes.titlecolor":      PALETTE["ink"],
        "xtick.color":          PALETTE["muted"],
        "ytick.color":          PALETTE["muted"],

        # Spines — strip non-data ink
        "axes.linewidth":       0.7,
        "axes.spines.top":      False,
        "axes.spines.right":    False,

        # Ticks — outward, light
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     4,
        "ytick.major.size":     4,
        "xtick.major.width":    0.7,
        "ytick.major.width":    0.7,
        "xtick.major.pad":      4,
        "ytick.major.pad":      4,

        # Gridlines — present but recessive
        "axes.grid":            True,
        "axes.grid.axis":       "y",         # only horizontal — cleaner for distributions
        "grid.color":           PALETTE["grid"],
        "grid.linewidth":       0.5,
        "grid.linestyle":       "--",
        "grid.alpha":           0.7,
        "axes.axisbelow":       True,

        # Figure background — off-white
        "figure.facecolor":     PALETTE["paper"],
        "axes.facecolor":       PALETTE["paper"],
        "savefig.facecolor":    PALETTE["paper"],
        "savefig.edgecolor":    "none",
        "figure.dpi":           110,
        "savefig.dpi":          200,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.15,

        # Lines — slightly heavier than default for readability
        "lines.linewidth":      1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",

        # Legend — frameless, tight
        "legend.frameon":       False,
        "legend.borderaxespad": 0.5,
        "legend.handlelength":  1.6,
        "legend.handletextpad": 0.6,
    })


def annotate_value(ax, x, y, label, color=None, side="right", offset=(8, 0)):
    """
    Place a floating value label next to a data point. Used for percentile
    callouts on histograms and inline labels on CCDF reference lines.
    """
    color = color or PALETTE["ink"]
    ha = "left" if side == "right" else "right"
    ax.annotate(
        label, xy=(x, y),
        xytext=offset, textcoords="offset points",
        ha=ha, va="center",
        fontsize=9, color=color, weight="medium",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=PALETTE["paper"],
                  edgecolor="none", alpha=0.85),
    )