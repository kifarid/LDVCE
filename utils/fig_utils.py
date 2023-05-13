from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

WIDTH_PT = 397.48499 # NeurIPS width
LEGEND_FONTSIZE = 10
TICK_FONTSIZE = 7
LABEL_FONTSIZE = 10
TITEL_FONTSIZE = 10

def get_concat_h(*imgs):
    width = sum(img.width for img in imgs)
    height = imgs[0].height
    dst = Image.new('RGB', (width, height))
    place_w = 0
    for img in imgs:
        dst.paste(img, (place_w, 0))
        place_w += img.width
    return dst

def get_concat_v(*imgs):
    width = imgs[0].width
    height = sum(img.height for img in imgs)
    dst = Image.new('RGB', (width, height))
    place_h = 0
    for img in imgs:
        dst.paste(img, (0, place_h))
        place_h += img.height
    return dst

def set_size(width_pt=WIDTH_PT, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def save_fig(fig, filename, output_dir, dpi: int = 200):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    filename = filename.replace(" ", "_").replace("/", "")
    # fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight", dpi=dpi)
    fig.savefig(output_dir / f"{filename}.png", bbox_inches="tight", dpi=dpi)
    os.chmod(output_dir / f"{filename}.png", 0o777)
    print(f'Saved to "{output_dir}/{filename}.pdf"')

def set_general_plot_style(presentation: bool = False, use_tex: bool = True):
    """summary
    """
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("deep")
    plt.switch_backend("pgf")
    if presentation:
        plt.rcParams.update(
            {
                "text.usetex": use_tex,
                "pgf.texsystem": "pdflatex",
                "pgf.rcfonts": False,
                "font.family": "serif",
                "font.serif": [],
                "font.sans-serif": [],
                "font.monospace": [],
                "font.size": "10",
                "legend.fontsize": "9.90",
                "xtick.labelsize": "small",
                "ytick.labelsize": "small",
                "legend.title_fontsize": "small",
                "pgf.preamble": r"""
                    \usepackage[T1]{fontenc}
                    \usepackage[utf8x]{inputenc}
                    \usepackage{microtype}
                    \usepackage{mathptmx}
                """,
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": use_tex,
                "pgf.texsystem": "pdflatex",
                "pgf.rcfonts": False,
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
                "font.sans-serif": [],
                #"font.sans-serif": ["Times New Roman"],
                "font.monospace": [],
                "font.size": max(LEGEND_FONTSIZE, TICK_FONTSIZE, TITEL_FONTSIZE, LABEL_FONTSIZE),
                "legend.fontsize": LEGEND_FONTSIZE,
                "xtick.labelsize": TICK_FONTSIZE - 1,
                "ytick.labelsize": TICK_FONTSIZE - 1,
                "legend.title_fontsize": LEGEND_FONTSIZE,
                "pgf.preamble": r"""
                    \usepackage[utf8]{inputenc}
                    \usepackage[T1]{fontenc}
                    \usepackage{microtype}
                    \usepackage{nicefrac}
                    \usepackage{amsfonts}
                """,
            }
        )