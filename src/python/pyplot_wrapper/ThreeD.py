import matplotlib.pyplot as plt
import seaborn as sns


def init_3d_surf():
    """Setup matplotlib figure"""
    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("crest"))
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Serif"],
        }
    )
