import matplotlib.pyplot as plt
import seaborn as sns


def init_plot():
    """Setup matplotlib figure"""
    sns.set_style("whitegrid")
    palette = ["#F72585", "#7209B7", "#3A0CA3", "#4361EE", "#4CC9F0"]
    sns.set_palette(palette)
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Serif"],
        }
    )
