import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""Create function to plot"""
x_series = np.linspace(-4, 5, 200)

# mu, sigma
normal_distributions = np.array(
    [
        [-2.0, 0.5],
        [0.0, 1.0],
        [2.0, 3.0],
    ]
)

normal_dist = (
    lambda x, mu, sigma: 1
    / np.sqrt(2.0 * np.pi * sigma)
    * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
)


""" Setup matplotlib figure"""
sns.set_style("whitegrid")
palette = ["#F72585", "#7209B7", "#3A0CA3", "#4361EE", "#4CC9F0"]
sns.set_palette(palette)
# sns.set_palette(sns.color_palette("crest", 3))
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Serif"],
    }
)

fig, axs = plt.subplots(1, 1)

for normal in normal_distributions:
    axs.plot(
        x_series,
        normal_dist(x_series, normal[0], normal[1]),
        label=("$\\mu=\\,$" + str(normal[0]) + ",\\ $\\sigma=\\,$ " + str(normal[1])),
    )

axs.set(
    xlabel=r"$x$",
    xticks=list(normal_distributions[:, 0]),
    ylabel=r"$f(x)$",
    title=r"\textbf{Normal Distribution}",
)
plt.legend()

""" Save Figure """
# sns.despine()     # remove unnecessary spines
plt.savefig("pdf_plots/test.pdf")
