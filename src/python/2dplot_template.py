import matplotlib.pyplot as plt
import numpy as np

from pyplot_wrapper import TwoD

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

""" Plotting """
TwoD.init_plot()
fig, axs = plt.subplots(1, 1, figsize=(5, 5))

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
plt.tight_layout()
plt.savefig("pdf_plots/test.pdf")
