import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from pyplot_wrapper import ThreeD

"""Create function to plot"""
X = np.linspace(-5, 5, 500)
Y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(X, Y)

normal_dist = (
    lambda x, y, mu, sigma: 1
    / np.sqrt(2.0 * np.pi * sigma)
    * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    * 1
    / np.sqrt(2.0 * np.pi * sigma)
    * np.exp(-((y - mu) ** 2) / (2 * sigma**2))
)

Z = normal_dist(X, Y, 0, 1.75)

""" Plot Surface"""
ThreeD.init_3d_surf()
fig, axs = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(5, 5))

# Plot the surface.
surf = axs.plot_surface(
    X,
    Y,
    Z,
    linewidth=0,
    rstride=5,
    cstride=5,
    antialiased=False,
    cmap=cm.coolwarm,
    alpha=0.9,
)

axs.set(
    xlabel=r"$x$",
    ylabel=r"$y$",
    zlabel=r"$f(x,y)$",
    title=r"\textbf{Normal Distribution}",
)
# plt.legend()

""" Save Figure """
plt.tight_layout()
plt.savefig("pdf_plots/surf_test.pdf")
