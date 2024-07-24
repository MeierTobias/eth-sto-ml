import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pyplot_wrapper import ThreeD

"""Create function to plot"""
X = np.linspace(-5, 5, 310)
Y = np.linspace(-5, 5, 300)
dx = X[1]-X[0]
dy = Y[1]-Y[0]
X, Y = np.meshgrid(X, Y)

mu_x = 0
mu_y = 0
sigma_x = 1.7
sigma_y = 0.8

marginal_plot = False

cond_x_index = 170
cond_x = X[0, cond_x_index]

normal_dist = (
    lambda x, y: 1
    / np.sqrt(2.0 * np.pi * sigma_x)
    * np.exp(-((x - mu_x) ** 2) / (2 * sigma_x**2))
    * 1
    / np.sqrt(2.0 * np.pi * sigma_y)
    * np.exp(-((y - mu_y) ** 2) / (2 * sigma_y**2))
)

marginal_norm_dist_x = (
    lambda x, y_range, normal_dist_func:
    np.sum(normal_dist_func(x, y_range), axis=0) * dy
)

marginal_norm_dist_y = (
    lambda y, x_range, normal_dist_func:
    np.sum(normal_dist_func(x_range, y), axis=1) * dx
)

Z = normal_dist(X, Y)
marg_X = marginal_norm_dist_x(X, Y, normal_dist)
marg_Y = marginal_norm_dist_y(Y, X, normal_dist)

cond_dist = Z[:, cond_x_index]/marg_X[cond_x_index]

""" Multidimensional distribution"""
ThreeD.init_3d_surf()
fig, axs = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(7.5, 5))
axs.set_box_aspect([1, 1, 0.4])

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
    zorder=1
)

axs.set(
    xlabel=r"$x$",
    ylabel=r"$y$",
    zlabel=r"$f(x,y)$",
    # title=r"\textbf{Normal Distribution}",
)
# plt.legend()

plt.tight_layout(pad=0.0, rect=(0, -0.1, 0.9, 1.17))
axNEW = axs
fig.show()
# fig.savefig("pdf_plots/multi_dist.pdf")

if marginal_plot:
    """ Marginals of Multi. Dist."""

    marg_X_plot = axs.plot(
        X[0, :],
        marg_X,
        zdir="y",
        zs=Y.max(),
        color='firebrick'
    )
    margin_X_poly = [(X[0, :][i], Y.max(), marg_X[i]) for i in range(len(X[0, :]))] + [(X[0, :].max(), Y.max(), 0), (X[0, :].min(), Y.max(), 0)]
    axs.add_collection3d(Poly3DCollection([margin_X_poly], color='firebrick'))
    axs.text(0, Y.max(), marg_X.max()*1.4, r'$f_X (x)$')

    marg_Y_plot = axs.plot(
        Y[:, 0],
        marg_Y,
        zdir="x",
        zs=X.min(),
        color='firebrick'
    )
    margin_Y_poly = [(X.min(), Y[:, 0][i], marg_Y[i]) for i in range(len(Y[:, 0]))] + [(X.min(), Y[:, 0].max(), 0), (X[0, :].min(), X.min(), 0)]
    axs.add_collection3d(Poly3DCollection([margin_Y_poly], color='firebrick'))
    axs.text(X.min(), 0, marg_Y.max()*1.2, r'$f_Y (y)$')

    fig.show()
    # fig.savefig("pdf_plots/multi_dist_marg.pdf")
else:
    """ Conditional of Multi. Dist."""

    cond_plot = axs.plot(
        Y[:, cond_x_index],
        cond_dist,
        zdir="x",
        zs=cond_x,
        color='firebrick',
        zorder=10
    )
    cond_poly = [(cond_x, Y[:, cond_x_index][i], cond_dist[i]) for i in range(len(Y[:, cond_x_index]))] + [(cond_x, Y[:, cond_x_index].max(), 0), (cond_x, Y[:, cond_x_index].min(), 0)]
    coll = Poly3DCollection([cond_poly], color='firebrick', zorder=20)
    # unfortunatly the zorder does not work correctly for polygons https://github.com/matplotlib/matplotlib/issues/3894
    axs.add_collection3d(coll)
    axs.text(cond_x, 0, cond_dist.max() * 1.2, r'$f(y|x = ' + str(np.round(cond_x,1)) + r') = \frac{f(x,y)}{f_X(x)}$')

    fig.show()
    fig.savefig("pdf_plots/multi_dist_cond.pdf")
