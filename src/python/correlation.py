import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from pyplot_wrapper import TwoD

""" Create datasets """
# Define parameters for datasets
means = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]  # Means for the datasets
covariances = [
    np.ones((2, 2)),  # perfect positive correlation
    np.array([[1, 0.9], [0.9, 1]]),  # positive correlation
    np.eye(2),  # identity covariance matrix
    np.array([[1, -0.9], [-0.9, 1]]),  # negative correlation
    np.array([[1, -1], [-1, 1]]),  # perfect negative correlation
]

# Number of samples per dataset
num_samples = 1000

""" Plotting """
TwoD.init_plot()
fig, axs = plt.subplots(1, 5, figsize=(15, 3))

for i, (mean, cov) in enumerate(zip(means, covariances), 0):
    # Generate random samples from multivariate normal distribution
    data = multivariate_normal.rvs(mean=mean, cov=cov, size=num_samples)
    correlation = np.corrcoef(data[:, 0], data[:, 1])[0, 1]

    axs[i].scatter(data[:, 0], data[:, 1], marker="o", s=3)
    axs[i].set(
        title="$r_{xy}=$ " + str(round(correlation, 3)), xticks=[], yticks=[]
    )
    axs[i].axis("off")
    axs[i].title.set_fontsize(30)

plt.tight_layout()

plt.savefig("pdf_plots/correlation.pdf")
