import numpy as np
import matplotlib.pyplot as plt
import yaml


def gauss_pdf(x, y, mean, covariance):
  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)
  return prob

def gmm_pdf(x, y, means, covariances, weights):
  prob = 0.0
  for i in range(len(means)):
    prob += weights[i] * gauss_pdf(x, y, means[i], covariances[i])
  return prob


np.random.seed(9)
COMPONENTS_NUM = 4
AREA_W = 20
means = -0.5*AREA_W + AREA_W * np.random.rand(COMPONENTS_NUM, 2)
covariances = []
for i in range(COMPONENTS_NUM):
  # cov = 2*np.random.rand(2, 2)
  # cov = cov @ cov.T  # Ensure positive semi-definite covariance
  cov = 5*np.eye(2) + 0.1 * np.random.rand(2, 2)
  covariances.append(cov)
# weights = np.random.dirichlet(np.ones(COMPONENTS_NUM)) 
weights = 1/COMPONENTS_NUM * np.ones(COMPONENTS_NUM)

data = {
    'means': means.tolist(),
    'covariances': [cov.tolist() for cov in covariances],
    'weights': weights.tolist()
}

# Write to YAML file
with open('config/config.yaml', 'a') as file:
    yaml.dump(data, file, default_flow_style=False)


COMPONENTS_NUM = 4
AREA_W = 20
means = -0.5*AREA_W + AREA_W * np.random.rand(COMPONENTS_NUM, 2)
covariances = []
for i in range(COMPONENTS_NUM):
  # cov = 2*np.random.rand(2, 2)
  # cov = cov @ cov.T  # Ensure positive semi-definite covariance
  cov = 5*np.eye(2) + 0.1 * np.random.rand(2, 2)
  covariances.append(cov)
weights = np.random.dirichlet(np.ones(COMPONENTS_NUM)) 

xg = np.linspace(-10, 10, 100)
yg = np.linspace(-10, 10, 100)
xx, yy = np.meshgrid(xg, yg)
Z = gmm_pdf(xx, yy, means, covariances, weights)
fig, ax = plt.subplots()
ax.contourf(xx, yy, Z.reshape(xx.shape), levels=10, cmap='YlOrRd', alpha=0.75)
plt.show()