import numpy as np
import casadi as ca
from scipy.spatial import Voronoi
from shapely import Polygon, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
    
def pdf_func(x, mean, covariance):
  coeff = 1 / ca.sqrt((2 * ca.pi) ** 2 * ca.det(covariance))
  # exponent = -0.5 * ((x[0] - mean[0])**2 + (x[1] - mean[1])**2)
  # m = ca.reshape(mean, 2, 1)
  # expanded_mean = ca.repmat(m.T, x.shape[0], 1)
  # exponent = -0.5 * ((x - expanded_mean) @ ca.inv(covariance) @ ((x - expanded_mean)).T)
  mahalanobis_dist = []
  for i in range(x.shape[0]):
    diff = x[i, :] - mean
    mahalanobis_dist.append(diff.T @ ca.inv(covariance) @ diff)
  mahalanobis_dist = ca.vertcat(*mahalanobis_dist)
  exponent = -0.5 * mahalanobis_dist
  # print("Mahal shape: ", mahalanobis_dist.shape)
  # print("exponent shape: ", exponent.shape)
  # print("coeff shape: ", coeff.shape)
  return coeff * ca.exp(exponent)

def human_pdf_func(x, mean ,cov):
  coeff = 1 / ca.sqrt((2 * ca.pi) ** 2 * ca.det(cov))
  diff = x - mean
  exponent = -0.5 * (diff.T @ ca.inv(cov) @ diff)
  return coeff * ca.exp(exponent)

def human_cost_func(x, mean, cov, alpha=0.1, eps=1e-6):
  diff = x - mean
  cost = diff.T @ ca.inv(cov) @ diff + eps
  return 1/(alpha*cost)  



def gmm_pdf_func(x, means, covariances, weights):
  prob = 0.0
  for i in range(len(means)):
    prob += weights[i] * pdf_func(x, means[i], covariances[i])
  return prob

def voronoi_cost(pi, q, mean, covariance):
  """
  pi: 2D position of the robot
  q: (N, 2) discretized points inside i-th Voronoi cell
  """

  dists = np.linalg.norm(pi - q, axis=1)
  pdfs = gauss_pdf(q[:, 0], q[:, 1], mean, covariance)
  cost = np.sum(np.square(dists) * pdfs)

  return -cost

def voronoi_cost_func(pi, q, mean, covariance):
  p = ca.reshape(pi, 1, q.shape[1])
  p = ca.repmat(p, q.shape[0], 1)
  sq_dists = ca.sum2((p - q)**2)
  pdfs = pdf_func(q, mean, covariance)
  cost = ca.sum1(sq_dists * pdfs)
  return cost

def gmm_voronoi_cost_func(pi, q, means, covariances, weights):
  p = ca.reshape(pi, 1, q.shape[1])
  p = ca.repmat(p, q.shape[0], 1)
  sq_dists = ca.sum2((p - q)**2)
  pdfs = gmm_pdf_func(q, means, covariances, weights)
  cost = ca.sum1(sq_dists * pdfs)
  return cost


def mirror(points):
    mirrored_points = []

    # Define the corners of the square
    square_corners = [(-0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, 0.5*AREA_W), (-0.5*AREA_W, 0.5*AREA_W)]

    # Mirror points across each edge of the square
    for edge_start, edge_end in zip(square_corners, square_corners[1:] + [square_corners[0]]):
        edge_vector = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])

        for point in points:
            # Calculate the vector from the edge start to the point
            point_vector = (point[0] - edge_start[0], point[1] - edge_start[1])

            # Calculate the mirrored point by reflecting across the edge
            mirrored_vector = (point_vector[0] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[0],
                               point_vector[1] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[1])

            # Translate the mirrored vector back to the absolute coordinates
            mirrored_point = (edge_start[0] + mirrored_vector[0], edge_start[1] + mirrored_vector[1])

            # Add the mirrored point to the result list
            mirrored_points.append(mirrored_point)

    return mirrored_points

def draw_ellipse(mean, cov, n_std=1, ax=None, edgecolor='tab:blue', facecolor='none', lw=2, alpha=1):
  # n_std: number of standard deviations (e.g., 1 for 68%, 2 for 95%, 3 for 99.7%)
  if ax is None:
    fig, ax = plt.subplots()
  
  # Compute eigenvalues and eigenvectors
  vals, vecs = np.linalg.eigh(cov)
  # Sort by eigenvalue size (descending)
  order = vals.argsort()[::-1]
  vals = vals[order]
  vecs = vecs[:, order]

  # Compute the angle of the ellipse
  theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

  # Width and height are 2 * sqrt(eigenvalue) * n_std
  width, height = 2 * n_std * np.sqrt(vals)

  ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                    edgecolor=edgecolor, facecolor=facecolor, linewidth=lw, alpha=alpha)
  
  ax.add_patch(ellipse)
  ax.scatter(mean[0], mean[1], marker='o', color=edgecolor, alpha=alpha)  # mark the center


# 0. parameters
np.random.seed(10)
T = 10
dt = 0.1
sim_time = 10.0
NUM_STEPS = int(sim_time / dt)
# x1 = np.zeros(2)
# x2 = np.array([2.5, 3.0])
# mean = np.array([5.0, -2.5])
R = 5.0
r = R/2
AREA_W = 10.0
ROBOTS_NUM = 6
OBS_NUM = 5       # obstacles
Ds = 0.5          # safety distance to obstacles
# points = np.concatenate((x1.reshape(1, -1), x2.reshape(1, -1)),axis=0)
points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 2)
# mean = -0.5*AREA_W + AREA_W * np.random.rand(2)
# cov = 2.0 * np.diag([1.0, 1.0])
x_obs = -0.5*AREA_W + AREA_W * np.random.rand(OBS_NUM, 2)
human_traj = np.zeros((NUM_STEPS+T, 3))
human_traj[0, :2] = -0.5*AREA_W + AREA_W*np.random.rand(2)
# human_traj[0, 2] = 2*np.pi * np.random.rand()  # initial heading
human_traj[0, 2] = 0.0
# human_covs = np.zeros((NUM_STEPS+T, 2, 2))
# for i in range(NUM_STEPS+T):
#   human_covs[i, :, :] = (i%10) * 0.2 * np.eye(2) + 0.1 * np.random.rand(2, 2)
alpha_human = 0.1
# Generate human trajectory (straight line bouncing off walls)
for i in range(1, NUM_STEPS+T):
  xn = human_traj[i-1, 0] + 0.1 * np.cos(human_traj[i-1, 2]) #+ np.random.normal(0, 0.1)
  yn = human_traj[i-1, 1] + 0.1 * np.sin(human_traj[i-1, 2]) #+ np.random.normal(0, 0.1)
  if xn < -0.5*AREA_W or xn > 0.5*AREA_W:
    human_traj[i, 0] = np.clip(xn, -0.5*AREA_W, 0.5*AREA_W)
    human_traj[i, 2] = np.pi - human_traj[i-1, 2]  # Reflect heading
  elif yn < -0.5*AREA_W or yn > 0.5*AREA_W:
    human_traj[i, 1] = np.clip(yn, -0.5*AREA_W, 0.5*AREA_W)
    human_traj[i, 2] = -human_traj[i-1, 2]
  else:
    human_traj[i, 0] = xn
    human_traj[i, 1] = yn
    human_traj[i, 2] = human_traj[i-1, 2] + np.random.normal(0, 0.1)

    

# GMM parameters
COMPONENTS_NUM = 4
# means = -0.5*AREA_W + AREA_W * np.random.rand(COMPONENTS_NUM, 2)
means = -0.0*AREA_W + 0.5*AREA_W * np.random.rand(COMPONENTS_NUM, 2)
covariances = []
for i in range(COMPONENTS_NUM):
  # cov = 2*np.random.rand(2, 2)
  # cov = cov @ cov.T  # Ensure positive semi-definite covariance
  cov = 1.75*np.eye(2) + 0.1 * np.random.rand(2, 2)
  covariances.append(cov)
weights = np.random.dirichlet(np.ones(COMPONENTS_NUM))  # Dirichlet distribution makes weights sum to 1
# print("weights: ", weights)


nx = 2
nu = 2

# plotting stuff
xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, 100)
yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, 100)
X, Y = np.meshgrid(xg, yg)
Z = gmm_pdf(X, Y, means, covariances, weights)
# plt.figure(figsize=(8, 8))
# plt.contourf(X, Y, Z.reshape(X.shape), levels=10, cmap='YlOrRd', alpha=0.75)
# plt.scatter(points[:, 0], points[:, 1], marker='o', color='tab:blue', label='Robots')
# plt.scatter(x_obs[:, 0], x_obs[:, 1], marker='x', color='k', label='Obstacles')
# plt.scatter(means[:, 0], means[:, 1], marker='*', color='tab:orange', label='GMM Means')
# plt.scatter(human_traj[0, 0], human_traj[0, 1], marker='^', color='tab:green', label='Human Start')
# plt.plot(human_traj[:, 0], human_traj[:, 1], color='tab:green', label='Human Trajectory')
# plt.xlim(-0.5*AREA_W, 0.5*AREA_W)
# plt.ylim(-0.5*AREA_W, 0.5*AREA_W)
# plt.show()
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 8))

def dynamics(state, ctrl):
  return state + ctrl * dt
  
wall_obs_num = int(0.6*AREA_W / Ds)*2
for i in range(wall_obs_num):
  x_obs = np.concatenate((x_obs, np.expand_dims(np.array([-1.5, -0.5*AREA_W + i*0.5*Ds]), 0)))
  x_obs = np.concatenate((x_obs, np.expand_dims(np.array([1.5, 0.5*AREA_W - i*0.5*Ds]), 0)))


robots_hist = np.zeros((NUM_STEPS+1, ROBOTS_NUM, 2))
robots_hist[0, :, :] = points
for s in range(NUM_STEPS):
  planned_trajectories = np.zeros((T+1, ROBOTS_NUM, nx))
  polygons = []
  positions_now = points.copy()



  for idx in range(ROBOTS_NUM):
    # ------- Voronoi partitioning ---------
    dummy_points = np.zeros((5*ROBOTS_NUM, 2))
    dummy_points[:ROBOTS_NUM, :] = positions_now
    mirrored_points = mirror(positions_now)
    mir_pts = np.array(mirrored_points)
    dummy_points[ROBOTS_NUM:, :] = mir_pts

    # Voronoi partitioning
    vor = Voronoi(dummy_points)

    region = vor.point_region[idx]
    poly_vert = []
    for vert in vor.regions[region]:
        v = vor.vertices[vert]
        poly_vert.append(v)

    poly = Polygon(poly_vert)

    # Limited range cell
    range_vert = []
    for th in np.arange(0, 2*np.pi, np.pi/10):
      vx = positions_now[idx, 0] + r * np.cos(th)
      vy = positions_now[idx, 1] + r * np.sin(th)
      range_vert.append((vx, vy))
    range_poly = Polygon(range_vert)
    lim_region = poly.intersection(range_poly)
    polygons.append(lim_region)
    robot = vor.points[idx]

    xmin, ymin, xmax, ymax = poly.bounds
    discr_points = 20
    qs = []
    for i in np.linspace(xmin, xmax, discr_points):
        for j in np.linspace(ymin, ymax, discr_points):
            pt_i = Point(i, j)
            if lim_region.contains(pt_i):
                qs.append(np.array([i, j]))

    qs = np.array(qs)
    # ---------------- End Voronoi partitioning ----------------

    # 1. Variables
    U = ca.SX.sym('U', nu+1, T)
    x0 = ca.SX.sym('x0', nx)
    m_var = ca.SX.sym('m', nx)
    c_var = ca.SX.sym('c', nx, nx)

    # 2. Dynamics: single integrator
    vmax = 1.5

    # 3. cost
    cost_expr = gmm_voronoi_cost_func(x0, qs, means, covariances, weights)
    cost_fn = ca.Function('cost', [x0], [cost_expr])
    human_cost_expr = human_cost_func(x0, m_var, c_var, alpha=alpha_human)
    human_cost_fn = ca.Function('human_cost', [x0, m_var, c_var], [human_cost_expr])

    # 4. Build optimizaiton problem
    obj = 0.0
    x_curr = x0
    g_list = []
    human_covs = np.zeros((T, 2, 2))
    human_preds = np.zeros((T, 3))
    Q = np.diag([0.1, 0.1, 0.01, 0.01])
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    pred_0 = human_traj[s]
    human_preds[0] = pred_0
    Q = 0.1 * np.eye(2)
    human_covs[0] = Q
    for i in range(1, T):
      human_preds[i, 0] = human_preds[i-1, 0] + 0.1 * np.cos(human_preds[i-1, 2])
      human_preds[i, 1] = human_preds[i-1, 1] + 0.1 * np.sin(human_preds[i-1, 2])
      human_preds[i, 2] = human_preds[i-1, 2]
      # human_covs[i, :, :] = (i+1) * 0.25 * np.eye(2) + 0.1 * np.random.rand(2, 2)
      human_covs[i, :, :] = np.eye(2) @ human_covs[i-1] @ np.eye(2).T + Q
    for k in range(T):
      # print("Planning step: ", k)
      obj += cost_fn(x_curr) #+ human_cost_fn(x_curr, human_traj[s+k, :2], human_covs[k])
      
      # obstacle avoidance constraint: Ds**2 - ||x - x_obs||**2 < 0
      for obs in x_obs:
        dist_squared = ca.sumsqr(x_curr - obs)
        g_list.append(Ds**2 - dist_squared)

      # Human avoidance constaint
      # Probabilistic Collision Checking With Chance Constraints, Du Toit et al. 2011 (T-RO)
      # diff = x_curr - human_traj[s+k, :2]
      diff = x_curr - human_preds[k, :2]
      mahalanobis_dist_sq = diff.T @ ca.inv(human_covs[k]) @ diff
      prob = 0.05  # Desired probability of collision
      coeff = ca.sqrt(ca.det(2*ca.pi * human_covs[k]))
      # vol = 4/3 * ca.pi * Ds**3
      vol = ca.pi * Ds**2
      g_list.append(-2* ca.log(coeff * prob / vol) - mahalanobis_dist_sq + U[2, k])
      w_k = 100 * (1 - k/T)
      obj += w_k * U[2, k]**2  # Add the slack variable to the cost


      
      x_curr = dynamics(x_curr, U[:nu, k])

    # print("cost: ", obj)

    # 5. Constraints
    # Position (within env bounds)
    g_list.append(x_curr[0] - 0.5*AREA_W)
    g_list.append(x_curr[1] - 0.5*AREA_W)
    g_list.append(-x_curr[0] - 0.5*AREA_W)
    g_list.append(-x_curr[1] - 0.5*AREA_W)
    g = ca.vertcat(*g_list)

    # 7. problem
    opt_vars = ca.vec(U)
    nlp = {
      'x': opt_vars,
      'f': obj,
      # 'g': ca.SX.zeros(1),
      'g': g,
      'p': x0
    }

    opts = {'ipopt': {'print_level': 0, 'sb': 'yes', 'max_iter': 1000, 'tol': 1e-5}, 'print_time': False}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    x_init = robot.copy()

    # 8. bounds and initial guess
    u_min = np.array([-vmax, -vmax, 0])
    u_max = np.array([ vmax, vmax, 0])
    lbx = np.tile(u_min, T)
    ubx = np.tile(u_max, T)
    
    # bounds on g: g <= 0
    lbg = -np.inf * np.ones(g.shape)
    ubg = np.zeros(g.shape)               # Ds**2 - ||x - x_obs||**2 < 0
    u0_guess = np.zeros(opt_vars.shape)

    # solve
    sol = solver(x0=u0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=x_init)
    u_opt = sol['x'].full().reshape(T, nu+1)
    u_opt, slack = u_opt[:, :2], u_opt[:, 2]
    # print("Slack variable: ", slack)
    # print("uopt: ", u_opt)
    planned_traj = np.zeros((T+1, nx))
    planned_traj[0, :] = x_init
    for k in range(T):
      planned_traj[k+1, :] = dynamics(planned_traj[k, :], u_opt[k, :])
    planned_trajectories[:, idx, :] = planned_traj
    points[idx, :] = dynamics(positions_now[idx, :], u_opt[0, :])
    robots_hist[s+1, idx, :] = points[idx, :]
  
  ax.cla()



  ax.contourf(X, Y, Z.reshape(X.shape), levels=10, cmap='YlOrRd', alpha=0.75)
  ax.plot(human_traj[:s+2, 0], human_traj[:s+2, 1], color='tab:purple', label='Human Trajectory')
  ax.scatter(means[:, 0], means[:, 1], marker='*', color='tab:orange', label='GMM Means')
  for t in range(T):
    alpha = np.exp(-np.log(10) * t / T)
    # draw_ellipse(human_traj[s+1+t, :], human_covs[t], n_std=1, ax=ax, alpha=alpha)
    draw_ellipse(human_preds[t, :2], human_covs[t], n_std=1, ax=ax, alpha=alpha)
  # ax.contourf(X, Y, alpha_human*human_pdf.reshape(X.shape), levels=10, cmap='Blues', alpha=0.5)
  for idx in range(ROBOTS_NUM):
    ax.plot(robots_hist[:s+2, idx, 0], robots_hist[:s+2, idx, 1], label='Robot Trajectory', color='tab:blue')
    ax.scatter(robots_hist[s+1, idx, 0], robots_hist[s+1, idx, 1], color='tab:blue')
    ax.plot(planned_trajectories[:, idx, 0], planned_trajectories[:, idx, 1], label='Planned Trajectory', color='tab:green')
    x, y = polygons[idx].exterior.xy
    ax.plot(x, y, c='tab:red')
  
  for obs in x_obs:
    ax.scatter(obs[0], obs[1], marker='x', color='k', label='Obstacle')
    xc = obs[0] + Ds * np.cos(np.linspace(0, 2*np.pi, 20))
    yc = obs[1] + Ds * np.sin(np.linspace(0, 2*np.pi, 20))
    ax.plot(xc, yc, c='k', label='Safety distance')
  # ax.legend()
  plt.pause(0.01)
  plt.xlim(-0.5*AREA_W, 0.5*AREA_W)
  plt.ylim(-0.5*AREA_W, 0.5*AREA_W)

plt.ioff()
plt.show()

