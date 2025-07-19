import numpy as np
import casadi as ca
from scipy.spatial import Voronoi
from shapely import Polygon, Point
import matplotlib.pyplot as plt

def gauss_pdf(x, y, mean, covariance):

  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)
  return prob


def pdf_func(x, mean, covariance):
  coeff = 1 / ca.sqrt((2 * ca.pi) ** 2 * ca.det(covariance))
  # exponent = -0.5 * ((x[0] - mean[0])**2 + (x[1] - mean[1])**2)
  m = ca.reshape(mean, 1, 2)
  # expanded_mean = ca.repmat(m.T, x.shape[0], 1)
  # exponent = -0.5 * ((x - expanded_mean) @ ca.inv(covariance) @ ((x - expanded_mean)).T)
  mahalanobis_dist = []
  for i in range(x.shape[0]):
    diff = x[i, :] - mean
    mahalanobis_dist.append(diff @ ca.inv(covariance) @ diff.T)
  mahalanobis_dist = ca.vertcat(*mahalanobis_dist)
  exponent = -0.5 * mahalanobis_dist
  # print("Mahal shape: ", mahalanobis_dist.shape)
  # print("exponent shape: ", exponent.shape)
  # print("coeff shape: ", coeff.shape)
  return coeff * ca.exp(exponent)

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

def control_effort_cost(u, R):
  """
  u: control input (nu, T)
  R: weight matrix
  """
  costs = ca.vertcat(*[ca.mtimes([U[:,i].T, R, U[:,i]]) for i in range(u.shape[1])])
  cost = ca.sum1(costs)
  return cost

def collision_cost(x, x_obs, Ds, alpha=10.0, beta=5.0):
  """
  x: state [x, y]
  x_obs: obstacle state [x, y]
  Ds: safety distance
  """
  # dist = ca.sqrt(ca.sum1(x - x_obs)**2)
  # cost = 1 / (dist - Ds)
  dist_sq = ca.sumsqr(x - x_obs)
  cost = alpha * ca.exp(-beta * (dist_sq - Ds**2))

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


# 0. parameters
T = 10
dt = 0.1
sim_time = 10.0
NUM_STEPS = int(sim_time / dt)
np.random.seed(96)
# x1 = np.zeros(2)
# x2 = np.array([2.5, 3.0])
# mean = np.array([5.0, -2.5])
R = 5.0
r = R/2
AREA_W = 10.0
ROBOTS_NUM = 6
OBS_NUM = 5       # obstacles
Ds = 0.5          # safety distance to obstacles

nx = 3    # [x, y, theta]
nu = 2    # [v, omega]
# points = np.concatenate((x1.reshape(1, -1), x2.reshape(1, -1)),axis=0)
points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, nx)
points[:, 2] = np.random.rand(ROBOTS_NUM) * 2 * np.pi
# mean = -0.5*AREA_W + AREA_W * np.random.rand(2)
mean = np.array([-1.0, 2.5])
cov = 2.0 * np.diag([1.0, 1.0])
x_obs = -0.5*AREA_W + AREA_W * np.random.rand(OBS_NUM, 2)




# plotting stuff
xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, 100)
yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, 100)
X, Y = np.meshgrid(xg, yg)
Z = gauss_pdf(X, Y, mean, cov)
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 8))


def dynamics(state, ctrl):
  return state + ctrl * dt

def unicycle_dynamics(state, ctrl):
  """
  state: [x, y, theta]
  ctrl: [v, omega]
  """

  x = state[0]
  y = state[1]
  theta = state[2]
  v = ctrl[0]
  omega = ctrl[1]

  # Update the state using the unicycle model
  x_new = x + v * ca.cos(theta) * dt
  y_new = y + v * ca.sin(theta) * dt
  theta_new = theta + omega * dt

  # Wrap the angle to be within [-pi, pi]
  # theta_new = ca.atan2(ca.sin(theta_new), ca.cos(theta_new))
  state_new = ca.vertcat(x_new, y_new, theta_new)
  return state_new


robots_hist = np.zeros((NUM_STEPS+1, ROBOTS_NUM, nx))
robots_hist[0, :, :] = points
for s in range(NUM_STEPS):
  planned_trajectories = np.zeros((T+1, ROBOTS_NUM, nx))
  polygons = []
  positions_now = points.copy()
  # ------- Voronoi partitioning ---------
  dummy_points = np.zeros((5*ROBOTS_NUM, 2))
  dummy_points[:ROBOTS_NUM, :] = positions_now[:, :2]
  mirrored_points = mirror(positions_now[:, :2])
  mir_pts = np.array(mirrored_points)
  dummy_points[ROBOTS_NUM:, :] = mir_pts

  # Voronoi partitioning
  vor = Voronoi(dummy_points)
  
  for idx in range(ROBOTS_NUM):
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
    U = ca.SX.sym('U', nu, T)
    x0 = ca.SX.sym('x0', nx)
    obs_sym = ca.SX.sym('obs', 2)

    # 2. Dynamics: unicycle
    vmax = 1.5
    wmax = 2.0

    # 3. cost (coverage + control effort)
    coverage_cost_expr = voronoi_cost_func(x0[:2], qs, mean, cov)
    coverage_cost_fn = ca.Function('voronoi_cost', [x0[:2]], [coverage_cost_expr])
    # R_u = np.diag([0.01, 0.01])
    R_u = np.zeros(nu)
    u_cost_expr = control_effort_cost(U, R_u)
    u_cost_fn = ca.Function('u_cost', [U], [u_cost_expr])
    collision_cost_expr = collision_cost(x0[:2], obs_sym, Ds)
    collision_cost_fn = ca.Function('collision_cost', [x0[:2], obs_sym], [collision_cost_expr])

    # print("Cost function: ", cost_fn)

    # 4. Build optimizaiton problem
    obj = 0.0
    x_curr = x0
    g_list = []
    for k in range(T):
      # print("Planning step: ", k)
      obj += coverage_cost_fn(x_curr[:2]) + u_cost_fn(U[:, k])
      
      # obstacle avoidance constraint: Ds**2 - ||x - x_obs||**2 < 0
      for obs in x_obs:
        dist_squared = ca.sumsqr(x_curr[:2] - obs)
        # g_list.append(Ds**2 - dist_squared)
        # g_list.append(Ds - ca.sqrt(dist_squared))
        obj += collision_cost_fn(x_curr[:2], obs)
      x_curr = unicycle_dynamics(x_curr, U[:, k])

    # print("cost: ", obj)

    # 5. Constraints
    # Position (within env bounds)
    g_list.append(x_curr[0] - 0.5*AREA_W)
    g_list.append(x_curr[1] - 0.5*AREA_W)
    g_list.append(-x_curr[0] - 0.5*AREA_W)
    g_list.append(-x_curr[1] - 0.5*AREA_W)
    # Velocity
    g_list.append(x_curr[2] - vmax)
    g_list.append(x_curr[3] - vmax)
    g_list.append(-x_curr[2] - vmax)
    g_list.append(-x_curr[3] - vmax)
    # Stack all constraints
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

    opts = {'ipopt': {'print_level': 0, 'max_iter': 1000, 'tol': 1e-5}}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    x_init = positions_now[idx, :].copy()

    # 8. bounds and initial guess
    u_min = np.array([-vmax, -wmax])
    u_max = np.array([ vmax, wmax])
    lbx = np.tile(u_min, T)
    ubx = np.tile(u_max, T)
    
    # bounds on g: g <= 0
    lbg = -np.inf * np.ones(g.shape)
    ubg = np.zeros(g.shape)               # Ds**2 - ||x - x_obs||**2 < 0
    # ubg = np.inf * np.ones(g.shape)
    u0_guess = np.zeros(opt_vars.shape)

    # solve
    sol = solver(x0=u0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=x_init)
    u_opt = sol['x'].full().reshape(T, nu)
    print("uopt: ", u_opt[0])
    planned_traj = np.zeros((T+1, nx))
    planned_traj[0, :] = x_init
    for k in range(T):
      planned_traj[k+1, :] = unicycle_dynamics(planned_traj[k, :], u_opt[k, :]).full().flatten()
    planned_trajectories[:, idx, :] = planned_traj
    points[idx, :] = unicycle_dynamics(positions_now[idx, :], u_opt[0, :]).full().flatten()
    robots_hist[s+1, idx, :] = points[idx, :]
  
  ax.cla()
  ax.contourf(X, Y, Z.reshape(X.shape), levels=10, cmap='YlOrRd', alpha=0.75)
  for obs in x_obs:
    ax.scatter(obs[0], obs[1], marker='x', color='k', label='Obstacle')
    xc = obs[0] + Ds * np.cos(np.linspace(0, 2*np.pi, 20))
    yc = obs[1] + Ds * np.sin(np.linspace(0, 2*np.pi, 20))
    ax.plot(xc, yc, c='k', label='Safety distance')
  for idx in range(ROBOTS_NUM):
    ax.plot(robots_hist[:s+2, idx, 0], robots_hist[:s+2, idx, 1], label='Robot Trajectory', color='tab:blue')
    ax.scatter(robots_hist[s+1, idx, 0], robots_hist[s+1, idx, 1], color='tab:blue')
    # heading
    xh = points[idx, 0] + 0.5 * np.cos(points[idx, 2])
    yh = points[idx, 1] + 0.5 * np.sin(points[idx, 2])
    ax.plot([points[idx, 0], xh], [points[idx, 1], yh], color='tab:orange')
    ax.plot(planned_trajectories[:, idx, 0], planned_trajectories[:, idx, 1], label='Planned Trajectory', color='tab:green')
    x, y = polygons[idx].exterior.xy
    ax.plot(x, y, c='tab:red')
  
  # ax.legend()
  plt.pause(0.01)

plt.ioff()
plt.show()

