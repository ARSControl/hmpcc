import numpy as np
import casadi as ca
from scipy.spatial import Voronoi
from shapely import Polygon, Point
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as patches

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

def collision_cost_func(pi, x_obs, Ds, alpha=0.1):
  p = ca.reshape(pi, 1, x_obs.shape[1])
  p = ca.repmat(p, x_obs.shape[0], 1)
  dist = ca.sqrt(ca.sum2(p - x_obs)**2)
  return ca.sum1(ca.exp(-alpha * dist))
  


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

def dynamics(state, ctrl):
  px = state[0]
  py = state[1]
  vx = state[2]
  vy = state[3]
  ax = ctrl[0]
  ay = ctrl[1]
  px_dot = vx
  py_dot = vy
  vx_dot = ax
  vy_dot = ay
  x_dot = ca.vertcat(px_dot, py_dot, vx_dot, vy_dot)
  return state + x_dot * dt

def single_int_dynamics(state, ctrl):
  return state + ctrl * dt


# 0. parameters
np.random.seed(42)
T = 10
dt = 0.05
sim_time = 15.0
NUM_STEPS = int(sim_time / dt)
GRAPHICS_ON = True
NUM_EPISODES = 10
# x1 = np.zeros(2)
# x2 = np.array([2.5, 3.0])
# mean = np.array([5.0, -2.5])
nx = 2
nu = 2
R = 5.0
r = R/2
AREA_W = 10.0
ROBOTS_NUM = 6
HUMANS_NUM = 0
OBS_NUM = 0       # obstacles
Ds = 0.5          # safety distance to obstacles
colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:pink", "tab:olive"]
coverage_over_time = np.zeros((NUM_EPISODES, NUM_STEPS))

eval_path = f"results/maze_eval_{ROBOTS_NUM}r.txt"
with open(eval_path, "w") as f:
  f.write("It\tNh\tColl\tA\tH\n")
collisions = np.zeros(NUM_EPISODES)               # check for collisions with humans / obstacles
effectiveness = np.zeros(NUM_EPISODES)            # Coverage effectiveness  
coverage_func = np.zeros(NUM_EPISODES)            # range-unlimited Coverage function 
wall_obs_num = int(0.6*AREA_W / Ds)*2
for ep in range(NUM_EPISODES):
  x_obs = -0.5*AREA_W + AREA_W * np.random.rand(OBS_NUM, 2)
  for i in range(wall_obs_num):
    x_obs = np.concatenate((x_obs, np.expand_dims(np.array([-1.5, -0.5*AREA_W + i*0.5*Ds]), 0)))
    x_obs = np.concatenate((x_obs, np.expand_dims(np.array([1.5, 0.5*AREA_W - i*0.5*Ds]), 0)))
  # points = np.concatenate((x1.reshape(1, -1), x2.reshape(1, -1)),axis=0)
  # points = -0.4*AREA_W + 0.8*AREA_W * np.random.rand(ROBOTS_NUM, nx)
  # mean = -0.5*AREA_W + AREA_W * np.random.rand(2)
  # cov = 2.0 * np.diag([1.0, 1.0])
  human_traj = np.zeros((NUM_STEPS+T, HUMANS_NUM, 3))
  human_traj[0, :, :2] = -0.5*AREA_W + AREA_W*np.random.rand(HUMANS_NUM, 2)
  # human_traj[0, 2] = 2*np.pi * np.random.rand()  # initial heading
  human_traj[0, :, 2] = 2*np.pi * np.random.rand(HUMANS_NUM)

  points = []
  while len(points) < ROBOTS_NUM:
      candidate = -0.4 * AREA_W + 0.5 * AREA_W * np.random.rand(1, nx)
      pos = candidate[0, :2]  # x, y

      obs_dists = np.linalg.norm(x_obs - pos, axis=1)
      h_dists = np.linalg.norm(human_traj[0, :, :2] - pos, axis=1)
      if np.all(obs_dists > 3*Ds) and np.all(h_dists > 3*Ds):
          points.append(candidate[0])

  points = np.array(points)  # shape (ROBOTS_NUM, nx)
  points[:, 2:] = 0.0
  # human_covs = np.zeros((NUM_STEPS+T, 2, 2))
  # for i in range(NUM_STEPS+T):
  #   human_covs[i, :, :] = (i%10) * 0.2 * np.eye(2) + 0.1 * np.random.rand(2, 2)
  alpha_human = 0.1
  # Generate human trajectory (straight line bouncing off walls)
  for h in range(HUMANS_NUM):
    for i in range(1, NUM_STEPS+T):
      x_prev, y_prev, theta_prev = human_traj[i-1, h]
      xn = x_prev + 0.1 * np.cos(theta_prev) #+ np.random.normal(0, 0.1)
      yn = y_prev + 0.1 * np.sin(theta_prev) #+ np.random.normal(0, 0.1)
      theta_new = theta_prev
      if xn < -0.5*AREA_W or xn > 0.5*AREA_W:
        xn = np.clip(xn, -0.5*AREA_W, 0.5*AREA_W)
        theta_new = np.pi - theta_prev
      if yn < -0.5*AREA_W or yn > 0.5*AREA_W:
        yn = np.clip(yn, -0.5*AREA_W, 0.5*AREA_W)
        theta_new = -theta_new
      
      if -0.5*AREA_W < xn < 0.5*AREA_W and -0.5*AREA_W < yn < 0.5*AREA_W:  
        theta_new += np.random.normal(0, 0.1)

      human_traj[i, h] = [xn, yn, theta_new]


  # GMM parameters
  COMPONENTS_NUM = 4
  means = -0.5*AREA_W + AREA_W * np.random.rand(COMPONENTS_NUM, 2)
  means[0, :] = np.array([2.5, -1.5])
  means[1, :] = np.array([3.5, 0])
  means[2, :] = np.array([3.5, 2.5])
  means[3, :] = np.array([3.5, 2.5])
  covariances = []
  for i in range(COMPONENTS_NUM):
    # cov = 2*np.random.rand(2, 2)
    # cov = cov @ cov.T  # Ensure positive semi-definite covariance
    cov = 1.75*np.eye(2) + 0.1 * np.random.rand(2, 2)
    covariances.append(cov)
  weights = np.random.dirichlet(np.ones(COMPONENTS_NUM))  # Dirichlet distribution makes weights sum to 1
  # print("weights: ", weights)




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
  if GRAPHICS_ON:
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))



  robots_hist = np.zeros((NUM_STEPS+1, ROBOTS_NUM, 2))
  robots_hist[0, :, :] = points[:, :2]
  for s in range(NUM_STEPS):
    planned_trajectories = np.zeros((T+1, ROBOTS_NUM, nx))
    polygons = []
    positions_now = points[:, :2].copy()

    # Check collisions
    diffs = positions_now[:, np.newaxis, :] - x_obs[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    coll = np.any(dists < Ds)
    collisions[ep] = collisions[ep] or coll
    if coll:
      print("[WARNING] Collision with obstacle detected")

    diffs = positions_now[:, np.newaxis, :] - human_traj[s, np.newaxis, :, :2]
    dists = np.linalg.norm(diffs, axis=2)
    coll = np.any(dists < Ds)
    collisions[ep] = collisions[ep] or coll
    if coll:
      print("[WARNING] Collision with human detected")


    # ------- Voronoi partitioning ---------
    dummy_points = np.zeros((5*ROBOTS_NUM, 2))
    dummy_points[:ROBOTS_NUM, :] = positions_now
    mirrored_points = mirror(positions_now)
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
      U = ca.SX.sym('U', nu+HUMANS_NUM, T)
      x0 = ca.SX.sym('x0', nx)
      m_var = ca.SX.sym('m', 2)
      c_var = ca.SX.sym('c', 2, 2)

      # 2. Dynamics: single integrator
      vmax = 1.5
      amax = 5.0

      # 3. cost
      cost_expr = gmm_voronoi_cost_func(x0[:2], qs, means, covariances, weights)
      cost_fn = ca.Function('cost', [x0[:2]], [cost_expr])
      collision_expr = collision_cost_func(x0[:2], x_obs, Ds)
      collision_fn = ca.Function('obstacle_cost', [x0[:2]], [collision_expr])
      # human_cost_expr = human_cost_func(x0[:2], m_var, c_var, alpha=alpha_human)
      # human_cost_fn = ca.Function('human_cost', [x0[:2], m_var, c_var], [human_cost_expr])

      # 4. Build optimizaiton problem
      obj = 0.0
      x_curr = x0
      g_list = []
      human_covs = np.zeros((HUMANS_NUM, T, 2, 2))
      human_preds = np.zeros((HUMANS_NUM, T, 3))
      Q = np.diag([0.1, 0.1, 0.01, 0.01])
      F = np.eye(4)
      F[0, 2] = dt
      F[1, 3] = dt
      pred_0 = human_traj[s, :]
      human_preds[:, 0] = pred_0
      Q = 0.1 * np.eye(2)
      human_covs[:, 0] = Q
      for h in range(HUMANS_NUM):
        for i in range(1, T):
          human_preds[h, i, 0] = human_preds[h, i-1, 0] + 0.1 * np.cos(human_preds[h, i-1, 2])
          human_preds[h, i, 1] = human_preds[h, i-1, 1] + 0.1 * np.sin(human_preds[h, i-1, 2])
          human_preds[h, i, 2] = human_preds[h, i-1, 2]
          # human_covs[i, :, :] = (i+1) * 0.25 * np.eye(2) + 0.1 * np.random.rand(2, 2)
          human_covs[h, i, :, :] = np.eye(2) @ human_covs[h, i-1] @ np.eye(2).T + Q
      for k in range(T):
        # print("Planning step: ", k)
        obj += cost_fn(x_curr[:2]) #+ human_cost_fn(x_curr, human_traj[s+k, :2], human_covs[k])
        
        # obstacle avoidance constraint: Ds**2 - ||x - x_obs||**2 < 0
        for obs in x_obs:
          dist_squared = ca.sumsqr(x_curr[:2] - obs)
          g_list.append(Ds - ca.sqrt(dist_squared))
          # obj += 0.1*collision_fn(x_curr[:2])

        # Human avoidance constaint
        # Probabilistic Collision Checking With Chance Constraints, Du Toit et al. 2011 (T-RO)
        # diff = x_curr - human_traj[s+k, :2]
        for h in range(HUMANS_NUM):
          diff = x_curr[:2] - human_preds[h, k, :2]
          mahalanobis_dist_sq = diff.T @ ca.inv(human_covs[h, k]) @ diff
          prob = 0.01  # Desired probability of collision
          coeff = ca.sqrt(ca.det(2*ca.pi * human_covs[h, k]))
          # vol = 4/3 * ca.pi * Ds**3
          vol = ca.pi * Ds**2
          g_list.append(-2* ca.log(coeff * prob / vol) - mahalanobis_dist_sq + U[2+h, k])
          w_k = 100 * (1 - k/T)
          obj += w_k * U[2+h, k]**2  # Add the slack variable to the cost


        
        x_curr = single_int_dynamics(x_curr, U[:nu, k])

        # print("cost: ", obj)
        # 5. Constraints
        # Position (within env bounds) + vel
        g_list.append(x_curr[0] - 0.5*AREA_W)
        g_list.append(x_curr[1] - 0.5*AREA_W)
        g_list.append(-x_curr[0] - 0.5*AREA_W)
        g_list.append(-x_curr[1] - 0.5*AREA_W)
        if nx > 2:
          g_list.append(x_curr[2] - vmax)
          g_list.append(-x_curr[2] - vmax)
          g_list.append(x_curr[3] - vmax)
          g_list.append(-x_curr[3] - vmax)
      
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

      x_init = points[idx].copy()

      # 8. bounds and initial guess
      u_min = np.concatenate([np.array([-amax, -amax]), -np.inf * np.ones(HUMANS_NUM)])
      u_max = np.concatenate([np.array([amax, amax]), np.inf * np.ones(HUMANS_NUM)])
      lbx = np.tile(u_min, T)
      ubx = np.tile(u_max, T)
      
      # bounds on g: g <= 0
      lbg= -np.inf * np.ones(g.shape)
      ubg = np.zeros(g.shape)               # Ds**2 - ||x - x_obs||**2 < 0
      u0_guess = np.zeros(opt_vars.shape)
      # solve
      sol = solver(x0=u0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=x_init)
      stats = solver.stats()
      if not stats['success']:
        print("[IPOPT] Optimization failed with status: ", stats['return_status'])
      u_opt = sol['x'].full().reshape(T, nu+HUMANS_NUM)
      u_opt, slack = u_opt[:, :2], u_opt[:, 2:]

      # print("Slack variable: ", slack)
      # print("uopt: ", u_opt)
      planned_traj = np.zeros((T+1, nx))
      planned_traj[0, :] = x_init
      for k in range(T):
        planned_traj[k+1, :] = single_int_dynamics(planned_traj[k, :], u_opt[k, :])
      planned_trajectories[:, idx, :] = planned_traj
      points[idx, :] = single_int_dynamics(x_init, u_opt[0, :])
      robots_hist[s+1, idx, :] = points[idx, :2]
    
    if GRAPHICS_ON:
      ax.cla()
      # ax.contourf(X, Y, Z.reshape(X.shape), levels=10, cmap='YlOrRd', alpha=0.75)
      # for h in range(HUMANS_NUM):
      #   ax.plot(human_traj[:s+2, h, 0], human_traj[:s+2, h, 1], color='tab:purple', label='Human Trajectory')
      # ax.scatter(means[:, 0], means[:, 1], marker='*', color='tab:orange', label='GMM Means')
      # for t in range(T):
      #   alpha = np.exp(-np.log(10) * t / T)
      #   # draw_ellipse(human_traj[s+1+t, :], human_covs[t], n_std=1, ax=ax, alpha=alpha)
      #   for h in range(HUMANS_NUM):
      #     draw_ellipse(human_preds[h, t, :2], human_covs[h, t], n_std=2, ax=ax, alpha=alpha)
      # # ax.contourf(X, Y, alpha_human*human_pdf.reshape(X.shape), levels=10, cmap='Blues', alpha=0.5)
      # for idx in range(ROBOTS_NUM):
      #   ax.plot(robots_hist[:s+2, idx, 0], robots_hist[:s+2, idx, 1], label='Robot Trajectory', color='tab:blue')
      #   ax.scatter(robots_hist[s+1, idx, 0], robots_hist[s+1, idx, 1], color='tab:blue')
      #   ax.plot(planned_trajectories[:, idx, 0], planned_trajectories[:, idx, 1], label='Planned Trajectory', color='tab:green')
      #   x, y = polygons[idx].exterior.xy
      #   ax.plot(x, y, c='tab:red')
      
      # for obs in x_obs:
      #   ax.scatter(obs[0], obs[1], marker='x', color='k', label='Obstacle')
      #   xc = obs[0] + Ds * np.cos(np.linspace(0, 2*np.pi, 20))
      #   yc = obs[1] + Ds * np.sin(np.linspace(0, 2*np.pi, 20))
      #   ax.plot(xc, yc, c='k', label='Safety distance')
      # # ax.legend()
      # ax.set_aspect('equal', adjustable='box')   # keeps squares square
      # ax.set_autoscale_on(False)                 # stop anything else changing it
      # ax.set_xlim(-0.5*AREA_W, 0.5*AREA_W)
      # ax.set_ylim(-0.5*AREA_W, 0.5*AREA_W)
      lw=5
      # ax.contourf(X, Y, Z.reshape(X.shape), levels=10, cmap='Greys', alpha=0.75)
      ax.pcolormesh(X, Y, Z.reshape(X.shape), cmap='Greys', alpha=0.75)
      alpha_wall = 1
      square = patches.Rectangle((-1.5-Ds, -0.5*AREA_W), 2*Ds, 0.6*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      square2 = patches.Rectangle((1.5-Ds, -0.1*AREA_W), 2*Ds, 0.6*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      ax.add_patch(square)
      ax.add_patch(square2)
      wall1 = patches.Rectangle((-0.55*AREA_W, -0.55*AREA_W), 0.05*AREA_W, 1.1*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      wall2 = patches.Rectangle((0.5*AREA_W, -0.55*AREA_W), 0.05*AREA_W, 1.1*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      wall3 = patches.Rectangle((-0.55*AREA_W, -0.55*AREA_W), 1.1*AREA_W, 0.05*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      wall4 = patches.Rectangle((-0.55*AREA_W, 0.5*AREA_W), 1.1*AREA_W, 0.05*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      ax.add_patch(wall1)
      ax.add_patch(wall2)
      ax.add_patch(wall3)
      ax.add_patch(wall4)
      for idx in range(ROBOTS_NUM):
        if idx != 0 and idx != 4:
          ax.plot(robots_hist[:s+1, idx, 0], robots_hist[:s+1, idx, 1], linewidth=lw, label='Robot Trajectory', color=colors[idx], alpha=0.75)
          ax.scatter(robots_hist[s, idx, 0], robots_hist[s, idx, 1], s=20*lw, color=colors[idx])
          # x, y = polygons[idx].exterior.xy
          # ax.plot(x, y, c='tab:red', linewidth=lw, alpha=0.75)
      # for idx in range(ROBOTS_NUM):
        # ax.scatter(robots_hist[-1, idx, 0], robots_hist[-1, idx, 1], s=20*lw, color=colors[idx])
      
      for obs in x_obs[:OBS_NUM]:
        ax.scatter(obs[0], obs[1], marker='x', color='k', label='Obstacle')
        xc = obs[0] + Ds * np.cos(np.linspace(0, 2*np.pi, 20))
        yc = obs[1] + Ds * np.sin(np.linspace(0, 2*np.pi, 20))
        ax.plot(xc, yc, c='k', label='Safety distance')
      
      
      # ax.legend()
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_aspect('equal', adjustable='box')   # keeps squares square
      ax.set_autoscale_on(False)                 # stop anything else changing it
      ax.set_xlim(-0.55*AREA_W, 0.55*AREA_W)
      ax.set_ylim(-0.55*AREA_W, 0.55*AREA_W)
      fig.canvas.draw()
      plt.savefig(f"pics/tmp_maze/eval_{s:03d}.png")
      plt.pause(0.01)

    # cov_fn = 0.0
    # dx = 0.25
    # for idx in range(ROBOTS_NUM):
    #     region = vor.point_region[idx]
    #     poly_vert = []
    #     for vert in vor.regions[region]:
    #         v = vor.vertices[vert]
    #         poly_vert.append(v)

    #     poly = Polygon(poly_vert)

    #     xmin, ymin, xmax, ymax = poly.bounds
    #     for i in np.arange(xmin, xmax, dx):
    #         for j in np.arange(ymin, ymax, dx):
    #             pt_i = Point(i, j)
    #             if poly.contains(pt_i):
    #               dist = np.linalg.norm(np.array([i,j]) - positions_now[idx])
    #               cov_fn -= dist**2 * gmm_pdf(i, j, means, covariances, weights) * dx**2
    # coverage_over_time[ep, s] = cov_fn.item()


  """
  =======================================
  ===========  EVALUATION ===============
  =======================================
  """
  # Coverage effectiveness
  dx = 0.25
  ef_num = 0.0
  cov_fn = 0.0
  den = 0.0
  for idx in range(ROBOTS_NUM):
      region = vor.point_region[idx]
      poly_vert = []
      for vert in vor.regions[region]:
          v = vor.vertices[vert]
          poly_vert.append(v)

      poly = Polygon(poly_vert)

      xmin, ymin, xmax, ymax = poly.bounds
      for i in np.arange(xmin, xmax, dx):
          for j in np.arange(ymin, ymax, dx):
              pt_i = Point(i, j)
              if poly.contains(pt_i):
                dist = np.linalg.norm(np.array([i,j]) - positions_now[idx])
                cov_fn -= dist**2 * gmm_pdf(i, j, means, covariances, weights) * dx**2

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
      for i in np.arange(xmin, xmax, dx):
          for j in np.arange(ymin, ymax, dx):
              pt_i = Point(i, j)
              if lim_region.contains(pt_i):
                ef_num += gmm_pdf(i, j, means, covariances, weights) * dx**2

  for i in np.arange(-0.5*AREA_W, 0.5*AREA_W, dx):
    for j in np.arange(-0.5*AREA_W, 0.5*AREA_W, dx):
      den += gmm_pdf(i, j, means, covariances, weights) * dx**2

  effectiveness[ep] = (ef_num / den).item()
  coverage_func[ep] = (cov_fn).item()
  
  # log
  print(f"Ep: {ep}: Collisions: {collisions[ep]} | Effectiveness: {effectiveness[ep]} | Coverage: {coverage_func[ep]}")
  with open(eval_path, "a") as f:
    f.write(f"{ep}\t{HUMANS_NUM}\t{collisions[ep]}\t{effectiveness[ep]}\t{coverage_func[ep]}\n")  

  

  if GRAPHICS_ON:
    plt.ioff()
    plt.show()

  # fig, ax = plt.subplots(figsize=(8,8))
  # lw=5
  # # ax.contourf(X, Y, Z.reshape(X.shape), levels=10, cmap='Greys', alpha=0.75)
  # ax.pcolormesh(X, Y, Z.reshape(X.shape), cmap='Greys', alpha=0.75)
  # alpha_wall = 1
  # square = patches.Rectangle((-1.5-Ds, -0.5*AREA_W), 2*Ds, 0.5*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
  # square2 = patches.Rectangle((1.5-Ds, -0.1*AREA_W), 2*Ds, 0.6*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
  # ax.add_patch(square)
  # ax.add_patch(square2)
  # wall1 = patches.Rectangle((-0.55*AREA_W, -0.55*AREA_W), 0.05*AREA_W, 1.1*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
  # wall2 = patches.Rectangle((0.5*AREA_W, -0.55*AREA_W), 0.05*AREA_W, 1.1*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
  # wall3 = patches.Rectangle((-0.55*AREA_W, -0.55*AREA_W), 1.1*AREA_W, 0.05*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
  # wall4 = patches.Rectangle((-0.55*AREA_W, 0.5*AREA_W), 1.1*AREA_W, 0.05*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
  # ax.add_patch(wall1)
  # ax.add_patch(wall2)
  # ax.add_patch(wall3)
  # ax.add_patch(wall4)
  # for idx in range(ROBOTS_NUM):
  #   ax.plot(robots_hist[:, idx, 0], robots_hist[:, idx, 1], linewidth=lw, label='Robot Trajectory', color=colors[idx], alpha=0.75)
  #   ax.scatter(robots_hist[0, idx, 0], robots_hist[0, idx, 1], s=20*lw, facecolors='none', edgecolors=colors[idx])
  #   x, y = polygons[idx].exterior.xy
  #   ax.plot(x, y, c='tab:red', linewidth=lw, alpha=0.75)
  # for idx in range(ROBOTS_NUM):
  #   ax.scatter(robots_hist[-1, idx, 0], robots_hist[-1, idx, 1], s=20*lw, color=colors[idx])
  
  # for obs in x_obs[:OBS_NUM]:
  #   ax.scatter(obs[0], obs[1], marker='x', color='k', label='Obstacle')
  #   xc = obs[0] + Ds * np.cos(np.linspace(0, 2*np.pi, 20))
  #   yc = obs[1] + Ds * np.sin(np.linspace(0, 2*np.pi, 20))
  #   ax.plot(xc, yc, c='k', label='Safety distance')
  
  
  # # ax.legend()
  # ax.set_aspect('equal', adjustable='box')   # keeps squares square
  # ax.set_autoscale_on(False)                 # stop anything else changing it
  # ax.set_xlim(-0.55*AREA_W, 0.55*AREA_W)
  # ax.set_ylim(-0.55*AREA_W, 0.55*AREA_W)
  # plt.show()

# Save coverage function over time
# with open('results/cov_fn_mpc.npy', 'wb') as f:
#     np.save(f, coverage_over_time)
# print("Saved coverage function")
