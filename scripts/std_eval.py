import numpy as np
import casadi as ca
from scipy.spatial import Voronoi
from shapely import Polygon, Point
import matplotlib.pyplot as plt
import argparse
import matplotlib.patches as patches


def parse_args():
  parser = argparse.ArgumentParser(description='Coverage Control Simulation')
  parser.add_argument('--model', type=str, default='double_int', help='Model type: unicycle, single_int, double_int')
  return parser.parse_args()

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


def double_int_dynamics(state, u, t_step):
  A = np.array([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])
  B = np.array([[0, 0],
                [0, 0],
                [1, 0],
                [0, 1]])

  x_dot = A @ state + B @ u
  return state + x_dot * t_step

def single_int_dynamics(state, u, t_step):
  return state + u * t_step

def unicycle_dynamics(state, u, t_step):
  x = state[0]
  y = state[1]
  theta = state[2]
  v = u[0]
  w = u[1]

  x_dot = v * np.cos(theta)
  y_dot = v * np.sin(theta)
  theta_dot = w

  return np.array([x + x_dot * t_step, y + y_dot * t_step, theta + theta_dot * t_step])

def getInputFromLinear(vx, vy, theta):
  v = vx * np.cos(theta) + vy * np.sin(theta)
  th_des = np.arctan2(vy, vx)
  w = th_des - theta
  w = np.arctan2(np.sin(w), np.cos(w))  # Normalize the angle

  return np.array([v, w])

  
args = parse_args()
# MODEL = "unicycle"              # unicycle, single_int, double_int
MODEL = args.model
assert MODEL in ["unicycle", "single_int", "double_int"], "Model not supported"
print("Model: ", MODEL)

# 0. parameters
T = 2
dt = 0.1
sim_time = 10.0
NUM_STEPS = int(sim_time / dt)
NUM_EPISODES = 10
GRAPHICS_ON = True
np.random.seed(2)
# x1 = np.zeros(2)
# x2 = np.array([2.5, 3.0])
# mean = np.array([5.0, -2.5])
R = 5.0
r = R/2
AREA_W = 10.0
ROBOTS_NUM = 4
OBS_NUM = 5       # obstacles
HUMANS_NUM = 3
Ds = 0.5          # safety distance to obstacles
vmax = 3.0

nx = 2    # [x, y]
nu = 2    # [vx, vy]
if MODEL == "double_int":
  nx = 4    # [x, y, vx, vy]
elif MODEL == "unicycle":
  nx = 3    # [x, y, theta]

mod = "di"
if MODEL == "unicycle":
  mod = "uni"
eval_path = f"results/std_eval_{int(HUMANS_NUM/AREA_W**2*100):03d}_{mod}.txt"
with open(eval_path, "w") as f:
  f.write("It\tNh\tColl\tA\tH\n")
collisions = np.zeros(NUM_EPISODES)               # check for collisions with humans / obstacles
effectiveness = np.zeros(NUM_EPISODES)            # Coverage effectiveness  
coverage_func = np.zeros(NUM_EPISODES)            # range-unlimited Coverage function 
coverage_over_time = np.zeros((NUM_EPISODES, NUM_STEPS))
effectiveness_over_time = np.zeros((NUM_EPISODES, NUM_STEPS))

for ep in range(NUM_EPISODES):
  x_obs = -0.5*AREA_W + AREA_W * np.random.rand(OBS_NUM, 2)

  human_traj = np.zeros((NUM_STEPS+T, HUMANS_NUM, 3))
  human_traj[0, :, :2] = -0.5*AREA_W + AREA_W*np.random.rand(HUMANS_NUM, 2)
  # human_traj[0, 2] = 2*np.pi * np.random.rand()  # initial heading
  human_traj[0, :, 2] = 2*np.pi * np.random.rand(HUMANS_NUM)

  # points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, nx)
  points = []
  while len(points) < ROBOTS_NUM:
      candidate = -0.4 * AREA_W + 0.8 * AREA_W * np.random.rand(1, nx)
      pos = candidate[0, :2]  # x, y

      obs_dists = np.linalg.norm(x_obs - pos, axis=1)
      h_dists = np.linalg.norm(human_traj[0, :, :2] - pos, axis=1)
      if np.all(obs_dists > 3*Ds) and np.all(h_dists > 3*Ds):
          points.append(candidate[0])

  points = np.array(points)  # shape (ROBOTS_NUM, nx)
  points[:, 2:] = 0.0
  if MODEL == "double_int":
    points[:, 2:] = 0.0                   # robots start with zero velocity
  elif MODEL == "unicycle":
    points[:, 2] = 2 * np.pi * np.random.rand(ROBOTS_NUM)  # random initial heading

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
  if GRAPHICS_ON:
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))


  robots_hist = np.zeros((NUM_STEPS+1, ROBOTS_NUM, nx))
  robots_hist[0, :, :] = points
  for s in range(NUM_STEPS):
    planned_trajectories = np.zeros((T+1, ROBOTS_NUM, nx))
    polygons = []
    centroids = []
    positions_now = points.copy()

    # Check collisions
    diffs = positions_now[:, np.newaxis, :2] - x_obs[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    coll = np.any(dists < Ds)
    collisions[ep] = collisions[ep] or coll
    if coll:
      print("[WARNING] Collision with obstacle detected")

    diffs = positions_now[:, np.newaxis, :2] - human_traj[s, np.newaxis, :, :2]
    dists = np.linalg.norm(diffs, axis=2)
    coll = np.any(dists < Ds)
    collisions[ep] = collisions[ep] or coll
    if coll:
      print("[WARNING] Collision with human detected")

    # check if outside bounds
    coll = np.any(positions_now[:, :2] > 0.5*AREA_W) or np.any(positions_now[:, :2] < -0.5*AREA_W)
    collisions[ep] = collisions[ep] or coll
    if coll:
      print("[WARNING] Out of bounds.")
      
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
      Cx = 0.0  
      Cy = 0.0
      A = 0.0
      for i in np.linspace(xmin, xmax, discr_points):
          for j in np.linspace(ymin, ymax, discr_points):
              pt_i = Point(i, j)
              if lim_region.contains(pt_i):
                dA_pdf = gmm_pdf(i, j, means, covariances, weights)
                Cx += i * dA_pdf
                Cy += j * dA_pdf
                A += dA_pdf

      Cx /= A
      Cy /= A
      C = np.array([Cx, Cy]).T
      centroids.append(C)
      # ---------------- End Voronoi partitioning ----------------

      # Coverage action
      Kp = 0.8
      u_cov = Kp * (C - positions_now[idx, :2])

      # ------------------ Repulsive action ------------------
      # see: Cheng, Bin, et al. 
      # "Development and Application of Coverage Control Algorithms: A Concise Review."
      # Section 3B.1
      # Original from: C. Franco, D. M. Stipanović, G. López-Nicolás, C. Sagüés, and
      # S. Llorente, “Persistent coverage control for a team of agents with
      # collision avoidance”
      # ---------------------------------------------------- 


      D_rep = 2.0           # distance for repulsion to be effective
      gamma = 2.0           # high gamma allows getting closer to obstacles
      u_rep = np.zeros((2,))
      obs_and_humans = np.concatenate((x_obs, human_traj[s, :, :2]))
      for obs in obs_and_humans:
        dist = np.linalg.norm(positions_now[idx, :2] - obs)
        if dist < D_rep:
          k_il = np.power((0.5 + 0.5 * np.cos(np.pi * (dist - Ds)/(D_rep - Ds))), gamma)
          d_norm = -(obs - positions_now[idx, :2]) / dist
          u_rep += k_il * d_norm / (dist - Ds)

      vel = u_cov.squeeze(0) + u_rep
      vel = np.clip(vel, -vmax, vmax)
      u = vel.copy()
      if MODEL == "unicycle":
        u = getInputFromLinear(vel[0], vel[1], positions_now[idx, 2])
        points[idx, :] = unicycle_dynamics(positions_now[idx], u, dt)
      elif MODEL == "double_int":
        Kv = 0.5
        u = Kv * (vel - positions_now[idx, 2:4])
        points[idx, :] = double_int_dynamics(positions_now[idx], u, dt)
      else:
        points[idx, :] = single_int_dynamics(positions_now[idx], u, dt)
      
      
      robots_hist[s+1, idx, :] = points[idx, :]

      
    if GRAPHICS_ON:
      ax.cla()
      ax.pcolormesh(X, Y, Z.reshape(X.shape), cmap='Greys', alpha=0.75)
      alpha_wall = 1.0
      lw = 5
      wall1 = patches.Rectangle((-0.55*AREA_W, -0.55*AREA_W), 0.05*AREA_W, 1.1*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      wall2 = patches.Rectangle((0.5*AREA_W, -0.55*AREA_W), 0.05*AREA_W, 1.1*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      wall3 = patches.Rectangle((-0.55*AREA_W, -0.55*AREA_W), 1.1*AREA_W, 0.05*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      wall4 = patches.Rectangle((-0.55*AREA_W, 0.5*AREA_W), 1.1*AREA_W, 0.05*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      ax.add_patch(wall1)
      ax.add_patch(wall2)
      ax.add_patch(wall3)
      ax.add_patch(wall4)
      square = patches.Rectangle((-1.5-Ds, -0.5*AREA_W), 2*Ds, 0.5*AREA_W, facecolor='black', edgecolor='black', alpha=alpha_wall)
      ax.add_patch(square)
      for obs in x_obs[:OBS_NUM]:
        # ax.scatter(obs[0], obs[1], marker='x', color='k', label='Obstacle')
        # xc = obs[0] + Ds * np.cos(np.linspace(0, 2*np.pi, 20))
        # yc = obs[1] + Ds * np.sin(np.linspace(0, 2*np.pi, 20))
        # ax.plot(xc, yc, c='k', label='Safety distance')
        circle = patches.Circle(obs, radius=Ds, color='black')
        ax.add_patch(circle)
      # for h in range(HUMANS_NUM):
        # ax.plot(human_traj[:s+2, h, 0], human_traj[:s+2, h, 1], color='tab:purple', linewidth=lw, label='Human')
      # ax.scatter(means[:, 0], means[:, 1], marker='*', color='tab:orange', label='GMM Means')
      # for t in range(T):
      #   alpha = np.exp(-np.log(10) * t / T)
        # draw_ellipse(human_traj[s+1+t, :], human_covs[t], n_std=1, ax=ax, alpha=alpha)
      for h in range(HUMANS_NUM):
        ax.scatter(human_traj[s, h, 0], human_traj[s, h, 1], color='tab:purple', s=20*lw)
        xh = human_traj[s, h, 0] + Ds * np.cos(np.linspace(0, 2*np.pi, 20))
        yh = human_traj[s, h, 1] + Ds * np.sin(np.linspace(0, 2*np.pi, 20))
        ax.plot(xh, yh, c='tab:purple', label='Safety distance', linewidth=lw, alpha=0.5)
          # draw_ellipse(human_preds[h, t, :2], human_covs[h, t], edgecolor='tab:purple', lw=lw, n_std=1, ax=ax, alpha=alpha)
      # ax.contourf(X, Y, alpha_human*human_pdf.reshape(X.shape), levels=10, cmap='Blues', alpha=0.5)
      for idx in range(ROBOTS_NUM):
        x, y = polygons[idx].exterior.xy
        ax.plot(x, y, c='tab:red', alpha=0.75, linewidth=lw)
        # ax.plot(robots_hist[:s+1, idx, 0], robots_hist[:s+1, idx, 1], label='Trajectory', color='tab:blue', linewidth=lw)
        ax.plot(planned_trajectories[:, idx, 0], planned_trajectories[:, idx, 1], label='Planned Trajectory', linewidth=0.5*lw, color='tab:green')
        ax.scatter(robots_hist[s, idx, 0], robots_hist[s, idx, 1], color='tab:blue', s=20*lw)
        
      # ax.legend()
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_aspect('equal', adjustable='box')   # keeps squares square
      ax.set_autoscale_on(False)                 # stop anything else changing it
      ax.set_xlim(-0.55*AREA_W, 0.55*AREA_W)
      ax.set_ylim(-0.55*AREA_W, 0.55*AREA_W)
      fig.canvas.draw()
      # plt.legend()
      plt.savefig(f"pics/temp/std_eval_{s:03d}.png")
      plt.pause(0.01)
    
    # cov_fn = 0.0
    # ef_num = 0.0
    # dx = 0.25
    # for idx in range(ROBOTS_NUM):
    #   region = vor.point_region[idx]
    #   poly_vert = []
    #   for vert in vor.regions[region]:
    #       v = vor.vertices[vert]
    #       poly_vert.append(v)

    #   poly = Polygon(poly_vert)

    #   xmin, ymin, xmax, ymax = poly.bounds
    #   for i in np.arange(xmin, xmax, dx):
    #       for j in np.arange(ymin, ymax, dx):
    #           pt_i = Point(i, j)
    #           if poly.contains(pt_i):
    #             dist = np.linalg.norm(np.array([i,j]) - positions_now[idx, :2])
    #             cov_fn -= dist**2 * gmm_pdf(i, j, means, covariances, weights) * dx**2
      
    #   # Limited range cell
    #   range_vert = []
    #   for th in np.arange(0, 2*np.pi, np.pi/10):
    #     vx = positions_now[idx, 0] + r * np.cos(th)
    #     vy = positions_now[idx, 1] + r * np.sin(th)
    #     range_vert.append((vx, vy))
    #   range_poly = Polygon(range_vert)
    #   lim_region = poly.intersection(range_poly)
    #   polygons.append(lim_region)
    #   robot = vor.points[idx]

    #   xmin, ymin, xmax, ymax = poly.bounds
    #   for i in np.arange(xmin, xmax, dx):
    #       for j in np.arange(ymin, ymax, dx):
    #           pt_i = Point(i, j)
    #           if lim_region.contains(pt_i):
    #             ef_num += gmm_pdf(i, j, means, covariances, weights) * dx**2

    # den = 0.0
    # for i in np.arange(-0.5*AREA_W, 0.5*AREA_W, dx):
    #   for j in np.arange(-0.5*AREA_W, 0.5*AREA_W, dx):
    #     den += gmm_pdf(i, j, means, covariances, weights) * dx**2

    
    # effectiveness_over_time[ep, s] = (ef_num / den).item()
    # coverage_over_time[ep, s] = cov_fn.item()
    
  if GRAPHICS_ON:
    plt.ioff()
    plt.show()

    

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
                dist = np.linalg.norm(np.array([i,j]) - positions_now[idx, :2])
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


# with open('results/cov_fn_std.npy', 'wb') as f:
#   np.save(f, coverage_over_time)
# with open('results/effect_std.npy', 'wb') as f:
#   np.save(f, effectiveness_over_time)
# print("Saved coverage function")