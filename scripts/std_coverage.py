import numpy as np
import casadi as ca
from scipy.spatial import Voronoi
from shapely import Polygon, Point
import matplotlib.pyplot as plt
import argparse


def parse_args():
  parser = argparse.ArgumentParser(description='Coverage Control Simulation')
  parser.add_argument('--model', type=str, default='unicycle', help='Model type: unicycle, single_int, double_int')
  return parser.parse_args()

def gauss_pdf(x, y, mean, covariance):

  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)
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
T = 15
dt = 0.1
sim_time = 20.0
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

nx = 2    # [x, y]
nu = 2    # [vx, vy]
if MODEL == "double_int":
  nx = 4    # [x, y, vx, vy]
elif MODEL == "unicycle":
  nx = 3    # [x, y, theta]

# points = np.concatenate((x1.reshape(1, -1), x2.reshape(1, -1)),axis=0)
points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, nx)
if MODEL == "double_int":
  points[:, 2:] = 0.0                   # robots start with zero velocity
elif MODEL == "unicycle":
  points[:, 2] = 2 * np.pi * np.random.rand(ROBOTS_NUM)  # random initial heading
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


robots_hist = np.zeros((NUM_STEPS+1, ROBOTS_NUM, nx))
robots_hist[0, :, :] = points
for s in range(NUM_STEPS):
  planned_trajectories = np.zeros((T+1, ROBOTS_NUM, nx))
  polygons = []
  centroids = []
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
    Cx = 0.0  
    Cy = 0.0
    A = 0.0
    for i in np.linspace(xmin, xmax, discr_points):
        for j in np.linspace(ymin, ymax, discr_points):
            pt_i = Point(i, j)
            if lim_region.contains(pt_i):
              dA_pdf = gauss_pdf(i, j, mean, cov)
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
    for obs in x_obs:
      dist = np.linalg.norm(positions_now[idx, :2] - obs)
      if dist < D_rep:
        k_il = np.power((0.5 + 0.5 * np.cos(np.pi * (dist - Ds)/(D_rep - Ds))), gamma)
        d_norm = -(obs - positions_now[idx, :2]) / dist
        u_rep += k_il * d_norm / (dist - Ds)

    vel = u_cov.squeeze(0) + u_rep
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
    if MODEL == "unicycle":
      # heading
      xh = points[idx, 0] + 0.5 * np.cos(points[idx, 2])
      yh = points[idx, 1] + 0.5 * np.sin(points[idx, 2])
      ax.plot([points[idx, 0], xh], [points[idx, 1], yh], color='tab:orange')
    x, y = polygons[idx].exterior.xy
    ax.plot(x, y, c='tab:red')
  
  # ax.legend()
  plt.pause(0.01)

plt.ioff()
plt.show()

