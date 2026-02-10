import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
import json

import numba
from numba import njit, prange

from utils import utilities, costFunctions, models

"""
===============================
Parameters
===============================
"""
param_file = path + '/params/' + 'params.json'

with open(param_file, 'r') as f:
    param_data = json.load(f)

params = lambda: None

for key, value in param_data.items():
    setattr(params, key, value)

np.random.seed(params.random_seed)

# 0. parameters
T = params.horizon_steps
dt = params.dt
sim_time = params.sim_time
NUM_STEPS = int(sim_time / dt)
R = params.robot_range
r = R/2
AREA_W = params.width
ROBOTS_NUM = params.num_agents
OBS_NUM = params.num_obstacles       # obstacles
Ds = params.obstacle_radius          # safety distance to obstacles
GRID_SIZE = params.num_cells
VIDEO = params.record_video
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

# Get dynamics model configuration from params
model_type = getattr(params, 'dynamics', 'double_integrator')
model_config = models.get_model_config(model_type)
nx = model_config['nx']
nu = model_config['nu']

# Get dynamics functions
dynamics_numba = models.get_dynamics(model_type, backend='numba')
dynamics_casadi = models.get_dynamics(model_type, backend='casadi')
simulate_trajectory_numba = models.simulate_trajectory_numba

# points = np.concatenate((x1.reshape(1, -1), x2.reshape(1, -1)),axis=0)
points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, nx)
if nx >= 3:
    points[:, 2:] = 0.0                   # robots start with zero velocity/heading
# mean = -0.5*AREA_W + AREA_W * np.random.rand(2)
# mean = np.array([0.0, 0.0])
# cov = 2.0 * np.diag([1.0, 1.0])
x_obs = -0.5*AREA_W + AREA_W * np.random.rand(OBS_NUM, 2)
means = np.array(params.means)
covs = np.array(params.covariances)
gmm_ws = np.array(params.weights)

# plotting stuff
xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_SIZE)
yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_SIZE)
X, Y = np.meshgrid(xg, yg)
xy_grid = np.column_stack([X.flatten(), Y.flatten()])
Z = utilities.gmm_pdf(X, Y, means, covs, gmm_ws)
pdf = Z / np.sum(Z)  # normalize to get a valid PDF


# Build CasADi solvers for each robot
# opts = {'ipopt': {'print_level': 0, 'sb': 'yes', 'max_iter': 1000, 'tol': 1e-5}, 'print_time': False}
t_start = time.perf_counter()
opts = {
#   'jit': True,
#   'jit_options': {'flags': '-O2'},
#   'compiler': 'shell',
  'ipopt': {
      'print_level': 0,
      'sb': 'yes',
      'max_iter': 300,
      'tol': 1e-4,
      'warm_start_init_point': 'yes',
      'warm_start_bound_push': 1e-6,
      'warm_start_mult_bound_push': 1e-6,
  },
  'print_time': False
}
solvers = []
R_u = np.zeros((nu,))
GRID_DM = ca.DM(xy_grid)

for idx in range(ROBOTS_NUM):
    U = ca.SX.sym('U', nu, T)
    x0 = ca.SX.sym('x0', nx)
    W = ca.SX.sym('W', GRID_SIZE**2)  # weights

    x = x0
    obj = 0

    for k in range(T):
        obj += costFunctions.coverage_cost(x[:2], GRID_DM, W)
        # obj += ca.sumsqr(U[:, k]) * 0.01
        # obj += control_effort_cost(U[:, k].reshape(-1,1), R_u)
        for obs in x_obs:
            obj += costFunctions.collision_cost(x[:2], obs, Ds)
        x = dynamics_casadi(x, U[:, k], dt)

    g = ca.vertcat(
        x[0] - 0.5*AREA_W,
        -x[0] - 0.5*AREA_W,
        x[1] - 0.5*AREA_W,
        -x[1] - 0.5*AREA_W
    )

    solver = ca.nlpsol(
        f'solver_{idx}',
        'ipopt',
        {'x': ca.vec(U), 'f': obj, 'g': g, 'p': ca.vertcat(x0, W)},
        opts
    )

    solvers.append(solver)

# bounds and initial guess
amax = 2.0
u_min = np.array([-amax, -amax])
u_max = np.array([amax, amax])
lbx = np.tile(u_min, T)
ubx = np.tile(u_max, T)

# bounds on g: g <= 0
lbg = -np.inf * np.ones(g.shape)
ubg = np.zeros(g.shape)               # Ds**2 - ||x - x_obs||**2 < 0
t_end = time.perf_counter()
print(f"Solvers initialized in {t_end - t_start:.2f} seconds")

# Pre-allocate arrays for simulation loop
robots_hist = np.zeros((NUM_STEPS+1, ROBOTS_NUM, nx))
robots_hist[0, :, :] = points
u_prev = np.zeros((ROBOTS_NUM, nu * T))
planned_trajectories = np.zeros((T+1, ROBOTS_NUM, nx))

if VIDEO:
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))
for s in range(NUM_STEPS):
    t_start = time.perf_counter()
    planned_trajectories.fill(0.0)
    positions_now = points.copy()
    grid_pts = []
    
    # Compute Voronoi partitioning
    voronoi_masks = utilities.compute_voronoi_partitioning(xy_grid, positions_now[:, :2], r)
    
    # Pre-compute weights for all robots
    weights_list = []
    for idx in range(ROBOTS_NUM):
        weights = pdf * voronoi_masks[idx]
        weights_list.append(weights)
    
    for idx in range(ROBOTS_NUM):
        voronoi_points = xy_grid[voronoi_masks[idx], :]
        grid_pts.append(voronoi_points)
        
        p = np.concatenate([positions_now[idx], weights_list[idx]])
        x_init = positions_now[idx, :].copy()
        
        # Solve optimization
        u0_guess = u_prev[idx, :].copy()
        sol = solvers[idx](x0=u0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
        u_opt = sol['x'].full().reshape(T, nu)
        u_prev[idx, :] = u_opt.flatten()
        
        # Simulate trajectory using JIT-compiled function
        planned_traj = simulate_trajectory_numba(x_init, u_opt, dt, T, nx, dynamics_numba)
        planned_trajectories[:, idx, :] = planned_traj
        
        # Update state using JIT-compiled dynamics
        points[idx, :] = dynamics_numba(positions_now[idx, :], u_opt[0, :], dt)
        robots_hist[s+1, idx, :] = points[idx, :]
    
    t_end = time.perf_counter()
    print(f"Step {s+1}/{NUM_STEPS} computed in {t_end - t_start:.2f} seconds")
    
    # Visualization
    if VIDEO:
        ax.cla()
        ax.contourf(X, Y, pdf.reshape(X.shape), levels=10, cmap='YlOrRd', alpha=0.75)
        # for i in range(ROBOTS_NUM):
        #     voronoi_points = grid_pts[i]
        #     ax.scatter(voronoi_points[:, 0], voronoi_points[:, 1], c=colors[i], s=1)
        for obs in x_obs:
            ax.scatter(obs[0], obs[1], marker='x', color='k', s=50)
            xc = obs[0] + Ds * np.cos(np.linspace(0, 2*np.pi, 20))
            yc = obs[1] + Ds * np.sin(np.linspace(0, 2*np.pi, 20))
            ax.plot(xc, yc, c='k', linestyle='--', alpha=0.5)
        for idx in range(ROBOTS_NUM):
            ax.plot(robots_hist[:s+2, idx, 0], robots_hist[:s+2, idx, 1], 
                    color=colors[idx], linewidth=1.5)
            ax.scatter(robots_hist[s+1, idx, 0], robots_hist[s+1, idx, 1], 
                    color=colors[idx], s=50, zorder=5)
            ax.plot(planned_trajectories[:, idx, 0], planned_trajectories[:, idx, 1], 
                    color=colors[idx], linestyle='--', alpha=0.5, linewidth=1)
    
    if VIDEO:
        plt.pause(0.01)


if VIDEO:
    plt.ioff()
    plt.show()

