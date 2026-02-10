import numpy as np
import casadi as ca

import numba
from numba import njit, prange

#### 
# Cost function for Discrete PDF MPC
####
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

def coverage_cost(
    robot_pos,          # SX (1, 2)
    grid_points,        # DX (GRID_CELLS**2, 2)
    weights             # np.ndarray (GRID_CELLS**2,) 
):
  """
  Compute coverage cost for a robot at position `robot_pos` given a set of `grid_points` and their corresponding `weights`.
  
  :param robot_pos: 2D position of the robot
  :param grid_points: Grid points of the environment
  :param weights: pdf[k] if k inside robot's Voronoi region, else 0
  """
  diff = grid_points - ca.repmat(robot_pos.T, grid_points.shape[0], 1)
  d2 = ca.sum2(diff**2)
  return ca.dot(d2, weights)