#!/usr/bin/python3

import rospy
import math
import numpy as np
import casadi as ca
from scipy.spatial import Voronoi
from shapely import Polygon, Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from io import BytesIO
from PIL import Image
import imageio.v2 as imageio
import cv2

from geometry_msgs.msg import PoseStamped
import tf.transformations as t
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

def is_covariance_matrix(matrix, tol=1e-8):
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False, "Matrix is not square."
    
    # Check symmetry
    if not np.allclose(matrix, matrix.T, atol=tol):
        return False, "Matrix is not symmetric."
    
    # Check positive semi-definite by eigenvalues
    eigvals = np.linalg.eigvalsh(matrix)
    if np.any(eigvals < -tol):
        return False, "Matrix is not positive semi-definite."
    
    return True, "Matrix is a valid covariance matrix."

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
  mahalanobis_dist = []
  for i in range(x.shape[0]):
    diff = x[i, :] - mean
    mahalanobis_dist.append(diff.T @ ca.inv(covariance) @ diff)
  mahalanobis_dist = ca.vertcat(*mahalanobis_dist)
  exponent = -0.5 * mahalanobis_dist
  return coeff * ca.exp(exponent)

def gmm_pdf_func(x, means, covariances, weights):
  prob = 0.0
  for i in range(len(means)):
    prob += weights[i] * pdf_func(x, means[i], covariances[i])
  return prob

def gmm_voronoi_cost_func(pi, q, means, covariances, weights):
  p = ca.reshape(pi, 1, q.shape[1])
  p = ca.repmat(p, q.shape[0], 1)
  sq_dists = ca.sum2((p - q)**2)
  pdfs = gmm_pdf_func(q, means, covariances, weights)
  cost = ca.sum1(sq_dists * pdfs)
  return cost

def mirror(points, w):
    mirrored_points = []

    # Define the corners of the square
    square_corners = [(-0.5*w, -0.5*w), (0.5*w, -0.5*w), (0.5*w, 0.5*w), (-0.5*w, 0.5*w)]

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




class Controller():
    def __init__(self):
        rospy.init_node("coverage_mpc_node")

        """
        ===============================
        ROS params
        ===============================
        """
        # General
        self.humans_num = rospy.get_param("humans_num", 3)
        self.robots_num = rospy.get_param("robots_num", 6)
        self.seed = rospy.get_param("seed", 0)
        np.random.seed(self.seed)
        self.prob = rospy.get_param("prob", 0.1)       # human risk factor
        self.dt = rospy.get_param("dt", 0.5)
        self.graphics = rospy.get_param("~graphics_on", False)
        self.save_video = rospy.get_param("~save_video", False)

        # Robot
        self.T = rospy.get_param("horizon", 10)
        self.id = rospy.get_param("~id", 0)
        self.range = rospy.get_param("~robot_range", 5.0)
        self.half_range = 0.5*self.range
        self.Ds = rospy.get_param("safety_dist", 1.0)
        self.vmax = rospy.get_param("~v_max", 0.3)
        self.wmax = rospy.get_param("~w_max", 0.5)
        self.amax = rospy.get_param("~a_max", 1.0)

        # Environment
        self.width = rospy.get_param("~area_width", 20)
        means = rospy.get_param("means", [[-1.0, 5.0], 
                                            [5.0, 5.0]])
        self.means = np.array(means)
        covs = rospy.get_param("covariances", [[0.5, 0.2, 0.2, 1.0], 
                                                [0.5, 0.2, 0.2, 1.0]])
        self.covariances = np.array(covs).reshape(-1, 2, 2)
        weights = rospy.get_param("weights", [0.5, 
                                                0.5])
        self.weights = np.array(weights)
        obstacles = rospy.get_param("obstacles", [[-4.5, 4.5], 
                                                    [-3.5, -4.5],
                                                    [2.5, -2.5], 
                                                    [4.5, 2.5]])
        self.obstacles = np.array(obstacles)

        self.max_frames = 2000
        self.frames = []

        assert self.means.shape[0] == self.covariances.shape[0]
        assert self.means.shape[0] == self.weights.shape[0]
        for cov in self.covariances:
            assert is_covariance_matrix(cov)

        # ROS pubs/subs
        robots_subs = [None] * self.robots_num
        humans_subs = [None] * self.humans_num
        self.robot_poses = np.zeros((self.robots_num, 3))
        self.human_poses = np.zeros((self.humans_num, 3))
        for i in range(self.robots_num):
            robots_subs[i] = rospy.Subscriber(
                f"/tb3_{i}/odom", 
                Odometry, 
                self.robot_cb,
                callback_args=i
            )
        for i in range(self.humans_num):
            humans_subs[i] = rospy.Subscriber(
                f"/human{i}/actor_pose", 
                PoseStamped, 
                self.human_cb, 
                callback_args=i
            )

        self.vel_pub = rospy.Publisher(
            f"/tb3_{self.id}/cmd_vel",
            Twist,
            queue_size=10
        )

        self.path_pub = rospy.Publisher(
            f"/tb3_{self.id}/planned_trajectory",
            Path,
            queue_size=10
        )

        self.obs_pub = rospy.Publisher(
            f"/tb3_{self.id}/obstacles",
            MarkerArray,
            queue_size=10
        )
        
        self.timer = rospy.Timer(
            rospy.Duration(self.dt),
            self.timer_cb
        )

        self.viz_timer = rospy.Timer(
            rospy.Duration(self.dt),
            self.viz_cb
        )

        rospy.on_shutdown(self.shutdown_hook)

        xg = np.linspace(-0.5*self.width, 0.5*self.width, 100)
        yg = np.linspace(-0.5*self.width, 0.5*self.width, 100)
        self.X, self.Y = np.meshgrid(xg, yg)
        self.Z = gmm_pdf(self.X, self.Y, self.means, self.covariances, self.weights)
            
        if self.graphics:
            # plotting stuff
            # plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            # self.ax.contourf(self.X, self.Y, self.Z.reshape(self.X.shape), levels=10, cmap='YlOrRd', alpha=0.75)

    def shutdown_hook(self):
        rospy.loginfo(f"Shutting down robot {self.id} ...")
        self.timer.shutdown()
        self.viz_timer.shutdown()
        if self.save_video:
            rospy.loginfo("Saving video...")
            # Write video
            # with imageio.get_writer('/home/user/video.mp4') as writer:
            #     for frame in self.frames:
            #         writer.append_data(frame)
            height, width = 480, 640  # set your frame size here
            fps = 10
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter('/home/user/output_video.avi', fourcc, fps, (width, height))
            for frame in self.frames:
                video.write(frame)
            video.release()

    
    def robot_cb(self, msg, i):
        self.robot_poses[i, 0] = msg.pose.pose.position.x
        self.robot_poses[i, 1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        _, _, yaw = t.euler_from_quaternion(quaternion)
        self.robot_poses[i, 2] = yaw
    
    def human_cb(self, msg, i):
        self.human_poses[i, 0] = msg.pose.position.x
        self.human_poses[i, 1] = msg.pose.position.y
        q = msg.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        _, _, yaw = t.euler_from_quaternion(quaternion)
        self.human_poses[i, 2] = yaw

    def viz_cb(self, e):
        obs_msg = MarkerArray()
        for i, obs in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = "odom"  # or "base_link", "odom", etc.
            marker.header.stamp = rospy.Time.now()
            marker.ns = f"obs_{i}"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position of the square's center
            marker.pose.position.x = obs[0]
            marker.pose.position.y = obs[1]
            marker.pose.position.z = 0.5*self.Ds
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2*self.Ds  # Diameter in meters
            marker.scale.y = 2*self.Ds
            marker.scale.z = 2*self.Ds
            # Red color, fully opaque
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            obs_msg.markers.append(marker)

        self.obs_pub.publish(obs_msg)

    
    def di_dynamics(self, state, ctrl):
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
        return state + x_dot * self.dt

    def unicycle_dynamics(self, state, ctrl):
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
        x_new = x + v * ca.cos(theta) * self.dt
        y_new = y + v * ca.sin(theta) * self.dt
        theta_new = theta + omega * self.dt

        # Wrap the angle to be within [-pi, pi]
        # theta_new = ca.atan2(ca.sin(theta_new), ca.cos(theta_new))
        state_new = ca.vertcat(x_new, y_new, theta_new)
        return state_new


    def timer_cb(self, e):
        t_start = rospy.Time.now()
        positions_now = self.robot_poses[:, :2].copy()
        print("Im in ", positions_now[self.id])
        """
        ===============================
        Voronoi partitioning
        ===============================
        """
        dummy_points = np.zeros((5*self.robots_num, 2))
        dummy_points[:self.robots_num, :] = positions_now
        mirrored_points = mirror(positions_now, self.width)
        mir_pts = np.array(mirrored_points)
        dummy_points[self.robots_num:, :] = mir_pts

        # Voronoi partitioning
        vor = Voronoi(dummy_points)
        region = vor.point_region[self.id]
        poly_vert = []
        for vert in vor.regions[region]:
            v = vor.vertices[vert]
            poly_vert.append(v)

        poly = Polygon(poly_vert)

        # Limited range cell
        range_vert = []
        for th in np.arange(0, 2*np.pi, np.pi/10):
            vx = positions_now[self.id, 0] + self.half_range * np.cos(th)
            vy = positions_now[self.id, 1] + self.half_range * np.sin(th)
            range_vert.append((vx, vy))
        range_poly = Polygon(range_vert)
        lim_region = poly.intersection(range_poly)
        # polygons.append(lim_region)
        # robot = vor.points[idx]

        xmin, ymin, xmax, ymax = lim_region.bounds
        discr_points = 20
        qs = []
        for i in np.linspace(xmin, xmax, discr_points):
            for j in np.linspace(ymin, ymax, discr_points):
                pt_i = Point(i, j)
                if lim_region.contains(pt_i):
                    qs.append(np.array([i, j]))

        qs = np.array(qs)

        """
        ===============================
        Optimization problem
        ===============================
        """
        # 1. Variables
        nx = 3
        nu = 2
        U = ca.SX.sym('U', nu+self.humans_num, self.T)
        x0 = ca.SX.sym('x0', nx)
        m_var = ca.SX.sym('m', 2)
        c_var = ca.SX.sym('c', 2, 2)

        # 2. cost
        cost_expr = gmm_voronoi_cost_func(x0[:2], qs, self.means, self.covariances, self.weights)
        cost_fn = ca.Function('cost', [x0[:2]], [cost_expr])

        # 3. Human trajectory prediction
        human_covs = np.zeros((self.humans_num, self.T, 2, 2))
        human_preds = np.zeros((self.humans_num, self.T, 3))
        pred_0 = self.human_poses
        human_preds[:, 0] = pred_0
        Q = 0.1 * np.eye(2)
        human_covs[:, 0] = Q
        for h in range(self.humans_num):
            for i in range(1, self.T):
                human_preds[h, i, 0] = human_preds[h, i-1, 0] + 0.1 * np.cos(human_preds[h, i-1, 2])
                human_preds[h, i, 1] = human_preds[h, i-1, 1] + 0.1 * np.sin(human_preds[h, i-1, 2])
                human_preds[h, i, 2] = human_preds[h, i-1, 2]
                # human_covs[i, :, :] = (i+1) * 0.25 * np.eye(2) + 0.1 * np.random.rand(2, 2)
                human_covs[h, i, :, :] = np.eye(2) @ human_covs[h, i-1] @ np.eye(2).T + Q

        # 4. Build optimization problem
        obj = 0.0
        x_curr = x0
        g_list = []
        for k in range(self.T):
            obj += cost_fn(x_curr[:2])

            # Obstacle avoidance constraint: Ds**2 - ||x - x_obs||**2 < 0
            for obs in self.obstacles:
                dist_squared = ca.sumsqr(x_curr[:2] - obs)
                g_list.append(self.Ds**2 - dist_squared)
            
            # Human avoidance constraint
            # from "Probabilistic Collision Checking With Chance Constraints, Du Toit et al. 2011 (T-RO)"
            for h in range(self.humans_num):
                diff = x_curr[:2] - human_preds[h, k, :2]
                mahalanobis_dist_sq = diff.T @ ca.inv(human_covs[h, k]) @ diff
                coeff = ca.sqrt(ca.det(2*ca.pi * human_covs[h, k]))
                vol = ca.pi * self.Ds**2
                g_list.append(-2 * ca.log(coeff * self.prob / vol) - mahalanobis_dist_sq + U[2+h, k])
                # add slack var to cost
                w_k = 100 * (1 - k/self.T)
                obj += w_k * U[2+h,k]**2

            x_curr = self.unicycle_dynamics(x_curr, U[:nu, k])

            # Position constraints
            g_list.append(x_curr[0] - 0.5*self.width)
            g_list.append(x_curr[1] - 0.5*self.width)
            g_list.append(-x_curr[0] - 0.5*self.width)
            g_list.append(-x_curr[1] - 0.5*self.width)
        
        g = ca.vertcat(*g_list)

        # 5. Problem
        opt_vars = ca.vec(U)
        nlp = {
            'x': opt_vars,
            'f': obj,
            'g': g,
            'p': x0
        }
        x_init = self.robot_poses[self.id, :]

        opts = {'ipopt': {'print_level': 0, 'sb': 'yes', 'max_iter': 1000, 'tol': 1e-5}, 'print_time': False}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # 6. Bounds and initial guess
        u_min = np.concatenate(
            [np.array([-self.vmax, -self.wmax]),
            -np.inf * np.ones(self.humans_num)]
        )
        u_max = np.concatenate(
            [np.array([self.vmax, self.wmax]),
            np.inf * np.ones(self.humans_num)]
        )
        lbx = np.tile(u_min, self.T)
        ubx = np.tile(u_max, self.T)
        lbg = -np.inf * np.ones(g.shape)
        ubg = np.zeros(g.shape)
        u0_guess = np.zeros(opt_vars.shape)

        # solve
        sol = solver(x0=u0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=x_init)
        stats = solver.stats()
        if not stats['success']:
            print("[IPOPT] Optimization failed with status: ", stats['return_status'])
        
        u_opt = sol['x'].full().reshape(self.T, nu+self.humans_num)
        u_opt, slack = u_opt[:, :nu], u_opt[:, nu:]

        # Planned trajectory
        planned_traj = np.zeros((self.T+1, nx))
        planned_traj[0, :] = x_init
        for k in range(self.T):
            planned_traj[k+1, :] = self.unicycle_dynamics(planned_traj[k, :], u_opt[k, :]).full().squeeze(1)
        
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        t_now = rospy.Time.now()
        path_msg.header.stamp = t_now
        for p in planned_traj:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.header.stamp = t_now
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

        """
        ===============================
        Robot control
        ===============================
        """
        msg = Twist()
        msg.linear.x = u_opt[0, 0]
        msg.angular.z = u_opt[0, 1]
        self.vel_pub.publish(msg)
        t_end = rospy.Time.now()
        duration = (t_end - t_start).to_sec()
        print(f"Total time: {duration} s")


        
        
        if self.graphics:
            self.ax.cla()
            self.ax.contourf(self.X, self.Y, self.Z.reshape(self.X.shape), levels=10, cmap='YlOrRd', alpha=0.75)
            #self.ax.scatter(self.means[:, 0], self.means[:, 1], marker='*', color='tab:orange', label='GMM Means')
            for t in range(self.T):
                alpha = np.exp(-np.log(10) * t / self.T)
                # draw_ellipse(human_traj[s+1+t, :], human_covs[t], n_std=1, ax=ax, alpha=alpha)
                for h in range(self.humans_num):
                    draw_ellipse(human_preds[h, t, :2], human_covs[h, t], n_std=2, ax=self.ax, alpha=alpha)
            # ax.contourf(X, Y, alpha_human*human_pdf.reshape(X.shape), levels=10, cmap='Blues', alpha=0.5)
            for idx in range(self.robots_num):
                # ax.plot(robots_hist[:s+2, idx, 0], robots_hist[:s+2, idx, 1], label='Robot Trajectory', color='tab:blue')
                # ax.scatter(robots_hist[s+1, idx, 0], robots_hist[s+1, idx, 1], color='tab:blue')
                
                self.ax.plot(planned_traj[:, 0], planned_traj[:, 1], label='Planned Trajectory', color='tab:green')
                x, y = lim_region.exterior.xy
                self.ax.plot(x, y, c='tab:red')
            
            for obs in self.obstacles:
                self.ax.scatter(obs[0], obs[1], marker='x', color='k', label='Obstacle')
                xc = obs[0] + self.Ds * np.cos(np.linspace(0, 2*np.pi, 20))
                yc = obs[1] + self.Ds * np.sin(np.linspace(0, 2*np.pi, 20))
                self.ax.plot(xc, yc, c='k', label='Safety distance')
            # ax.legend()
            self.ax.set_aspect('equal', adjustable='box')   # keeps squares square
            self.ax.set_autoscale_on(False)                 # stop anything else changing it
            self.ax.set_xlim(-0.5*self.width, 0.5*self.width)
            self.ax.set_ylim(-0.5*self.width, 0.5*self.width)
            self.fig.canvas.draw()
            plt.pause(0.01) 
            plt.ioff()

        if self.save_video:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.contourf(self.X, self.Y, self.Z.reshape(self.X.shape), levels=10, cmap='YlOrRd', alpha=0.75)
            #self.ax.scatter(self.means[:, 0], self.means[:, 1], marker='*', color='tab:orange', label='GMM Means')
            for t in range(self.T):
                alpha = np.exp(-np.log(10) * t / self.T)
                # draw_ellipse(human_traj[s+1+t, :], human_covs[t], n_std=1, ax=ax, alpha=alpha)
                for h in range(self.humans_num):
                    draw_ellipse(human_preds[h, t, :2], human_covs[h, t], n_std=2, ax=ax, alpha=alpha)
            # ax.contourf(X, Y, alpha_human*human_pdf.reshape(X.shape), levels=10, cmap='Blues', alpha=0.5)
            for idx in range(self.robots_num):
                # ax.plot(robots_hist[:s+2, idx, 0], robots_hist[:s+2, idx, 1], label='Robot Trajectory', color='tab:blue')
                # ax.scatter(robots_hist[s+1, idx, 0], robots_hist[s+1, idx, 1], color='tab:blue')
                
                ax.plot(planned_traj[:, 0], planned_traj[:, 1], label='Planned Trajectory', color='tab:green')
                x, y = lim_region.exterior.xy
                ax.plot(x, y, c='tab:red')
            
            for obs in self.obstacles:
                ax.scatter(obs[0], obs[1], marker='x', color='k', label='Obstacle')
                xc = obs[0] + self.Ds * np.cos(np.linspace(0, 2*np.pi, 20))
                yc = obs[1] + self.Ds * np.sin(np.linspace(0, 2*np.pi, 20))
                ax.plot(xc, yc, c='k', label='Safety distance')
            # ax.legend()
            ax.set_aspect('equal', adjustable='box')   # keeps squares square
            ax.set_autoscale_on(False)                 # stop anything else changing it
            ax.set_xlim(-0.5*self.width, 0.5*self.width)
            ax.set_ylim(-0.5*self.width, 0.5*self.width)
            
            # Save to a buffer in memory
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)

            # Convert to NumPy array
            img = Image.open(buf)
            frame = np.array(img)
            if len(self.frames) < self.max_frames:
                self.frames.append(frame)


if __name__ == '__main__':
    node = Controller()
    rospy.spin()
    # if node.save_video:
    #     rospy.loginfo("Saving video...")
    #     # Write video
    #     with imageio.get_writer('video.mp4', fps=5) as writer:
    #         for frame in node.frames:
    #             writer.append_data(frame)

        





        


