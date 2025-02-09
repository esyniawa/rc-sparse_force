import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from typing import Tuple, List


def create_dh_matrix(theta, alpha, a, d, radians=True):
    if not radians:
        alpha = np.radians(alpha)
        theta = np.radians(theta)

    A = np.array(
        [[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
         [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
         [0, np.sin(alpha), np.cos(alpha), d],
         [0, 0, 0, 1]])

    return A


class PlanarArm:
    # joint limits
    l_upper_arm_limit, u_upper_arm_limit = np.radians((-5., 175.))  # in degrees [°]
    l_forearm_limit, u_forearm_limit = np.radians((-5., 175.))  # in degrees [°]

    # xy limits
    x_limits = (-450, 450)
    y_limits = (-50, 400)

    # DH parameter
    scale = 1.0
    shoulder_length = scale * 50.0  # in [mm]
    upper_arm_length = scale * 220.0  # in [mm]
    forearm_length = scale * 160.0  # in [mm]


    def __init__(self, arm: str = 'right'):
        assert arm in ['left', 'right'], 'Arm must be "left" or "right"'

        self.arm = arm
        const = 1 if arm == 'right' else -1
        # shoulder translation
        self.A0 = create_dh_matrix(a=const * PlanarArm.shoulder_length, d=0, alpha=0, theta=0)

    def forward_kinematics(self, thetas: np.ndarray, radians: bool = False, check_limits: bool = True):

        if check_limits:
            theta1, theta2 = PlanarArm.check_values(thetas, radians)
        else:
            theta1, theta2 = thetas

        if self.arm == 'left':
            theta1 = np.pi - theta1
            theta2 = -theta2

        A1 = create_dh_matrix(a=PlanarArm.upper_arm_length, d=0,
                              alpha=0, theta=theta1)

        A2 = create_dh_matrix(a=PlanarArm.forearm_length, d=0,
                              alpha=0, theta=theta2)

        # Shoulder -> elbow
        A01 = self.A0 @ A1
        # Elbow -> hand
        A12 = A01 @ A2

        return np.column_stack(([0, 0], self.A0[:2, 3], A01[:2, 3], A12[:2, 3]))

    def inverse_kinematics(self,
                           end_effector: np.ndarray,
                           starting_angles: np.ndarray,
                           max_iterations: int = 1_000,
                           abort_criteria: float = 1e-6,
                           radians: bool = False):

        if not radians:
            starting_angles = np.radians(starting_angles)

        def objective(thetas):
            current_position = self.forward_kinematics(thetas=thetas, radians=True, check_limits=False)[:, -1]
            position_error = np.linalg.norm(current_position - end_effector) ** 2

            # Add a term to penalize large changes from the starting angles
            angle_change_penalty = np.linalg.norm(thetas - starting_angles) ** 2

            return position_error + 0.5 * angle_change_penalty

        if objective(starting_angles) < abort_criteria:
            return starting_angles
        else:
            bounds = [
                (PlanarArm.l_upper_arm_limit, PlanarArm.u_upper_arm_limit),
                (PlanarArm.l_forearm_limit, PlanarArm.u_forearm_limit)
            ]

            result = minimize(
                objective,
                starting_angles,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': abort_criteria, 'maxiter': max_iterations}
            )

            if result.success:
                return result.x
            else:
                # If optimization fails, try a different starting point
                new_starting_angles = np.array([
                    (PlanarArm.l_upper_arm_limit + PlanarArm.u_upper_arm_limit) / 2,
                    (PlanarArm.l_forearm_limit + PlanarArm.u_forearm_limit) / 2
                ])
                result = minimize(
                    objective,
                    new_starting_angles,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': abort_criteria, 'maxiter': max_iterations}
                )

                return result.x if result.success else starting_angles

    def generate_random_target(self,
                               current_pos: np.ndarray,
                               min_dist: float = 50,) -> Tuple[np.ndarray, np.ndarray]:

        # Generate random angles within limits
        theta_shoulder = np.random.uniform(PlanarArm.l_upper_arm_limit, PlanarArm.u_upper_arm_limit)
        theta_elbow = np.random.uniform(PlanarArm.l_forearm_limit, PlanarArm.u_forearm_limit)

        # Calculate end-effector positions for these angles
        target_pos = self.forward_kinematics(np.array([theta_shoulder, theta_elbow]), radians=True, check_limits=False)[:, -1]
        distance = np.linalg.norm(target_pos - current_pos)

        # If distance is too small or target out of bounds, call function again
        if distance < min_dist or not self.check_if_in_bounds(target_pos):
            return self.generate_random_target(current_pos, min_dist=min_dist)
        else:
            return np.array([theta_shoulder, theta_elbow]), target_pos

    def norm_xy(self, xy: np.ndarray, ) -> np.ndarray:
        # Calculate the midpoints of x and y ranges
        x_mid = (self.x_limits[0] + self.x_limits[1]) / 2
        y_mid = (self.y_limits[0] + self.y_limits[1]) / 2

        # Calculate the half-ranges
        x_half_range = (self.x_limits[1] - self.x_limits[0]) / 2
        y_half_range = (self.y_limits[1] - self.y_limits[0]) / 2

        # Normalize to [-1, 1]
        normalized_x = (xy[0] - x_mid) / x_half_range
        normalized_y = (xy[1] - y_mid) / y_half_range

        return np.array((normalized_x, normalized_y))

    def norm_distance(self, distance: np.ndarray,) -> np.ndarray:
        # Calculate x and y ranges
        x_range = abs(self.x_limits[1] - self.x_limits[0])
        y_range = abs(self.y_limits[1] - self.y_limits[0])

        # Normalize the distance components to [-1, 1]
        normalized_dx = distance[0] / x_range
        normalized_dy = distance[1] / y_range

        return np.array([normalized_dx, normalized_dy])

    def check_if_in_bounds(self, xy: np.ndarray) -> bool:
        return (self.x_limits[0] <= xy[0] <= self.x_limits[1]) and (self.y_limits[0] <= xy[1] <= self.y_limits[1])

    @staticmethod
    def check_values(thetas: np.ndarray, radians: bool = False) -> np.ndarray:
        """Check and clamp joint angles within limits."""
        theta1, theta2 = thetas
        if not radians:
            theta1 = np.radians(theta1)
            theta2 = np.radians(theta2)

        theta1 = np.clip(theta1, PlanarArm.l_upper_arm_limit, PlanarArm.u_upper_arm_limit)
        theta2 = np.clip(theta2, PlanarArm.l_forearm_limit, PlanarArm.u_forearm_limit)

        return np.array((theta1, theta2))


class PlanarArmTrajectory(PlanarArm):
    def __init__(self, arm: str = 'right', num_ik_points: int = 20, num_trajectory_points: int = 200):
        """
        Initialize the trajectory planner for the planar arm.
        :param arm: Arm to plan for
        :param num_ik_points: Number of points to calculate using inverse kinematics
        :param num_trajectory_points: Number of points in the final trajectory after interpolation
        """
        self.num_ik_points = num_ik_points
        self.num_trajectory_points = num_trajectory_points
        super().__init__(arm=arm)

    @staticmethod
    def minimum_jerk_trajectory(t):
        """
        Calculate minimum jerk trajectory scaling factor.
        Provides smooth acceleration and deceleration.
        """
        t3 = t ** 3
        t4 = t ** 4
        t5 = t ** 5
        return 10 * t3 - 15 * t4 + 6 * t5

    def plan_trajectory(self,
                        current_angles: np.ndarray,
                        target_coord: np.ndarray,
                        waiting_steps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plan a straight-line trajectory from current position to target coordinate.
        Uses fewer IK calculations and interpolates to get smooth trajectory.
        """
        # Verify current angles are within limits
        current_angles = self.check_values(current_angles, radians=True)

        # Get current end-effector position
        current_pos = self.forward_kinematics(current_angles, radians=True)[:, -1]

        # Create minimum-jerk trajectory in Cartesian space
        t_ik = np.linspace(0, 1, self.num_ik_points)
        s = self.minimum_jerk_trajectory(t_ik)

        x = current_pos[0] + (target_coord[0] - current_pos[0]) * s
        y = current_pos[1] + (target_coord[1] - current_pos[1]) * s
        cartesian_trajectory_sparse = np.column_stack((x, y))

        # Generate joint angle trajectory using inverse kinematics for sparse points
        joint_trajectory_sparse = np.zeros((self.num_ik_points, 2))
        joint_trajectory_sparse[0] = current_angles

        # Use the previous solution as initial guess for the next point
        for i in range(1, self.num_ik_points):
            joint_trajectory_sparse[i] = self.inverse_kinematics(
                cartesian_trajectory_sparse[i],
                joint_trajectory_sparse[i - 1],
                radians=True
            )

        # Create smooth splines for joint angles using sparse points
        theta1_spline = CubicSpline(t_ik, joint_trajectory_sparse[:, 0])
        theta2_spline = CubicSpline(t_ik, joint_trajectory_sparse[:, 1])

        # Generate dense trajectory using splines
        t_dense = np.linspace(0, 1, self.num_trajectory_points)
        joint_trajectory = np.column_stack((
            theta1_spline(t_dense),
            theta2_spline(t_dense)
        ))

        # Calculate corresponding Cartesian positions for dense trajectory
        cartesian_trajectory = np.zeros((self.num_trajectory_points, 2))
        for i in range(self.num_trajectory_points):
            positions = self.forward_kinematics(joint_trajectory[i], radians=True)
            cartesian_trajectory[i] = positions[:, -1]

        # Add waiting period at the start
        if waiting_steps > 0:
            joint_trajectory = np.vstack((np.tile(joint_trajectory[0], (waiting_steps, 1)),
                                          joint_trajectory,))
            cartesian_trajectory = np.vstack((np.tile(cartesian_trajectory[0], (waiting_steps, 1)),
                                              cartesian_trajectory,))

        return joint_trajectory, cartesian_trajectory

    def get_trajectory_velocities(self, joint_trajectory: np.ndarray, duration: float = 1.0) -> np.ndarray:
        """
        Calculate joint velocities from the trajectory.
        """
        dt = duration / (len(joint_trajectory) - 1)
        velocities = np.gradient(joint_trajectory, dt, axis=0)
        return velocities

    def get_trajectory_accelerations(self, joint_trajectory: np.ndarray, duration: float = 1.0) -> np.ndarray:
        """
        Calculate joint accelerations from the trajectory.
        """
        velocities = self.get_trajectory_velocities(joint_trajectory, duration)
        dt = duration / (len(joint_trajectory) - 1)
        accelerations = np.gradient(velocities, dt, axis=0)
        return accelerations


# Example usage:
if __name__ == "__main__":
    arm = PlanarArmTrajectory(arm='right', num_ik_points=11, num_trajectory_points=100)

    # Define initial joint angles (in radians)
    initial_angles = np.array([np.pi / 4, np.pi / 3])
    target = np.array([300, 200])

    # Generate trajectory
    joint_traj, cart_traj = arm.plan_trajectory(initial_angles, target)

    # Calculate velocities and accelerations (assuming 2-second movement)
    velocities = arm.get_trajectory_velocities(joint_traj, duration=4.0)
    accelerations = arm.get_trajectory_accelerations(joint_traj, duration=4.0)

    # Plotting
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))

    # Plot Cartesian trajectory
    plt.subplot(131)
    plt.plot(cart_traj[:, 0], cart_traj[:, 1], 'b-')
    plt.plot(cart_traj[0, 0], cart_traj[0, 1], 'go', label='Start')
    plt.plot(cart_traj[-1, 0], cart_traj[-1, 1], 'ro', label='End')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.title('Cartesian Trajectory')
    plt.legend()

    # Plot joint angles
    plt.subplot(132)
    plt.plot(joint_traj[:, 0], label='$\\theta_{shoulder}$')
    plt.plot(joint_traj[:, 1], label='$\\theta_{elbow}$')
    plt.grid(True)
    plt.xlabel('Time step')
    plt.ylabel('Angle [rad]')
    plt.title('Joint Angles')
    plt.legend()

    # Plot velocities
    plt.subplot(133)
    plt.plot(velocities[:, 0], label='$\\dot{\\theta}_{shoulder}$')
    plt.plot(velocities[:, 1], label='$\\dot{\\theta}_{elbow}$')
    plt.grid(True)
    plt.xlabel('Time step')
    plt.ylabel('Angular velocity [rad/s]')
    plt.title('Joint Velocities')
    plt.legend()

    plt.tight_layout()
    plt.show()
