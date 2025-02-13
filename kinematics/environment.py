import numpy as np
from .planar_arm import PlanarArm


class ReachingEnvironment:
    def __init__(self,
                 init_thetas: np.ndarray,
                 arm: str = 'right',
                 min_dist: float = 50,):
        """
        Environment for reaching task

        :param init_thetas: initial angles in radians
        :param arm: left or right arm
        :param min_dist: minimum distance between end-effector and target
        """

        # parameters
        self.arm_model = PlanarArm(arm=arm)
        self.min_dist = min_dist

        # current infos
        self.current_thetas = init_thetas
        self.current_pos = self.get_position(self.current_thetas)

        # target infos
        self.target_thetas, self.target_pos = self.random_target()
        self.norm_distance = self.arm_model.norm_distance(self.target_pos - self.current_pos)

    def random_target(self):
        self.target_thetas, self.target_pos = self.arm_model.generate_random_target(current_pos=self.current_pos,
                                                                                    min_dist=self.min_dist)
        return self.target_thetas, self.target_pos

    def get_position(self, thetas: np.ndarray, radians: bool = True, check_bounds: bool = True) -> np.ndarray:
        return self.arm_model.forward_kinematics(thetas=thetas, radians=radians, check_limits=check_bounds)[:, -1]

    def reset(self):
        self.target_thetas, self.target_pos = self.random_target()

        return np.concatenate([np.sin(self.current_thetas),
                               np.cos(self.current_thetas),
                               self.arm_model.norm_distance(self.target_pos - self.current_pos)])

    def step(self,
             action: np.ndarray,
             abort_criteria: float = 2,  # in [mm]
             scale_angle_change: float = np.radians(5),
             clip_thetas: bool = True,
             clip_penalty: bool = True):

        # Calculate new angles
        self.current_thetas += action * scale_angle_change
        reward = 0.
        # give penalty if action leads to out of joint bounds
        if clip_penalty:
            if self.current_thetas[0] < PlanarArm.l_upper_arm_limit or self.current_thetas[0] > PlanarArm.u_upper_arm_limit:
                reward -= 5.
            if self.current_thetas[1] < PlanarArm.l_forearm_limit or self.current_thetas[1] > PlanarArm.u_forearm_limit:
                reward -= 5.

        if clip_thetas:
            self.current_thetas = PlanarArm.check_values(self.current_thetas, radians=True)

        # Calculate new position
        self.current_pos = self.get_position(self.current_thetas, check_bounds=False)

        # Calculate error + reward
        distance = self.target_pos - self.current_pos
        self.norm_distance = self.arm_model.norm_distance(distance)

        error = np.linalg.norm(distance)
        reward += -1e-3 * error  # in [m]
        done = error < abort_criteria
        if done:
            reward += 5.

        return np.concatenate([np.sin(self.current_thetas),
                               np.cos(self.current_thetas),
                               self.norm_distance]), reward, done

if __name__ == '__main__':
    env = ReachingEnvironment(init_thetas=np.zeros(2), arm='right')
    print(env.reset())
    for _ in range(1000):
        action = np.random.uniform(-1, 1, size=2)
        state, reward, done = env.step(action)
        print(f'State: {state}, Reward: {reward}, Done: {done}')