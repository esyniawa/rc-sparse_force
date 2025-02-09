def norm_xy(xy: np.ndarray,
            x_bounds: Tuple[float, float] = PlanarArms.x_limits,
            y_bounds: Tuple[float, float] = PlanarArms.y_limits, ) -> np.ndarray:
    # Calculate the midpoints of x and y ranges
    x_mid = (x_bounds[0] + x_bounds[1]) / 2
    y_mid = (y_bounds[0] + y_bounds[1]) / 2

    # Calculate the half-ranges
    x_half_range = (x_bounds[1] - x_bounds[0]) / 2
    y_half_range = (y_bounds[1] - y_bounds[0]) / 2

    # Normalize to [-1, 1]
    normalized_x = (xy[0] - x_mid) / x_half_range
    normalized_y = (xy[1] - y_mid) / y_half_range

    return np.array((normalized_x, normalized_y))


def generate_random_movement(arm: str, min_distance: float = 50.):
    # Random joint angles
    init_shoulder_thetas, target_shoulder_thetas = np.random.uniform(low=PlanarArms.l_upper_arm_limit,
                                                                     high=PlanarArms.u_upper_arm_limit,
                                                                     size=2)

    init_elbow_thetas, target_elbow_thetas = np.random.uniform(low=PlanarArms.l_forearm_limit,
                                                               high=PlanarArms.u_forearm_limit,
                                                               size=2)

    init_thetas = np.array((init_shoulder_thetas, init_elbow_thetas))
    target_thetas = np.array((target_shoulder_thetas, target_elbow_thetas))

    # Calculate distance
    init_pos = PlanarArms.forward_kinematics(arm=arm,
                                             thetas=init_thetas,
                                             radians=True)[:, -1]

    target_pos = PlanarArms.forward_kinematics(arm=arm,
                                               thetas=target_thetas,
                                               radians=True)[:, -1]

    distance = np.linalg.norm(target_pos - init_pos)

    # If distance is too small, call function again
    if distance <= min_distance:
        return generate_random_movement(arm=arm, min_distance=min_distance)
    else:
        return init_thetas, target_thetas, init_pos, target_pos