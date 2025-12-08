import numpy as np
from matplotlib import pyplot as plt

try:
    import matplotlib

    matplotlib.use("TkAgg")

except:
    pass


def plot_camera_pyramid(ax, R, T, color="b", label="Cam", scale=0.1):
    """visualize a camera as a wireframe pyramid"""

    # position  -R-transpose * T
    R_inv = R.T
    pos = -R_inv @ T

    w = scale
    h = scale * 0.75
    z = scale * 1.5

    local_pts = np.array(
        [
            [0, 0, 0],  # center
            [w, h, z],  # top right
            [-w, h, z],  # top left
            [-w, -h, z],  # bot left
            [w, -h, z],  # bot right
        ]
    ).T

    # rotate to world space: P_world = R_inv * P_local + Pos
    world_pts = (R_inv @ local_pts) + pos
    pts = world_pts.T  # transpose for plotting

    # draw lines
    # from center to corners
    for i in range(1, 5):
        ax.plot(
            [pts[0, 0], pts[i, 0]],
            [pts[0, 1], pts[i, 1]],
            [pts[0, 2], pts[i, 2]],
            color=color,
        )

    base_indices = [1, 2, 3, 4, 1]
    ax.plot(
        pts[base_indices, 0], pts[base_indices, 1], pts[base_indices, 2], color=color
    )

    ax.text(pos[0, 0], pos[1, 0], pos[2, 0], label, color="black")

    return pos.flatten()


def visualize_3d_setup():
    # load data
    try:
        data = np.load('stereo_params.npz')
        T = data['T']
        R = data['R']
        print("Loaded calibration data.")
        print(f"Translation: {T.T}")
    except FileNotFoundError:
        print("Error: 'stereo_params.npz' not found.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot cam 1 (the reference) - identity matrix, zero translation
    pos1 = plot_camera_pyramid(ax, np.eye(3), np.zeros((3,1)), 'blue', "Cam 1 (Ref)")
    
    # plot cam 2 (relative to Cam 1)
    pos2 = plot_camera_pyramid(ax, R, T, 'red', "Cam 2")
    
    # auto-scale axes
    all_pos = np.vstack([pos1, pos2])
    min_xyz = np.min(all_pos, axis=0)
    max_xyz = np.max(all_pos, axis=0)
    mid_xyz = (min_xyz + max_xyz) / 2
    
    max_range = np.max(max_xyz - min_xyz) + 0.2
    
    ax.set_xlim(mid_xyz[0] - max_range/2, mid_xyz[0] + max_range/2)
    ax.set_ylim(mid_xyz[1] - max_range/2, mid_xyz[1] + max_range/2)
    ax.set_zlim(mid_xyz[2] - max_range/2, mid_xyz[2] + max_range/2)
    
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down)')
    ax.set_zlabel('Z (Forward)')
    ax.set_title("Stereo Camera Geometry")
    
    plt.show()

if __name__ == "__main__":
    visualize_3d_setup()
