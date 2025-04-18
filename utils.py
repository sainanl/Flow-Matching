from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def get_checker_board_samples(
    N=1000,
    length=4,
    value_bound=4,
    resolution=100,
    save_figure=True,
    save_path="test.png",
    checker_signal=1,
):
    """_summary_

    Args:
        N (int, optional): _description_. Defaults to 1000.
        length (int, optional): _description_. Defaults to 4.
        value_bound (int, optional): _description_. Defaults to 4.
        resolution (int, optional): _description_. Defaults to 100.
        save_figure (bool, optional): _description_. Defaults to True.
        save_path (str, optional): _description_. Defaults to "test.png".
        checker_signal (int, optional): A darker checker will be sampled or a lighter one, setting to 1 samples from darker one, 0 for lighter one. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # Parameters
    # N = 1000  # Number of points to sample
    x_min, x_max = -value_bound, value_bound
    y_min, y_max = -value_bound, value_bound
    # resolution = 100  # Resolution of the grid

    # Create the grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # Checkerboard pattern
    length = 4
    checkerboard = np.indices((length, length)).sum(axis=0) % 2

    # Sample points in regions where checkerboard pattern is 1
    sampled_points = []
    while len(sampled_points) < N:
        # Randomly sample a point within the x and y range
        x_sample = np.random.uniform(x_min, x_max)
        y_sample = np.random.uniform(y_min, y_max)

        # Determine the closest grid index
        i = int((x_sample - x_min) / (x_max - x_min) * length)
        j = int((y_sample - y_min) / (y_max - y_min) * length)

        # Check if the sampled point is in a region where checkerboard == 1
        if checkerboard[j, i] == checker_signal:
            sampled_points.append((x_sample, y_sample))

    # Convert to NumPy array for easier plotting

    sampled_points = np.array(sampled_points)
    if save_figure:
        # Plot the checkerboard pattern
        plt.figure(figsize=(6, 6))
        plt.imshow(
            checkerboard,
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            cmap=ListedColormap(["purple", "yellow"]),
        )

        # Plot sampled points
        plt.scatter(sampled_points[:, 0], sampled_points[:, 1], color="red", marker="o")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        # plt.show()
        plt.savefig(save_path)
    return sampled_points


def get_n_colors(n, cmap_name="tab20"):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / n) for i in range(n)]


def visualize_points(distributions, save_path):
    plt.figure(figsize=(6, 6))
    colors = get_n_colors(len(distributions))
    # print(colors)
    for distribution, color in zip(distributions, colors):
        print(distribution.shape, color)
        plt.scatter(distribution[:, 0], distribution[:, 1], color=color, marker="o")
    # plt.show()
    plt.savefig(save_path)


def generate_gif(image_dir, output_path, ext):
    # get all PNG files and sort them by name
    png_files = sorted([f for f in os.listdir(image_dir) if f.endswith(ext)])

    # load images
    images = [Image.open(os.path.join(image_dir, f)) for f in png_files]

    # Save as GIF
    if images:
        if os.path.exists(output_path):
            output_path = os.path.join(
                output_path, os.path.basename(image_dir) + ".gif"
            )
        else:
            output_path = image_dir + ".gif"
        print(f"Saving to {output_path}")
        images[0].save(
            output_path, save_all=True, append_images=images[1:], duration=500, loop=0
        )
        print(f"GIF saved at {output_path}")
    else:
        print("No PNG files found.")
