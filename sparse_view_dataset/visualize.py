from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation
import pyvista as pv
import torch

def plot_cloud_and_projs(gif_path: Path, cloud: torch.Tensor, projs: torch.Tensor) -> None:
    n_proj, h, w = projs.shape
    n_proj_, _, _ = cloud.shape
    assert n_proj == n_proj_

    x = np.linspace(0, 1, w)
    z = np.linspace(0, 1, h)
    x, z = np.meshgrid(x, z)
    y = np.ones_like(x)
    grid = pv.StructuredGrid(x, y, z)
    grid["value"] = projs[0].cpu().numpy().flatten()
    poly = pv.PolyData(cloud[0].cpu().numpy())
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(gif_path)
    plotter.add_mesh(grid, scalars="value", cmap="gray")
    plotter.add_mesh(poly, color="red", point_size=3, render_points_as_spheres=True)
    plotter.show_bounds(grid="back", location="outer", all_edges=True)  #type: ignore
    plotter.camera_position = "xz"
    plotter.camera.azimuth = -20
    plotter.camera.elevation = 10
    plotter.show(auto_close=False)

    for i in range(n_proj):
        grid["value"] = projs[i].cpu().numpy().flatten()
        poly.points = cloud[i].cpu().numpy()
        plotter.write_frame()
    plotter.close()


def save_gif(output_path: Path, frames: torch.Tensor | np.ndarray, fps_gif: int = 10, **imshow_kwargs) -> None:
    matplotlib.use("Agg")
    frames = frames.squeeze()
    if isinstance(frames, torch.Tensor):
        frames_np = frames.cpu().numpy()
    else:
        frames_np = frames

    if "vmin" not in imshow_kwargs or "vmax" not in imshow_kwargs:
        vmin, vmax = np.percentile(frames_np, [0.05, 99.5])
        imshow_kwargs.setdefault("vmin", vmin)
        imshow_kwargs.setdefault("vmax", vmax)

    h, w = frames_np.shape[1], frames_np.shape[2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes((0, 0, 1, 1))
    ax.axis("off")
    ims = []
    for i in range(frames_np.shape[0]):
        im = ax.imshow(frames_np[i], animated=True, **imshow_kwargs)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=1000 / fps_gif, blit=True, repeat_delay=1000)
    writer = animation.PillowWriter(fps=fps_gif)
    ani.save(output_path, writer=writer, dpi=dpi, savefig_kwargs={"pad_inches": 0})
    plt.close(fig)
