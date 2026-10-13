"""
3D visualization module for GeoArt.
"""

import numpy as np
import pyvista as pv
from matplotlib import cm
import gc

def plt_color(frac):
    """Elevation-band color for each slab."""
    return cm.terrain(frac)[:3]

def create_3d_model(layer_masks, z_list, coarse_shape, output_dir, step_xy, 
                    height_step, z_scale, visual=True):
    """
    Create and visualize 3D model from layers.
    
    Args:
        layer_masks (list): List of layer masks
        z_list (list): List of elevation values
        coarse_shape (tuple): Shape of coarse data
        output_dir (str): Output directory
        step_xy (int): Decimation factor
        height_step (int): Height step in meters
        z_scale (float): Z-axis scaling factor
        visual (bool): Whether to display the visualization
    """
    CELL = step_xy * 2
    layer_thickness = height_step * z_scale
    base_z = min(z_list) * z_scale
    ny_grid, nx_grid = coarse_shape

    # --- Build and render ---------------------------------------------------------
    plotter = pv.Plotter(window_size=(1400, 1000))
    plotter.set_background("white")

    for i, (z, layer) in enumerate(zip(z_list, layer_masks)):
        if layer is None:
            continue

        print(f"Processing layer {i+1}/{len(z_list)}: z={z}")

        layer = np.flipud(layer)
        ny, nx = layer.shape

        grid = pv.ImageData(dimensions=(nx, ny, 2), spacing=(CELL, CELL, CELL))
        vol_data = np.stack([layer.T, layer.T], axis=-1)
        grid.point_data["values"] = vol_data.flatten(order="F").astype(np.uint8)

        slab = grid.threshold(0.5, scalars="values").extract_surface(algorithm="dataset_surface")
        del grid, layer
        
        if slab.n_points == 0:
            continue

        # Thickness of each physical layer in scaled space
        layer_thickness = height_step * z_scale
        slab = slab.extrude((0, 0, layer_thickness), capping=True)
        
        # Position each layer consecutively with zero gap: base + index * thickness
        current_z = (min(z_list) * z_scale) + (i * layer_thickness)
        slab.translate((0, 0, current_z), inplace=True)

        frac = i / max(len(z_list) - 1, 1)
        plotter.add_mesh(slab, color=plt_color(frac),
                         opacity=1.0, smooth_shading=True)
        del slab

    # --- Add a solid rectangular black base underneath ----------------------------
    ny, nx = coarse_shape
    xmin, ymin = 0, 0
    xmax, ymax = nx * CELL, ny * CELL

    base_thickness = height_step * z_scale
    base_z = (min(z_list) * z_scale) - base_thickness  # exactly one layer below the first

    base_box = pv.Box(bounds=(xmin, xmax, ymin, ymax, base_z, min(z_list) * z_scale))
    plotter.add_mesh(base_box, color="black", opacity=1.0)

    # --- Camera and render ---------------------------------------------------------
    plotter.add_axes()
    ny, nx = coarse_shape
    center_x = (nx * CELL) / 2
    center_y = (ny * CELL) / 2
    center_z = (max(z_list) * z_scale) / 2

    # Position the camera off to the side, high up, and looking down at the center
    plotter.camera_position = [
        (center_x - 1.5 * nx * CELL, center_y - 1.5 * ny * CELL, center_z + 40000), # Camera location (X, Y, Z)
        (center_x, center_y, center_z),                                            # Look-at focal point
        (0, 0, 1)                                                                  # View up vector (Z is up)
    ]
    plotter.camera.zoom(1.1)
    if visual:
        try:
            plotter.show()
        finally:
            # Ensure proper cleanup of the plotter
            plotter.close()
            print("close")
            gc.collect()
            
