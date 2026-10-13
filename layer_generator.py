"""
Layer generation module for GeoArt.
"""

import gc
import numpy as np
from scipy.ndimage import gaussian_filter, find_objects
from skimage.measure import find_contours

# ----------------------------------------------------------------------------

def generate_layers(height_data, height_step, step_xy, blur_sigma, threshold, z_scale):
    """
    Generate elevation layers from height data.
    
    Args:
        height_data (np.array): Processed height data array
        height_step (int): Height step in meters per layer
        step_xy (int): Decimation factor
        blur_sigma (float): Gaussian blur sigma for smoothing
        threshold (float): Contour threshold
        z_scale (float): Z-axis scaling factor
        
    Returns:
        tuple: (layer_masks, z_list, coarse_shape)
    """
    # --- Decimate and crop to the canton extent ------------------------------------
    coarse = height_data[::step_xy, ::step_xy]
    del height_data
    gc.collect()

    valid = np.isfinite(coarse)
    slices = find_objects(valid.astype(np.uint8))[0]
    coarse = coarse[slices]
    del valid
    gc.collect()

    minHeight = int(np.nanmin(coarse))
    maxHeight = int(np.nanmax(coarse))
    n_layers = (maxHeight - minHeight) // height_step + 1
    print(f"Extent {minHeight}-{maxHeight} m | grid {coarse.shape} | {n_layers} layers")

    # Track the actual z positions used
    z_start = minHeight
    z_list = list(range(minHeight, maxHeight + height_step, height_step))
    
    # --- Pre-compute all valid layer masks so we can easily look ahead -------------
    layer_masks = []
    i=0
    for z in z_list:
        i = i+1
        layer = (coarse > z).astype(np.float32)
        layer = gaussian_filter(layer, blur_sigma) < threshold
        print(f"Layer {i} has sum: {layer.sum()}")
        layer_masks.append(layer)
        #if layer.sum() < 100 or layer.all():
        #    layer_masks.append(None)
        #else:
        #    layer_masks.append(layer)

    return layer_masks, z_list, coarse.shape
