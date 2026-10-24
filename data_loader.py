import glob
import gc
import os

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.features import geometry_mask

def load_and_process_data(data_dir, canton_name, canton_shapefile_path):
    """
    Load and process raster data for analysis.
    
    Args:
        data_dir (str): Directory containing input TIFF files
        canton_name (str): Name of canton to clip to
        canton_shapefile_path (str): Path to canton shapefile
        
    Returns:
        tuple: (height_array, transform, crs)
    """
    datasets = [rasterio.open(p) for p in sorted(glob.glob(os.path.join(data_dir, "*.tif")))]
    crs = datasets[0].crs                      # keep before closing!
    mosaic, transform = merge(datasets)
    for ds in datasets:
        ds.close()
    del datasets
    gc.collect()

    height = np.asarray(mosaic[0], dtype=np.float32)
    del mosaic
    gc.collect()

    cantons = gpd.read_file(canton_shapefile_path)
    ne = cantons[cantons["NAME"].str.contains(canton_name, case=False, na=False)]
    ne = ne.to_crs(crs)
    assert len(ne) == 1, f"Expected 1 {canton_name} polygon, got {len(ne)}"

    mask = geometry_mask(
        [geom for geom in ne.geometry],
        out_shape=height.shape,
        transform=transform,
        invert=True,                            # True inside the canton
    )

    height[~mask] = np.nan                      # clip in place — no extra copies
    del mask
    gc.collect()
    
    return height, transform, crs