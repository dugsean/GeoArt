"""
Main script for GeoArt data analysis and 3D model generation.

Usage:
    python main.py                          # use defaults (or config.json if present)
    python main.py --config my.json         # load JSON config file
    python main.py --height-step 50         # override individual settings
    python main.py --save-config out.json   # save current settings to JSON
"""

import argparse
import os
import sys
import gc

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg, load_config
from data_loader import load_and_process_data
from layer_generator import generate_layers
from svg_exporter import export_svgs
from visualization import create_3d_model

def parse_args():
    """Parse CLI arguments.  CLI values override JSON config values."""
    parser = argparse.ArgumentParser(
        description='GeoArt Data Analysis and 3D Model Generator')
    parser.add_argument('--config', default=None,
                        help='Path to JSON config file (default: config.json if present)')
    parser.add_argument('--save-config', default=None, metavar='PATH',
                        help='Save effective config to JSON file and exit')

    # Pipeline
    parser.add_argument('--data-dir', default=None,
                        help='Directory containing input TIFF files')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for results')
    parser.add_argument('--height-step', type=int, default=None,
                        help='Height step in meters per layer')
    parser.add_argument('--step-xy', type=int, default=None,
                        help='Decimation factor (2m * step_xy = resolution)')
    parser.add_argument('--blur-sigma', type=float, default=None,
                        help='Gaussian blur sigma for smoothing')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Contour threshold')
    parser.add_argument('--z-scale', type=float, default=None,
                        help='Z-axis scaling factor')
    parser.add_argument('--visual', action='store_true', default=None,
                        help='Enable visual rendering')
    parser.add_argument('--canton-name', default=None,
                        help='Name of canton to clip to')
    parser.add_argument('--canton-shapefile', default=None,
                        help='Path to canton shapefile')

    # Nesting overrides
    parser.add_argument('--nest-inner', dest='nest_allow_inner',
                        action='store_true', default=None,
                        help='Enable inner nesting (default: from config)')
    parser.add_argument('--no-nest-inner', dest='nest_allow_inner',
                        action='store_false', default=None,
                        help='Disable inner nesting')
    parser.add_argument('--raster-mm', type=float, default=None, dest='fab_raster_mm',
                        help='Fabrication raster cell size in mm')

    return parser.parse_args()

def apply_cli_overrides(args):
    """Apply CLI overrides on top of config defaults/JSON."""
    cli_map = {
        'data_dir':          args.data_dir,
        'output_dir':        args.output_dir,
        'height_step':       args.height_step,
        'step_xy':           args.step_xy,
        'blur_sigma':        args.blur_sigma,
        'threshold':         args.threshold,
        'z_scale':           args.z_scale,
        'visual':            args.visual,
        'canton_name':       args.canton_name,
        'canton_shapefile':  args.canton_shapefile,
        'nest_allow_inner':  args.nest_allow_inner,
        'fab_raster_mm':     args.fab_raster_mm,
    }
    for key, val in cli_map.items():
        if val is not None:
            cfg.set(key, val)

def main():
    args = parse_args()

    config_path = args.config
    if config_path is None and os.path.exists('config.json'):
        config_path = 'config.json'
    load_config(config_path)

    # CLI overrides
    apply_cli_overrides(args)
    cfg.apply_to_modules()

    # Save and exit if requested
    if args.save_config:
        cfg.save(args.save_config)
        return

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Loading and processing data...")
    height_data, transform, crs = load_and_process_data(
        cfg.data_dir, cfg.canton_name, cfg.canton_shapefile)

    print("Generating layers...")
    layers, z_list, coarse_shape = generate_layers(
        height_data, cfg.height_step, cfg.step_xy, cfg.blur_sigma,
        cfg.threshold, cfg.z_scale)

    print("Exporting SVG files...")
    export_svgs(layers, z_list, coarse_shape, cfg.output_dir, cfg.step_xy,
                cfg.height_step, cfg.z_scale)

    if cfg.visual:
        print("Creating 3D model...")
        create_3d_model(layers, z_list, coarse_shape, cfg.output_dir,
                        cfg.step_xy, cfg.height_step, cfg.z_scale)

if __name__ == "__main__":
    try:
        main()
    finally:
        print("Program STOP")
        gc.collect()
