"""
Main script for GeoArt data analysis and 3D model generation.
"""

import argparse
import os
import sys
import gc

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_process_data
from layer_generator import generate_layers
from svg_exporter import export_svgs
from visualization import create_3d_model

def main():
    parser = argparse.ArgumentParser(description='GeoArt Data Analysis and 3D Model Generator')
    parser.add_argument('--data-dir', default='data20', help='Directory containing input TIFF files')
    parser.add_argument('--output-dir', default='laser_output', help='Output directory for results')
    parser.add_argument('--height-step', type=int, default=100, help='Height step in meters per layer')
    parser.add_argument('--step-xy', type=int, default=15, help='Decimation factor (2m * step_xy = resolution)')
    parser.add_argument('--blur-sigma', type=float, default=5.0, help='Gaussian blur sigma for smoothing')
    parser.add_argument('--threshold', type=float, default=0.5, help='Contour threshold')
    parser.add_argument('--z-scale', type=float, default=2.0, help='Z-axis scaling factor')
    parser.add_argument('--visual', action='store_true', help='Enable visual rendering')
    parser.add_argument('--canton-name', default='Neuch', help='Name of canton to clip to')
    parser.add_argument('--canton-shapefile', default='./swissboundaries3d_2026-01_2056_5728.shp/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp', help='Path to canton shapefile')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading and processing data...")
    height_data, transform, crs = load_and_process_data(args.data_dir, args.canton_name, args.canton_shapefile)
    
    print("Generating layers...")
    layers, z_list, coarse_shape = generate_layers(height_data, args.height_step, 
                                                   args.step_xy, args.blur_sigma, 
                                                   args.threshold, args.z_scale)
    
    print("Exporting SVG files...")
    export_svgs(layers, z_list, coarse_shape, args.output_dir, args.step_xy, 
                args.height_step, args.z_scale)
    
    if(args.visual):
        print("Creating 3D model...")
        create_3d_model(layers, z_list, coarse_shape, args.output_dir, args.step_xy,
                        args.height_step, args.z_scale)

if __name__ == "__main__":
    try:
        main()
    finally:
        print("Program STOP")
        gc.collect()

