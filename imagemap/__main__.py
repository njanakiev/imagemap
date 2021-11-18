import os
import logging
import argparse
import pandas as pd
from imagemap import imagetiles

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        prog="imagemap",
        description="Creates tiles from georeferenced images")
    parser.add_argument(action='store',
        dest='src_filepath',
        help='input folder path')
    parser.add_argument(action='store',
        dest='dst_filepath',
        help='output folderpath or filepath')
    parser.add_argument('-s', '--steps', action='store',
        type=int, choices=range(0, 5), default=0,
        dest='steps',
        help='Number of steps for gridded images')
    parser.add_argument('--min-zoom', action='store',
        type=int, choices=range(0, 12), default=0,
        dest='min_zoom',
        help='Minimum zoom')
    parser.add_argument('--max-zoom', action='store',
        type=int, choices=range(1, 18), default=6,
        dest='max_zoom',
        help='Maximum zoom')
    parser.add_argument('--output-type', action='store',
        dest='output_type', choices=['folder', 'mbtiles'],
        default='folder',
        help='Output type')
    parser.add_argument('--gridded', action='store_true',
        dest='gridded_tiles', default=False,
        help='Create gridded tiles output')
    parser.add_argument('--non-geographic', action='store_true',
        dest='non_geographic', default=False,
        help='Non-geographic coordinates between 0 and 1.')
    parser.add_argument('-v', '--verbose', action='store_true',
        dest='verbose', default=False,
        help='Verbose console output')
    parser.add_argument('--overwrite', action='store_true',
        dest='overwrite', default=False,
        help='Overwrite existing output')
    args = parser.parse_args()


    if not os.path.isdir(args.src_filepath):
        df = pd.read_csv(args.src_filepath)
    else:
        raise NotImplementedError("Folder are not supported")

    imagetiles.create_grid_tiles(
        df,
        args.dst_filepath,
        output_type=args.output_type,
        min_zoom=args.min_zoom, 
        max_zoom=args.max_zoom, 
        steps=args.steps,
        gridded_tiles=args.gridded_tiles,
        non_geographic=args.non_geographic,
        overwrite=args.overwrite,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
