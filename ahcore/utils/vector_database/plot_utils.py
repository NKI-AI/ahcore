import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.data.dataset import TiledWsiDataset
from PIL import Image, ImageDraw
from pymilvus.client.abstract import Hit
from shapely import Polygon


def plot_wsi_and_annotation_overlay(
    image: SlideImage,
    annotation_1: WsiAnnotations,
    annotation_2: WsiAnnotations,
    mpp: float,
    tile_size: tuple[int, int],
    filename_appendage: str,
) -> None:
    """Plots a whole slide image with debug tiles in green with alpha transparency."""
    # Get a scaled view of the whole slide image based on the specified microns per pixel (mpp).
    scaled_region_view = image.get_scaled_view(image.get_scaling(mpp))

    # Create two datasets: one masked by the annotation and another for the total area.
    dataset_masked = TiledWsiDataset.from_standard_tiling(
        image.identifier, mpp=mpp, tile_size=tile_size, tile_overlap=(0, 0), mask=annotation_1
    )

    dataset_masked_2 = TiledWsiDataset.from_standard_tiling(
        image.identifier, mpp=mpp, tile_size=tile_size, tile_overlap=(0, 0), mask=annotation_2
    )

    dataset_total = TiledWsiDataset.from_standard_tiling(
        image.identifier, mpp=mpp, tile_size=tile_size, tile_overlap=(0, 0), mask=None
    )

    # Create a blank RGBA image for output.
    output = Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))

    # Define the color and transparency for the debug tiles.
    green_with_transparency = (0, 255, 0, 128)  # Green with 50% opacity (128 out of 255)
    red_with_transparency = (255, 0, 0, 128)  # Red with 50% opacity (128 out of 255)

    # Add the 'normal' image everywhere first.
    for d in dataset_total:
        tile = d["image"].numpy()
        tile = Image.fromarray(tile)
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
        output.paste(tile, box)

    # Loop through each tile in the masked dataset.
    for d in dataset_masked:
        # Get the coordinates of the tile.
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))

        # Create a green tile with specified size and alpha transparency.
        green_tile = Image.new("RGBA", tile_size, green_with_transparency)

        # Create a temporary image slice of the output where the green tile will be pasted.
        background = output.crop(box)

        # Blend the green tile onto the background slice.
        blended = Image.alpha_composite(background, green_tile)

        # Paste the blended image back onto the output image.
        output.paste(blended, box)

    # Loop through each tile in the masked dataset.
    for d in dataset_masked_2:
        # Get the coordinates of the tile.
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))

        # Create a green tile with specified size and alpha transparency.
        red_tile = Image.new("RGBA", tile_size, red_with_transparency)

        # Create a temporary image slice of the output where the green tile will be pasted.
        background = output.crop(box)

        # Blend the green tile onto the background slice.
        blended = Image.alpha_composite(background, red_tile)

        # Paste the blended image back onto the output image.
        output.paste(blended, box)

    # Save the output image to a file.
    output.save(f"debug_dataset_overlay_{filename_appendage}.png")


def plot_wsi_and_annotation_from_filename(filename: str) -> None:
    """Plots a whole slide image."""
    tile_size = (10, 10)
    mpp = 20

    data_dir = os.environ.get("DATA_DIR")
    annotations_dir = os.environ.get("ANNOTATIONS_DIR")

    image_filename = Path(data_dir) / filename
    annotation_filename = Path(annotations_dir) / filename.replace(".svs", ".svs.geojson")

    image = SlideImage.from_file_path(image_filename)
    annotation = WsiAnnotations.from_geojson(annotation_filename)
    annotation.filter("T")

    scaled_region_view = image.get_scaled_view(image.get_scaling(mpp))
    dataset = TiledWsiDataset.from_standard_tiling(
        image_filename, mpp=mpp, tile_size=tile_size, tile_overlap=(0, 0), mask=annotation
    )

    output = Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))
    for d in dataset:
        tile = d["image"].numpy()
        tile = Image.fromarray(tile)
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
        output.paste(tile, box)
        draw = ImageDraw.Draw(output)
        draw.rectangle(box, outline="red")

    output.save("dataset_example.png")


def plot_wsi_and_annotation(
    image: SlideImage, annotation: WsiAnnotations, mpp: float, tile_size: tuple[int, int]
) -> None:
    """Plots a whole slide image."""
    scaled_region_view = image.get_scaled_view(image.get_scaling(mpp))
    dataset = TiledWsiDataset.from_standard_tiling(
        image.identifier, mpp=mpp, tile_size=tile_size, tile_overlap=(0, 0), mask=annotation
    )

    output = Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))
    for d in dataset:
        tile = d["image"].numpy()
        tile = Image.fromarray(tile)
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
        output.paste(tile, box)
        draw = ImageDraw.Draw(output)
        draw.rectangle(box, outline="red")

    output.save("dataset_example.png")


def plot_polygon_and_box(polygon: Polygon, rectangle: Polygon) -> None:
    # Create matplotlib figure and axes
    fig, ax = plt.subplots()

    # Plot the polygon
    x, y = polygon.exterior.xy  # Get the x and y coordinates of the polygon
    ax.fill(x, y, alpha=0.5, fc="blue", edgecolor="black", label="Polygon")  # Plot the polygon

    # Plot the rectangle
    rx, ry = rectangle.exterior.xy  # Get the x and y coordinates of the rectangle
    ax.fill(rx, ry, alpha=0.5, fc="red", edgecolor="black", label="Rectangle")  # Plot the rectangle

    # Add grid, legend, and show the plot
    ax.grid(True)
    ax.legend()
    plt.savefig("poly.png")
    plt.close()


def plot_tile(hit: Hit) -> None:
    """Plots a tile from a SlideImage."""
    filename, x, y, width, height, mpp = (
        hit.filename,
        hit.coordinate_x,
        hit.coordinate_y,
        hit.tile_size,
        hit.tile_size,
        hit.mpp,
    )
    data_dir = os.environ.get("DATA_DIR")
    filename = Path(data_dir) / filename

    image = SlideImage.from_file_path(filename)
    scaling = image.get_scaling(mpp)
    tile = image.read_region((x, y), scaling, (width, height))
    tile = tile.numpy().astype(np.uint8)
    plt.imshow(tile)
    plt.savefig("/home/e.marcus/projects/ahcore/debug_tile_plots/tile.png")
    plt.close()
