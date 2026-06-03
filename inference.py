"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import json
import traceback
from pathlib import Path

import numpy as np
import SimpleITK

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run(model=None):
    """Run inference: read inputs, process, write outputs."""
    try:
        # The key is a tuple of the slugs of the input sockets
        interface_key = get_interface_key()

        # Lookup the handler for this particular set of sockets (i.e. the interface)
        handler = {
            ("generic-medical-image",): interf0_handler,
        }[interface_key]

        # Call the handler
        return handler()
    except Exception:
        # Print any exception so it shows up in the container logs
        traceback.print_exc()
        return 1


def interf0_handler():
    # Determine the input image location from socket relative_path
    image_location = get_image_location("generic-medical-image")
    print(f"Looking for input image in: {image_location}", flush=True)

    # Read the input image (an .mha region extracted via sample_mlab_image_region)
    input_image = load_image_file(location=image_location)

    print(f"Input image dimension: {input_image.GetDimension()}", flush=True)
    print(f"Input image size: {input_image.GetSize()}", flush=True)
    print(f"Input image spacing: {input_image.GetSpacing()}", flush=True)
    print(f"Input image origin: {input_image.GetOrigin()}", flush=True)
    print(f"Input image pixel type: {input_image.GetPixelIDTypeAsString()}", flush=True)
    print(f"Input image num components: {input_image.GetNumberOfComponentsPerPixel()}", flush=True)

    # Generate a sphere overlay of ~1/4 the size of the input region
    output_image = create_sphere_overlay(input_image)

    # Save your output
    output_location = OUTPUT_PATH / "images" / "generic-overlay"
    write_image_file(
        location=output_location,
        image=output_image,
    )

    return 0


def create_sphere_overlay(input_image: SimpleITK.Image) -> SimpleITK.Image:
    """
    Creates a binary sphere/circle overlay centered in the input image.
    The sphere radius is approximately 1/4 of the smallest spatial dimension.
    The output preserves the same spatial size, spacing, origin, and direction
    as the input (ignoring any vector components).
    Output is scalar uint8 with 0 for background and 255 for the sphere.
    """
    ndim = input_image.GetDimension()
    size = np.array(input_image.GetSize())
    spacing = np.array(input_image.GetSpacing())

    # Sphere radius: 1/4 of the smallest spatial dimension in physical units
    min_dim_voxels = size.min()
    radius_voxels = min_dim_voxels / 4.0
    radius_physical = radius_voxels * spacing.min()

    print(f"Creating sphere overlay: ndim={ndim}, size={size}, spacing={spacing}", flush=True)
    print(f"Radius: {radius_voxels:.1f} voxels, {radius_physical:.3f} physical units", flush=True)

    if ndim == 3:
        nx, ny, nz = int(size[0]), int(size[1]), int(size[2])
        center = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0])

        # Compute slice by slice to keep memory usage reasonable
        sphere_mask = np.zeros((nz, ny, nx), dtype=np.uint8)
        radius_sq = radius_physical ** 2
        for iz in range(nz):
            dz_sq = ((iz - center[2]) * spacing[2]) ** 2
            if dz_sq > radius_sq:
                continue
            y_coords = np.arange(ny, dtype=np.float64)
            x_coords = np.arange(nx, dtype=np.float64)
            dy_sq = ((y_coords - center[1]) * spacing[1]) ** 2
            dx_sq = ((x_coords - center[0]) * spacing[0]) ** 2
            dist_sq = dy_sq[:, np.newaxis] + dx_sq[np.newaxis, :] + dz_sq
            sphere_mask[iz] = np.where(
                dist_sq <= radius_sq, np.uint8(255), np.uint8(0)
            )
    elif ndim == 2:
        nx, ny = int(size[0]), int(size[1])
        center = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0])
        y, x = np.mgrid[0:ny, 0:nx]
        dist_sq = (
            ((x - center[0]) * spacing[0]) ** 2
            + ((y - center[1]) * spacing[1]) ** 2
        )
        sphere_mask = np.where(
            dist_sq <= radius_physical ** 2, np.uint8(255), np.uint8(0)
        )
    else:
        raise ValueError(f"Unsupported image dimension: {ndim}")

    # Convert to SimpleITK image preserving spatial metadata
    output_image = SimpleITK.GetImageFromArray(sphere_mask)
    output_image.SetOrigin(input_image.GetOrigin())
    output_image.SetSpacing(input_image.GetSpacing())
    output_image.SetDirection(input_image.GetDirection())

    print(f"Output image size: {output_image.GetSize()}", flush=True)
    print(f"Output image pixel type: {output_image.GetPixelIDTypeAsString()}", flush=True)

    return output_image


def get_interface_key():
    """Read inputs.json and return a sorted tuple of socket slugs."""
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def get_image_location(slug: str) -> Path:
    """
    Get the file system location of an image input by its socket slug.
    Uses the relative_path from inputs.json.
    """
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    for sv in inputs:
        if sv["socket"]["slug"] == slug:
            relative_path = sv["socket"].get("relative_path", "")
            location = INPUT_PATH / relative_path
            print(f"Socket '{slug}' -> relative_path='{relative_path}' -> {location}", flush=True)
            return location
    raise ValueError(f"Could not find input with slug: {slug}")


def load_json_file(*, location):
    """Reads a json file."""
    with open(location) as f:
        return json.loads(f.read())


def load_image_file(*, location):
    """
    Use SimpleITK to read an image file from the given directory.
    Searches for .mha, .tif, .tiff files (also in subdirectories).
    """
    location = Path(location)

    if not location.exists():
        raise FileNotFoundError(f"Input location does not exist: {location}")

    # Find image files
    input_files = (
        list(location.glob("**/*.mha"))
        + list(location.glob("**/*.tif"))
        + list(location.glob("**/*.tiff"))
    )

    if not input_files:
        raise FileNotFoundError(
            f"No image files (.mha, .tif, .tiff) found in {location}. "
            f"Contents: {list(location.rglob('*'))}"
        )

    print(f"Loading: {input_files[0]}", flush=True)
    result = SimpleITK.ReadImage(str(input_files[0]))
    return result


def write_image_file(*, location, image):
    """Write a SimpleITK image to the specified directory as output.mha."""
    location = Path(location)
    location.mkdir(parents=True, exist_ok=True)

    suffix = ".mha"
    output_path = location / f"output{suffix}"

    print(f"Writing output image to: {output_path}", flush=True)
    SimpleITK.WriteImage(
        image,
        str(output_path),
        useCompression=True,
    )


if __name__ == "__main__":
    # If running in exec mode (e.g. local testing with do_test_run.sh):
    #   LABEL org.grand-challenge.api-method="exec"
    #   ENTRYPOINT ["python", "inference.py"]
    from app import init_model
    raise SystemExit(run(model=init_model()))
