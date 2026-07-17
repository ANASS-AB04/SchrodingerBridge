from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from phdtruel import require_config_path
from phdtruel.fields import FieldOfInterest, Gaussian
from phdtruel.fields.meshes import RegularGrid

BASE_DATA_PATH = require_config_path("data_dir")


def _png_to_numpy_matrix(image_path: str | Path) -> np.ndarray:
    """
    Convert a PNG grayscale image to a numpy matrix

    Args:
        image_path (str): Path to the PNG image file

    Returns:
        numpy.ndarray: 2D numpy array representing the grayscale image
    """
    try:
        # Open the image
        img = Image.open(image_path)

        # Convert to grayscale if it's not already
        if img.mode != "L":
            img = img.convert("L")

        # Convert to numpy array
        img_array = (
            np.array(img)[::-1, :]
            .reshape(-1, 1, order="F")
            .reshape(img.size[1], img.size[0])
        )

        # Invert colors
        img_array = img_array.max() - img_array

        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        raise e


def load_image_as_field(image_path: str | Path) -> FieldOfInterest:
    matrix = _png_to_numpy_matrix(image_path)
    return FieldOfInterest(
        None,
        RegularGrid(
            [np.linspace(0, 1, matrix.shape[0]), np.linspace(0, 1, matrix.shape[1])]
        ),
        matrix.flatten(),
        fill_value=0.0,
    )


def load_pre_generated_image(
    image_name: Literal["A", "a", "a_rotated", "M", "W"],
) -> FieldOfInterest:
    img_path = BASE_DATA_PATH / f"{image_name}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Image {img_path} does not exist.")
    return load_image_as_field(img_path)


def save_field_as_png(
    field: FieldOfInterest,
    image_path: str | Path,
):
    """Save a 2D `FieldOfInterest` defined on a `RegularGrid` to a grayscale PNG.

    The field values are rescaled to the 0-255 uint8 range using min/max
    normalization. NaN values are replaced by `fill_value` if provided on the
    field, otherwise with 0.

    Args:
        field: FieldOfInterest defined on a `RegularGrid` mesh.
        image_path: Path where the PNG will be written.
    """
    # Resolve path and parent directory
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # Only support regular grids for straightforward 2D saving
    mesh = field.mesh
    from phdtruel.fields.meshes import RegularGrid

    if not isinstance(mesh, RegularGrid):
        raise TypeError("save_field_as_png currently supports only RegularGrid meshes")

    # Get values as 1D array then reshape to mesh shape
    vals = np.asarray(field.values).flatten()
    try:
        matrix = vals.reshape(mesh.shape)
    except Exception as e:
        raise ValueError(
            f"Cannot reshape field values of length {vals.size} to mesh.shape {mesh.shape}: {e}"
        )

    # Replace NaNs
    if np.isnan(matrix).any():
        fill_value = getattr(field, "_fill_value", None)
        fill = float(fill_value) if fill_value is not None else 0.0
        matrix = np.where(np.isnan(matrix), fill, matrix)

    # Clip to 0..255 and convert to uint8
    img_arr = np.clip(matrix, 0, 255).astype(np.uint8)

    # Transpose to correct 90° rotation from reading function's column-major reshape
    img_arr = img_arr.T

    # Invert colors so that white/bright values become dark and vice-versa
    img_arr = 255 - img_arr

    # Create PIL image and save (mode 'L' = (8-bit pixels, black and white))
    # Flip vertically so that coordinate origin matches typical image origin
    img = Image.fromarray(img_arr[::-1, :], mode="L")
    img.save(image_path)


def translate_scale_weight_field(
    field: FieldOfInterest,
    center: tuple[float, float] | np.ndarray,
    translation: tuple[float, float] = (1.0, 1.0),
    scale: float = 1.0,
    weight: float = 1.0,
    rotation_deg: float = 0.0,
    interpolation_parameter: float = 1.0,
) -> FieldOfInterest:
    center_array = np.asarray(center, dtype=float).reshape(-1, 2)
    translation_array = np.asarray(translation, dtype=float)
    rotation_rad = np.deg2rad(rotation_deg)

    translation_s = interpolation_parameter * translation_array
    rotation_s = interpolation_parameter * rotation_rad
    scale_s = (1 - interpolation_parameter) * 1 + interpolation_parameter * scale
    weight_s = (1 - interpolation_parameter) * 1 + interpolation_parameter * weight

    points = field.mesh.points
    rotation_matrix = np.array(
        [
            [np.cos(rotation_s), -np.sin(rotation_s)],
            [np.sin(rotation_s), np.cos(rotation_s)],
        ]
    )

    new_points = (
        (points - center_array - translation_s) @ rotation_matrix
    ) / scale_s + center_array

    new_field = field.eval(new_points, as_foi=True) * weight_s
    return new_field


def load_image_and_get_transformation(
    image_path: str | Path,
    translation: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    weight: float = 1.0,
    rotation_deg: float = 0.0,
    interpolation_parameter: float = 0.0,
) -> FieldOfInterest:
    field = load_image_as_field(image_path)

    g = Gaussian.from_field(field.mesh.points, field.values)
    center = g.mu

    transformed_field = translate_scale_weight_field(
        field,
        center,
        translation=translation,
        scale=scale,
        weight=weight,
        rotation_deg=rotation_deg,
        interpolation_parameter=interpolation_parameter,
    )
    return transformed_field


def main_generate_translated_scaled(
    source_image: str | Path,
    translation: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    weight: float = 1.0,
    rotation_deg: float = 0.0,
):
    A_path = str(BASE_DATA_PATH / "A.png")
    where_to_save_path = Path(source_image).parent

    image_name = Path(source_image).stem
    param_str = f"t{translation[0]:.2f}-{translation[1]:.2f}_s{scale:.2f}_f{weight:.2f}_r{rotation_deg:.1f}"

    gen_ref_path = where_to_save_path / f"{image_name}_{param_str}_ref.png"
    zero_path = where_to_save_path / f"{image_name}_{param_str}_0.png"
    one_path = where_to_save_path / f"{image_name}_{param_str}_1.png"
    superposed_transformations = (
        where_to_save_path / f"{image_name}_{param_str}_0ref1.png"
    )

    field = load_image_as_field(A_path)
    g = Gaussian.from_field(field.mesh.points, field.values)

    center = g.mu

    zero_field = translate_scale_weight_field(
        field,
        center,
        translation=translation,
        scale=scale,
        weight=weight,
        rotation_deg=rotation_deg,
        interpolation_parameter=0.0,
    )
    ref_field = translate_scale_weight_field(
        field,
        center,
        translation=translation,
        scale=scale,
        weight=weight,
        rotation_deg=rotation_deg,
        interpolation_parameter=0.5,
    )
    one_field = translate_scale_weight_field(
        field,
        center,
        translation=translation,
        scale=scale,
        weight=weight,
        rotation_deg=rotation_deg,
        interpolation_parameter=1.0,
    )

    save_field_as_png(one_field, one_path)
    save_field_as_png(zero_field, zero_path)
    save_field_as_png(ref_field, gen_ref_path)

    save_field_as_png(zero_field + ref_field + one_field, superposed_transformations)


if __name__ == "__main__":
    _a_png = BASE_DATA_PATH / "A.png"
    main_generate_translated_scaled(
        source_image=_a_png,
        translation=(0.3, -0.3),
        scale=0.7,
        weight=0.7,
    )

    main_generate_translated_scaled(
        source_image=_a_png,
        translation=(0.3, -0.3),
        scale=0.7,
        weight=0.7,
        rotation_deg=45,
    )
