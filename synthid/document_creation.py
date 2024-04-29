from __future__ import annotations

import json
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageFont

from tqdm import tqdm
from wand.color import Color as WandColor
from wand.drawing import Drawing as WandDrawing
from wand.image import Image as WandImage

from logger import LOGGER
from synthid.image_processing import RGBA_TRANSPARENT, overlap, cropped_resize, extend, darker_image_overlay
from synthid.image_effects import apply_image_effect
from synthid.personal_document_data import load_template_image, retrieve_random_data
from synthid.settings import (SIZE_DOCUMENT_SAMPLES, OUTPUT_EXTENSION, TEMPLATES_DOCUMENTS_PATH, TEMPLATES_FONTS_PATH,
                              CREATED_DOCS_PATH, TEMPLATES_DOCUMENT_DATA_FILE_NAME)


def create_document_samples():
    LOGGER.info(f'Creating {SIZE_DOCUMENT_SAMPLES} samples for each personal document type:')
    output_extension = '.' + OUTPUT_EXTENSION.lower()

    for doc_type_path in [f for f in TEMPLATES_DOCUMENTS_PATH.iterdir() if (not f.stem.startswith('.')) and f.is_dir()]:
        # For each document type, create the output folder and generate #SIZE_DOCUMENT_SAMPLES random samples
        doc_type = doc_type_path.stem
        LOGGER.info(f' - {doc_type}')

        doc_type_output_folder = CREATED_DOCS_PATH.joinpath(doc_type)
        doc_type_output_folder.mkdir()

        doc_compilation_data_path = doc_type_path.joinpath(TEMPLATES_DOCUMENT_DATA_FILE_NAME)
        with doc_compilation_data_path.open('rb') as f:
            doc_compilation_data = json.load(f)

        for i in tqdm(list(range(SIZE_DOCUMENT_SAMPLES))):
            folder_path = doc_type_output_folder.joinpath(str(i))
            folder_path.mkdir()
            document_samples, document_labels = create_random_document_sample(doc_type_path, doc_compilation_data)
            for doc_side in document_samples.keys():
                document_path = folder_path.joinpath(get_output_name(i, doc_side, output_extension))
                document_samples[doc_side].save(document_path, OUTPUT_EXTENSION)
                with open(str(document_path).replace(output_extension, '.json'), 'w') as f:
                    json.dump(document_labels[doc_side], f)


def get_output_name(i: int, doc_side: str, extension: str) -> str:
    assert 0 <= i < SIZE_DOCUMENT_SAMPLES and extension.startswith('.')
    # Could not use f-strings such as f'{i:0N}' since N is not fixed, and you cannot use variables as format specifiers
    max_size = SIZE_DOCUMENT_SAMPLES-1 if SIZE_DOCUMENT_SAMPLES % 10 == 0 else SIZE_DOCUMENT_SAMPLES
    max_i_len = len(str(max_size))
    actual_i_len = len(str(i))
    prog_number = '0' * (max_i_len - actual_i_len) + str(i)
    return f'{prog_number}_{doc_side}{extension}'


def create_random_document_sample(
        doc_type_path: Path,
        doc_compilation_data: dict
) -> tuple[dict[str, Image], dict[str, dict]]:
    template_image_front, template_image_back = load_template_image(doc_type_path)

    final_doc_images = dict()
    doc_labels = dict()
    if template_image_front is not None:
        final_doc_images['front'] = {
            'base': template_image_front,
            'text': Image.new('RGBA', template_image_front.size, color=RGBA_TRANSPARENT),
            'pictures': Image.new('RGBA', template_image_front.size, color=RGBA_TRANSPARENT)
        }
        doc_labels['front'] = dict()
    if template_image_back is not None:
        final_doc_images['back'] = {
            'base': template_image_back,
            'text': Image.new('RGBA', template_image_back.size, color=RGBA_TRANSPARENT),
            'pictures': Image.new('RGBA', template_image_back.size, color=RGBA_TRANSPARENT)
        }
        doc_labels['back'] = dict()

    doc_type = doc_type_path.stem

    global_position_offset = local_position_range = None  # Default value never used, but the IDE doesn't get it
    if not isinstance(doc_compilation_data['generic'].get('global_position_range', None), dict):
        global_position_range = get_position_range(doc_compilation_data['generic'].get('global_position_range', None))
        global_position_offset = get_position_offset(global_position_range)
    if not isinstance(doc_compilation_data['generic'].get('local_position_range', None), dict):
        local_position_range = get_position_range(doc_compilation_data['generic'].get('local_position_range', None))

    for doc_side in final_doc_images.keys():
        if isinstance(doc_compilation_data['generic'].get('global_position_range', None), dict):
            global_position_range = get_position_range(
                doc_compilation_data['generic']['global_position_range'].get(doc_side, None)
            )
            global_position_offset = get_position_offset(global_position_range)
        if isinstance(doc_compilation_data['generic'].get('local_position_range', None), dict):
            local_position_range = get_position_range(
                doc_compilation_data['generic']['local_position_range'].get(doc_side, None)
            )

        for info_type in doc_compilation_data[doc_side].keys():
            info_data = retrieve_random_data(info_type)

            info_compilation_data = dict()
            for k, v in doc_compilation_data['generic'].items():
                if isinstance(v, dict) and doc_side in v:
                    info_compilation_data[k] = v[doc_side]
                else:
                    info_compilation_data[k] = v
            for k, v in doc_compilation_data[doc_side][info_type].items():
                # Specific field info can override generic
                info_compilation_data[k] = v

            if 'global_position_range' in doc_compilation_data[doc_side][info_type] and \
                    doc_compilation_data[doc_side][info_type]['global_position_range'] is None:
                glob_pos_offset = get_position_offset(None)
            else:
                glob_pos_offset = global_position_offset

            if 'local_position_range' in doc_compilation_data[doc_side][info_type] and \
                    doc_compilation_data[doc_side][info_type]['local_position_range'] is None:
                loc_pos_range = None
            else:
                loc_pos_range = local_position_range
            loc_pos_offset = get_position_offset(loc_pos_range)

            info_compilation_data['position'] = adjust_position(
                info_compilation_data['position'], glob_pos_offset, loc_pos_offset
            )

            info_data = post_process_info_data(doc_type, doc_side, info_type, info_data, **info_compilation_data)
            info_category = 'pictures' if info_type in {'face_picture'} else 'text'
            image_overlap_function = add_document_picture if info_category == 'pictures' else add_document_text
            final_doc_images[doc_side][info_category] = image_overlap_function(
                final_doc_images[doc_side][info_category],
                info_data,
                **info_compilation_data,
            )
            update_data_labels(doc_labels, doc_type, doc_side, info_type, info_data)

        # Apply individual layer effects
        for layer_name in ['text', 'pictures']:
            final_doc_images[doc_side][layer_name] = add_layer_effects(
                final_doc_images[doc_side][layer_name], layer_name, doc_compilation_data
            )

        # Overlap 'text' and 'picture' layers over the 'base' image
        final_doc_images[doc_side] = overlap(
            overlap(
                final_doc_images[doc_side]['base'], final_doc_images[doc_side]['text']
            ),
            final_doc_images[doc_side]['pictures']
        )

        # Apply composition layer effects
        layer_name = 'composition'
        final_doc_images[doc_side] = add_layer_effects(final_doc_images[doc_side], layer_name, doc_compilation_data)

    return final_doc_images, doc_labels


def get_position_range(raw_position_range: int | list[int | list]) -> dict | None:
    def get_pos_range_dict(pos_range: int | list) -> dict:
        if isinstance(pos_range, int):
            # A single value has been specified, so the range goes from lower bound -|value| to upper bound +|value|
            pos_range = abs(pos_range)
            return {'min': -pos_range, 'max': pos_range}
        elif isinstance(pos_range, list) and len(pos_range) == 2 and all([isinstance(v, int) for v in pos_range]):
            # Two values have been specified, so the range goes from lower bound fist value to upper bound second value
            if pos_range[0] > pos_range[1]:
                raise ValueError(f'`pos_range` is not a valid range as the minimum is greater than the maximum.')
            return {'min': pos_range[0], 'max': pos_range[1]}
        else:
            raise ValueError(f'`pos_range` should be a single integer or a list of two integers.')

    if raw_position_range is None:
        # No range specified
        return None
    elif isinstance(raw_position_range, int):
        # A single value has been specified, so the range is the same for both x and y coordinates
        return {'x': get_pos_range_dict(raw_position_range), 'y': get_pos_range_dict(raw_position_range)}
    elif isinstance(raw_position_range, list) and len(raw_position_range) == 2:
        # Two values have been specified, so different values for x and y
        return {'x': get_pos_range_dict(raw_position_range[0]), 'y': get_pos_range_dict(raw_position_range[1])}
    else:
        raise ValueError(f'`raw_position_range` should be a single integer or a list of two elements.')


def get_position_offset(position_range: dict | None) -> list[int]:
    if position_range is None:
        return [0, 0]
    else:
        return [
            random.randint(position_range['x']['min'], position_range['x']['max']),
            random.randint(position_range['y']['min'], position_range['y']['max'])
        ]


def adjust_position(default_pos: list[int | list], global_offset: list[int], local_offset: list[int]) -> list[int | list]:
    # `default_pos` should be [x0, y0] or [[x0, y0], [x1, y1], ...]
    if isinstance(default_pos[0], int):
        return [sum(values) for values in zip(default_pos, global_offset, local_offset)]
    else:
        return [[sum(values) for values in zip(def_pos_i, global_offset, local_offset)] for def_pos_i in default_pos]


def update_data_labels(
        doc_labels: dict,
        doc_type: str,
        doc_side: str,
        info_type: str,
        info_data: str | Image
) -> None:
    # Don't store face pictures data!
    if info_type != 'face_picture':
        doc_labels[doc_side][info_type] = info_data


def post_process_info_data(
        doc_type: str,
        doc_side: str,
        info_type: str,
        info_data: str | Image,
        text_uppercase: bool,
        **_
) -> str | Image:
    if doc_type == 'digital-id':
        # Post-process date format
        if info_type == 'birthday' or 'date' in info_type:
            info_data = info_data.replace('/', '.')
    if isinstance(info_data, str) and text_uppercase:
        info_data = info_data.upper()

    return info_data


def add_layer_effects(layer_image: Image, layer_name: str, doc_compilation_data: dict) -> Image:
    if doc_compilation_data['layer_effects'].get(layer_name, None) is not None:
        for effect_info in doc_compilation_data['layer_effects'][layer_name]:
            if isinstance(effect_info, str):
                # Just the name of the effects, because it has no parameters
                effect_name = effect_info
                effect_params = dict()
            elif isinstance(effect_info, dict):
                # Effect with dictionary of custom parameters
                assert len(effect_info) == 1
                effect_name = list(effect_info.keys())[0]
                effect_params = effect_info[effect_name]
            else:
                raise ValueError(f'Layer effects must be strings or dictionaries.')
            layer_image = apply_image_effect(layer_image, ('layer', layer_name), effect_name, **effect_params)
    return layer_image


def add_document_picture(
        base_image: Image,
        picture_data: Image,
        position: list[int, int],
        size: list[int | None],
        picture_effects: list[str] = None,
        **_
) -> Image:
    size = (size[0], size[1],)
    picture_data = cropped_resize(picture_data, size)
    # TODO could add support to multiple positions for pictures too
    assert all([isinstance(v, int) for v in position])
    position = (position[0], position[1],)

    # Apply local picture effects
    if picture_effects is not None:
        for effect_name in picture_effects:
            picture_data = apply_image_effect(picture_data, ('individual', 'pictures'), effect_name)

    return overlap(base_image, picture_data, tuple(position))


def add_document_text(
        base_image: Image,
        text_data: str,
        position: list[int] | list[list[int]],
        font_size: int,
        font_type: str,
        font_color: list[int],
        font_boldness: float,
        text_uppercase: bool,
        text_interline_spacing: float = 0,
        text_interword_spacing: float = 0,
        text_kerning: float = 0,
        text_position_reference: list[bool] = None,
        text_effects: list[str] = None,
        **_
) -> Image:
    if text_position_reference is None:
        text_position_reference = ['l', 't']  # Default top-left alignment

    all_text_image = Image.new('RGBA', base_image.size, color=RGBA_TRANSPARENT)

    # Generate the text only
    text_image, text_size = generate_text(
        text_data, font_size, font_type, font_color, font_boldness, text_uppercase,
        text_interline_spacing, text_interword_spacing, text_kerning, base_image.size
    )

    # Apply local text effect and paste the text in all its requested positions
    if not isinstance(position[0], list):
        position = [position]
    if not isinstance(text_position_reference[0], list):
        text_position_reference = [text_position_reference for _ in position]
    assert len(position) == len(text_position_reference)
    for pos, text_pos_ref in zip(position, text_position_reference):
        # Extend text image to have space for deformation effects
        effect_margin = [10, 10]
        effect_text_image_size = (text_image.size[0]+2*effect_margin[0], text_image.size[1]+2*effect_margin[1],)
        text_image = extend(text_image, effect_text_image_size, 'center')
        # Apply local text effects
        effect_text_image = text_image
        if text_effects is not None and len(text_effects) > 0:
            for effect_name in text_effects:
                effect_text_image = apply_image_effect(
                    text_image, ('individual', 'text',), effect_name, font_size=font_size, text_size=text_size
                )
        # Overlap with the all text layer
        pos = adjust_position_reference(pos, effect_margin, text_size, text_pos_ref)
        all_text_image = overlap(all_text_image, effect_text_image, translation=pos)

    return overlap(base_image, all_text_image)


def adjust_position_reference(
        position: list[int],
        effect_margin: list[int],
        text_size: tuple[int, int],
        text_position_reference: list[str]
) -> tuple[int, int]:
    assert all([len(ls) == 2 for ls in [position, effect_margin, text_size, text_position_reference]])
    adjusted_positions = []
    for i in range(2):
        if text_position_reference[i] == ['l', 't'][i]:
            # Left/top alignment, no shift needed on the actual position
            shift = 0
        elif text_position_reference[i] == 'c':
            # Center alignment, shift by half the text size
            shift = int(text_size[i] / 2)
        elif text_position_reference[i] == ['r', 'b'][i]:
            # Right/bottom alignment, shift by the text size
            shift = text_size[i]
        else:
            raise ValueError(f'`text_position_reference` is "{text_position_reference}" but it should be "l" (left),'
                             f' "r" (right), "t" (top), "b" (bottom) or "c" (center)')
        adjusted_positions.append(position[i] - effect_margin[i] - shift)
    return adjusted_positions[0], adjusted_positions[1]


def generate_text(
        text: str,
        font_size: int,
        font_type: str,
        font_color: list[int],
        font_boldness: float,
        text_uppercase: bool,
        text_interline_spacing: float,
        text_interword_spacing: float,
        text_kerning: float,
        ref_size: tuple
) -> tuple[Image, tuple[int, int]]:
    import io
    import math

    actual_text = text.upper() if text_uppercase else text
    del text
    margin = int(math.ceil(font_boldness) + 3)
    ref_size = [dim + margin * 2 for dim in ref_size]

    with WandImage(width=ref_size[0], height=ref_size[1], pseudo='xc:transparent') as wand_text_image:
        with WandDrawing() as wand_draw:
            wand_draw.font_size = font_size
            wand_draw.font = get_font_path(font_type)
            wand_draw.fill_color = WandColor(f'rgb{tuple(font_color)}')
            wand_draw.text_interline_spacing = text_interline_spacing
            wand_draw.text_interword_spacing = text_interword_spacing
            wand_draw.text_kerning = text_kerning
            wand_draw.text(margin, margin + font_size, actual_text)
            wand_draw(wand_text_image)
        pil_text_image = Image.open(io.BytesIO(wand_text_image.make_blob('png'))).convert('RGBA')

    pil_text_image = pil_text_image.crop(pil_text_image.getbbox())
    text_size = pil_text_image.size
    pil_text_image = extend(pil_text_image, (pil_text_image.size[0]+margin, pil_text_image.size[1]+margin,), 'center')

    pil_text_image = apply_text_boldness(pil_text_image, font_boldness)

    return pil_text_image.crop(pil_text_image.getbbox()), text_size


def get_font_path(font_type: str) -> str:

    try:
        font_path = ImageFont.truetype(font_type).path
    except OSError:
        font_path = TEMPLATES_FONTS_PATH.joinpath(font_type)
        if not font_path.exists():
            raise FileNotFoundError(f'File font for "{font_type}" not found')
        font_path = str(font_path)
    return font_path


def apply_text_boldness(text_image: Image, boldness: float):
    """Increase the text stroke width (boldness) using dilation operations.
    The `boldness` parameter indicates the number of pixels for the dilation filter. The fractional part of the
     `boldness` value is used to scale the color of the last additional dilated pixel, providing finer control over the
     stroke width at a resolution higher than one pixel.
    """
    # TODO actually works only with grayscale colors (image can be RGBA) because `.filter` only works with grayscale.
    #  It could be extended to work with colors by using `.filter` for dilation on an image representing the percentage
    #  of intensity of that color and then scaling the original multi-channel image by this single-channel intensity
    #  percentage.
    import numpy as np

    def get_filter_size(pixels):
        assert isinstance(pixels, int)
        return pixels * 2 + 1

    def scale_to_range(percentage: float, target_range: tuple[int, int], resolution: int):
        """Scale `percentage` (value in [0:1]) to a value in `target_range` at steps of `resolution`."""
        import math
        assert 0 <= percentage <= 1
        assert (target_range[1] - target_range[0]) % resolution == 0,\
            f'You can\'t go from "{target_range[0]}" to "{target_range[1]}" at steps of "{resolution}".'
        resolution_step = (target_range[1] - target_range[0]) * math.ceil(percentage * resolution) / resolution
        return int(target_range[0] + resolution_step)

    start_image = text_image

    # "Integer boldness" is utilized for a dilation of `int_boldness` pixels.
    int_boldness = int(boldness)
    if int_boldness > 0:
        int_dilated_image = start_image.filter(ImageFilter.MaxFilter(get_filter_size(int_boldness)))
    else:
        int_dilated_image = start_image

    # "Float boldness" is utilized for a dilation of `1` pixel with a scaled color.
    # This transformation is somewhat resource-intensive but yields a visually pleasing result.
    # Additionally, the transformation is consistent, as it has been designed such that:
    #   - Skipping the transformation or applying it with a very low `float_boldness` value (e.g., 0.01) shows no
    #      apparent difference.
    #   - Utilizing a very high `float_boldness` value (e.g., 0.99) is visually equivalent to using `int_boldness + 1`
    #      with null `float_boldness`.
    float_boldness = boldness - int_boldness
    if float_boldness > 0:
        # One pixel dilation
        float_dilated_image = int_dilated_image.filter(ImageFilter.MaxFilter(get_filter_size(1)))
        # Scale the colors by a `float_boldness` factor
        float_dilated_image = Image.fromarray((np.array(float_dilated_image) * float_boldness).astype(np.uint8))
        # Paste the original image over the dilated one (image_darker_overlay could be better, but takes more time)
        float_dilated_image = Image.alpha_composite(float_dilated_image, int_dilated_image)
        # Make the composition smoother
        float_dilated_image = float_dilated_image.filter(ImageFilter.SMOOTH_MORE)
        pc = scale_to_range(float_boldness, (200, 400,), 10)
        th = rd = scale_to_range(float_boldness, (1, 2,), 1)
        float_dilated_image = float_dilated_image.filter(ImageFilter.UnsharpMask(radius=rd, percent=pc, threshold=th))
        # Paste again the original image over the composition to get rid of the introduced text "blurriness"
        float_dilated_image = darker_image_overlay(float_dilated_image, int_dilated_image)  # Resource-intensive
    else:
        float_dilated_image = int_dilated_image
    return float_dilated_image


def add_random_document_noise(document_image: Image, **_):
    LOGGER.warning('Function "add_random_document_noise" is not implemented yet.')
