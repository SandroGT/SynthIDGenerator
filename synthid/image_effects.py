import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import sys

from augraphy import *

from synthid.image_processing import RGBA_TRANSPARENT


def apply_image_effect(data_image: Image, effect_level: tuple[str, str], effect_name: str, **effect_params) -> Image:
    assert effect_level[0] in {'individual', 'layer'} and effect_level[1] in {'base', 'text', 'pictures', 'composition'}

    search_key = effect_name.replace(' ', '_').replace('-', '_').lower()
    effect_fun_name = f'apply_{effect_level[0]}_{effect_level[1]}_{search_key}_effect'
    effect_fun = getattr(sys.modules[__name__], effect_fun_name, None)
    if effect_fun is None:
        raise ValueError(f'The effect "{effect_name}" has no registered effect function.')
    else:
        return effect_fun(data_image, **effect_params)


def get_np_array_image(image: Image) -> np.array:
    return np.array(image)


def get_pil_image(image: np.array) -> Image:
    return Image.fromarray(image)


# --- Individual pictures effects --------------------------------------------------------------------------------------

def apply_individual_pictures_digital_id_effect(picture_image: Image) -> Image:
    from rembg import remove as remove_background

    picture_image = picture_image.convert('L')  # 8-bit pixels -> Grayscale
    return remove_background(picture_image)


def apply_individual_pictures_drv_license_effect(picture_image: Image) -> Image:
    picture_image = picture_image.convert('L')  # Multi-channel -> Grayscale
    picture_image.putalpha(200)

    background_im: Image = Image.new('LA', picture_image.size, RGBA_TRANSPARENT[:2])

    mask_im = Image.new('L', picture_image.size, 0)
    draw = ImageDraw.Draw(mask_im)
    offset = (10, 5, 50, 5)
    rectangle_size = (offset[0], offset[1],
                      picture_image.size[0] - offset[0] - offset[2], picture_image.size[1] - offset[1] - offset[3])
    draw.rectangle(rectangle_size, fill=255)
    mask_blur_size = int(min(picture_image.size) * 0.125)
    mask_im_blur = mask_im.filter(ImageFilter.GaussianBlur(mask_blur_size))

    background_im.paste(picture_image, (0, 0), mask_im_blur)
    return background_im.convert('RGBA')


# --- Individual text effects ------------------------------------------------------------------------------------------

def apply_individual_text_degradation_effect(text_image: Image, font_size: int, text_size: tuple[int, int]) -> Image:
    import math

    text_area = text_size[0] * min(1.0 * text_size[1], 2.5 * font_size)
    augraphy_effects = [
        LinesDegradation(
            line_roi=(0.0, 0.0, 1.0, 1.0),
            line_gradient_range=(0, 255),
            line_gradient_direction=(2, 2),
            line_split_probability=(0.01, 0.05),
            line_replacement_value=(0, 255),
            line_min_length=(0, 2 * font_size),
            line_long_to_short_ratio=(0, 0),
            line_replacement_probability=(0.85, 0.90),
            line_replacement_thickness=(1, max(1, int(round(font_size / 20)))),
            p=0.50  # Probability of effect application
        ),
        Folding(
            fold_count=random.randint(max(1, int(round(text_size[0] * 0.01))), max(3, int(round(text_size[0] * 0.04)))),
            fold_noise=0.1,
            fold_angle_range=(-1, 1),
            gradient_width=(0.03, 0.08),
            gradient_height=(0.01, 0.03),
            backdrop_color=RGBA_TRANSPARENT,
            p=0.50  # Probability of effect application
        ),
        InkShifter(
            text_shift_scale_range=(int(round(font_size / 15)) * 2, int(round(font_size / 15)) * 4),
            text_shift_factor_range=(0, 1),
            text_fade_range=(int(round(font_size / 15)) * 1, int(round(font_size / 15)) * 2),
            noise_type="random",
            p=0.50  # Probability of effect application
        ),
        Letterpress(
            n_samples=(max(1, int(round(math.pow(text_area, 2/3) * 0.072))),
                       max(2, int(round(math.pow(text_area, 2/3) * 0.113)))),
            n_clusters=(max(1, int(round(math.pow(text_area, 2/3) * 0.109))),
                        max(2, int(round(math.pow(text_area, 2/3) * 0.253)))),
            std_range=(int(round(math.pow(text_area, 3/4) * 0.089)),
                       int(round(math.pow(text_area, 3/4) * 0.167))),
            value_range=(200, 255),
            value_threshold_range=(128, 128),
            blur=0,
            p=0.50  # Probability of effect application
        ),
        ColorShift(
            color_shift_offset_x_range=(int(font_size * 0.2), int(font_size * 0.4)),
            color_shift_offset_y_range=(int(font_size * 0.2), int(font_size * 0.4)),
            color_shift_iterations=(2, 3),
            color_shift_brightness_range=(int(round(font_size / 20)) * 0.8, int(round(font_size / 20)) * 1.2),
            color_shift_gaussian_kernel_range=(3, 3),
            p=0.50  # Probability of effect application
        ),
    ]

    text_np_image = get_np_array_image(text_image)

    for effect in augraphy_effects:
        mod_text_np_image = effect(text_np_image)  # Returns None if effect has not been applied (p=<probability>)
        if mod_text_np_image is not None:
            text_np_image = mod_text_np_image

    return get_pil_image(text_np_image)


# --- Layer effects -----------------------------------------------------------------------------------------------

def apply_layer_composition_degradation_effect(composed_image: Image) -> Image:
    augraphy_effects = [
        SubtleNoise(
            subtle_range=random.randint(5, 50),
            p=0.50  # Probability of effect application
        ),
        LowLightNoise(
            num_photons_range=(20, 60),
            alpha_range=(0.8, 0.9),
            beta_range=(5, 10),
            gamma_range=(0.9, 1.1),
            p=0.50,  # Probability of effect application
        )
    ]

    composed_np_image = get_np_array_image(composed_image)

    for effect in augraphy_effects:
        mod_text_np_image = effect(composed_np_image)  # Returns None if effect has not been applied (p=<probability>)
        if mod_text_np_image is not None:
            composed_np_image = mod_text_np_image

    return get_pil_image(composed_np_image)


def apply_layer_composition_resize_effect(
        composed_image: Image, min_f: float = 0.20, max_f: float = 1.00, min_w: int = 350, min_h: int = 350
) -> Image:
    resize_probability = 1
    if random.uniform(0, 1) <= resize_probability:
        min_w_resize = min_w / composed_image.size[0]
        min_h_resize = min_h / composed_image.size[1]
        min_f = max(min_f, min_w_resize, min_h_resize)
        if max_f < min_f:
            max_f = min_f
        random_resize_factor = random.uniform(min_f, max_f)
        (w, h,) = [int(round(dim * random_resize_factor)) for dim in composed_image.size]
        composed_image = composed_image.resize((w, h,), Image.BICUBIC)

    return composed_image
