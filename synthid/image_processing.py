from PIL import Image

RGBA_TRANSPARENT = (0, 0, 0, 0,)


def extend(img: Image, size: tuple[int, int], translation_ref: str, translation: tuple[int, int] = (0, 0,), rotation: float = 0,
           fill_color: int | tuple[int, ...] = RGBA_TRANSPARENT) -> Image:
    def get_pos(_img, _size, _translation, i, ref: str):
        if ref == 'center':
            return (_size[i] - _img.size[i]) // 2 + _translation[i]
        elif ref == 'corner':
            return _translation[i]
        else:
            raise ValueError(f'Value "{ref}" is not a valid value for `ref`.')

    assert len(size) == 2 and len(translation) == 2 and all([img.size[i] <= size[i] for i in range(2)])
    fill_image = Image.new('RGBA', size, fill_color)
    img_position = (
        get_pos(img, size, translation, 0, translation_ref),
        get_pos(img, size, translation, 1, translation_ref),
    )
    paste_image = img.copy().rotate(rotation, expand=True, fillcolor=RGBA_TRANSPARENT)
    fill_image.paste(paste_image, img_position)
    return fill_image


def overlap(back_img: Image, front_img: Image, translation: tuple[int, int] = (0, 0,), rotation: float = 0):
    assert all([fs <= bs for fs, bs in zip(front_img.size, back_img.size)]), list(zip(front_img.size, back_img.size))
    assert len(translation) == 2

    # Modify front image
    new_front_img = front_img.rotate(rotation, expand=True, fillcolor=RGBA_TRANSPARENT)
    assert all([fs <= bs for fs, bs in zip(front_img.size, back_img.size)])
    new_front_img = extend(new_front_img, back_img.size, 'corner', translation, rotation, RGBA_TRANSPARENT)
    assert new_front_img.size == back_img.size

    overlap_img = back_img.copy()
    # TODO add some kind of shadow of the document on the background to make it look less fake
    overlap_img.paste(new_front_img, box=None, mask=new_front_img)

    return overlap_img


def cropped_resize(image: Image, size: tuple[int | None, int | None]) -> Image:
    assert len(size) == 2 and any([isinstance(s, int) for s in size])
    if None in size:
        rf = [new_s / old_s for old_s, new_s in zip(image.size, size) if new_s is not None][0]
        output_size = tuple([round(rf * s) for s in image.size])
        assert any([s1 == s2 for s1, s2 in zip(output_size, size)])
        return image.resize(output_size)
    else:
        resize_factors = [new_s / old_s for old_s, new_s in zip(image.size, size)]
        dim, rf = max(enumerate(resize_factors), key=lambda x: x[1])
        dim_ortho = (dim + 1) % 2
        tmp_size = [round(rf * s) for s in image.size]
        assert any([s1 == s2 for s1, s2 in zip(tmp_size, size)])
        assert image.size[dim_ortho] >= size[dim_ortho]
        tmp_image = image.resize(tmp_size)
        crop_size = [0, 0, tmp_image.size[0], tmp_image.size[1]]
        cut_size = tmp_image.size[dim_ortho] - size[dim_ortho]
        crop_size[dim_ortho] = round(cut_size / 2)
        crop_size[dim_ortho+2] = tmp_image.size[dim_ortho] - (cut_size - round(cut_size / 2))
        return tmp_image.crop(crop_size)


def darker_image_overlay(img_background: Image, img_foreground: Image) -> Image:
    """Overlap two pictures, retaining the darker parts from each."""
    import numpy as np

    def compute_channel_dark_score(channel):
        # alpha scaling * closeness to zero score
        return (channel[-1] / 255) * (765 - channel[:-1].sum())

    arr_b, arr_f = np.array(img_background), np.array(img_foreground)
    arr_b_score = np.apply_along_axis(compute_channel_dark_score, -1, arr_b)
    arr_f_score = np.apply_along_axis(compute_channel_dark_score, -1, arr_f)
    mask = arr_b_score >= arr_f_score

    return Image.fromarray(np.where(mask[..., np.newaxis], arr_b, arr_f))
