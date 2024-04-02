import os
from PIL import Image


def cut_images(image_path, save_path, box, out_shape):
    """
    从image_path输入图片，按照指定方法裁剪之后保存到save_path
    :param image_path: 输入图片的路径
    :param save_path: 输出图片的路径
    :param box: 输入图片的裁剪区域
    :param out_shape: 输出图片的形状
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.isdir(image_path):
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_files = [image_path]

    for image_file in image_files:
        image = Image.open(image_file)
        cropped_image = image.crop(box)
        cropped_w, cropped_h = cropped_image.size

        tile_h = cropped_h // out_shape[0]
        tile_w = cropped_w // out_shape[1]

        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
                tile = cropped_image.crop((j * tile_w, i * tile_h, (j + 1) * tile_w, (i + 1) * tile_h))
                tile.save(os.path.join(save_path, f'{os.path.basename(image_file).split(".")[0]}_{j}_{i}.png'))


if __name__ == '__main__':
    image_path = r'images'
    save_path = r'images_cut'
    box = (10, 10, 210, 230)
    out_shape = (3, 2)
    cut_images(image_path, save_path, box, out_shape)
