from PIL import Image
import glob

# 自动列出所有符合条件的 PNG 文件
image_paths = glob.glob('/Users/liyushuo/Desktop/image3/plot_2024-11-13 15-44-52_*.png')
image_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按文件名中的数字部分排序

# Open images
images = [Image.open(img_path) for img_path in image_paths]

# Save as GIF with 0.1s duration
output_path = "/Users/liyushuo/Desktop/animation_part1.gif"
images[0].save(output_path, save_all=True, append_images=images[1:], duration=100, loop=0)

