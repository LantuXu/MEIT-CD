import os


def scan_images_and_save(folder_path, output_txt):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    with open(output_txt, 'w', encoding='utf-8') as f:
        for img in image_files:
            f.write(img + '\n')

    print(f"A total of  {len(image_files)} images were found, saved to {output_txt}")


# 示例使用
folder_path = "your image folder path"  # Modify to your image folder path
output_txt = "image_list.txt"  # The output txt file name
scan_images_and_save(folder_path, output_txt)
