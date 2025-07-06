import struct
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
import numpy as np
from progress.bar import IncrementalBar
from tkinter import filedialog
import zlib


def get_image(path):
    with open(path, "rb") as img:
        img = img.read()




    index = 5
    description = "unknown"


    while True:
        now_elem = img[index:index+1]

        if now_elem == b"O":
            size = [struct.unpack('>I', img[index+1:index+5])[0], struct.unpack('>I', img[index+5:index+9])[0]]
            print(f"size: {size}")
            index += 8

        if now_elem == b"A":
            bytes_len = struct.unpack('>I', img[index+1:index+5])[0]
            index += 5
            decompressed = zlib.decompress(img[index:index+bytes_len])
            img = img.replace(img[index:index+bytes_len], decompressed)

            pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)


            data = img[index + 1:index + 1 + size[0] * size[1]]

            pixels[:, :, 0] = np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0])

            pixels[:, :, 1] = pixels[:, :, 0]
            pixels[:, :, 2] = pixels[:, :, 0]

            index += size[0] * size[1]

        if now_elem == b"T":
            if img[index + 1:index + 4] == b"IMG":
                description = "Image"
            elif img[index + 1:index + 4] == b"QRC":
                description = "QR-code"

        if now_elem == b"E":
            break

        index += 1


    original_image = Image.fromarray(pixels)
    original_image = ImageOps.exif_transpose(original_image)




    window_width = 800
    window_height = 600

    max_size = (window_width, window_height)


    image = original_image.copy()
    image.thumbnail(max_size, Image.Resampling.NEAREST)


    photo = ImageTk.PhotoImage(image)
    img_label.config(image=photo)
    img_label.image = photo



file_path = filedialog.askopenfilename(
    title="Выберите файл",
    filetypes=[("GPPIC файл", "*.gppic"), ("Все файлы", "*.*"), ("Текстовый файл", "*.txt")]
)



root = tk.Tk()
root.title("Image Viewer")
root.geometry("800x600")

img_label = tk.Label(root)
img_label.pack(expand=True)

get_image(file_path)

root.mainloop()