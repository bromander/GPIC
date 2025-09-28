import time
from PIL import Image, ImageTk, ImageOps
import numpy
from tkinter import filedialog
import tkinter as tk
from tkinter.messagebox import showerror, showwarning, showinfo
import lzma
import io
import logging
import struct
import os
from scipy.fftpack import dct, idct
from typing import Optional, List, Tuple, BinaryIO
import coloredlogs
import numpy as np
import sys
import traceback


def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

def catch_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])  # Получаем стек вызовов
            last_call = tb[-1]
            showerror(type(e).__name__, f'{type(e).__name__} on line {last_call.lineno}: {e}')
            return None


    return wrapper


class Work_gpic:

    @staticmethod
    def _idct2(block: numpy.ndarray) -> numpy.ndarray:
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def _blockify(arr: numpy.ndarray, block_size: int = 8) -> Tuple[numpy.ndarray, Tuple[int, int]]:
        '''
        divides an array into blocks
        :param arr: array of pixels
        :param block_size: size  of block
        :return: list of blocks & size of them
        '''
        h, w = arr.shape
        pad_h = (-h) % block_size
        pad_w = (-w) % block_size
        arr_padded = numpy.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant')
        H, W = arr_padded.shape
        blocks = (arr_padded
                  .reshape(H // block_size, block_size, W // block_size, block_size)
                  .swapaxes(1, 2))
        return blocks, (h, w)

    @staticmethod
    def idct2(block: np.ndarray) -> np.ndarray:
        if block.ndim != 2 or block.shape[0] != block.shape[1]:
            raise ValueError(f"idct2 waiting square block, got {block.shape}")
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def _unblockify(blocks: numpy.ndarray, orig_shape: Tuple[int, int], block_size: int = 8):
        n_v, n_h, _, _ = blocks.shape
        padded = blocks.swapaxes(1, 2).reshape(n_v * block_size, n_h * block_size)
        h, w = orig_shape
        return padded[:h, :w]

    @catch_errors
    def open_image(self, file_obj: BinaryIO) -> Image.Image:
        '''
        turns the gpic format into a regular image
        :param file_obj: buffered gpic image data
        :return: pillow Image.image object
        '''
        global gui
        data = file_obj.read()
        offset = 5  # skip the first 5 bytes
        width = height = None
        pixels = None

        logger.debug("reading image...")

        while offset < len(data):
            chunk_type = data[offset:offset + 1]
            offset += 1

            if chunk_type == b'O':
                # читаем два uint32 BE
                if offset + 8 > len(data):
                    raise ValueError("Not enough data for the image size")
                width, height = struct.unpack('>II', data[offset:offset + 8])
                offset += 8

            elif chunk_type == b'A':
                if width is None or height is None:
                    raise ValueError("The image dimensions are not set before the pixel data")
                if offset + 4 > len(data):
                    raise ValueError("Insufficient data for the length of the compressed block")
                (comp_len,) = struct.unpack('>I', data[offset:offset + 4])
                offset += 4
                if offset + comp_len > len(data):
                    raise ValueError("Not enough data for a compressed block")
                comp_data = data[offset:offset + comp_len]
                offset += comp_len

                raw = lzma.decompress(comp_data)

                # интерпретируем как int32, а не uint8
                inv = numpy.frombuffer(raw, dtype=numpy.float16).reshape((height, width))

                # разбиваем на блоки и применяем IDCT
                blocks2, _ = self._blockify(inv)
                for i in range(blocks2.shape[0]):
                    for j in range(blocks2.shape[1]):
                        blocks2[i, j] = self._idct2(blocks2[i, j])

                restored = self._unblockify(blocks2, (height, width))
                restored = numpy.clip(numpy.rint(restored), 0, 255).astype(numpy.uint8)

                # теперь у тебя готовая "картинка"
                pixels = numpy.stack([restored, restored, restored], axis=-1)

            elif chunk_type == b"T":
                title = data[offset:offset + 4]
                offset += 4
                root.title(f"GPIC | {title.decode('utf-8')}")
                logger.debug(f"Image type: {title.decode('utf-8')}")

            elif chunk_type == b'E':
                break

            else:
                if offset + 4 > len(data):
                    raise ValueError(f"Uncorrect chunk {chunk_type!r}: no length")
                (length,) = struct.unpack('>I', data[offset:offset + 4])
                offset += 4
                logger.warning(f"Skipping unknown chunk {chunk_type!r} of length {length}")
                offset += length

        if pixels is None:
            raise ValueError("Pixel block not found (chunk 'A')")

        img = Image.fromarray(pixels, mode='RGB')
        return ImageOps.exif_transpose(img)

class Gui:
    def create_window(self) -> None:
        global root

        root = tk.Tk()
        root.iconbitmap(resource_path("GPIC_logo.ico"))
        root.title("loading...")
        root.geometry("1000x800")
        root.resizable(False, False)
        self.create_menu()

    @staticmethod
    def re_create_window() -> None:
        root.destroy()
        main()

    @staticmethod
    def create_image_viewer(image) -> None:
        # creating frame of preview window
        image_viewer_frame = tk.Frame(root, bg="white", bd=5, relief=tk.GROOVE)
        image_viewer_frame.pack(anchor="center", pady=100)
        image_viewer_frame.configure(width=1000, height=800)
        image_viewer_frame.pack_propagate(False)

        image.thumbnail((560, 560), Image.Resampling.LANCZOS)

        image_ = ImageTk.PhotoImage(image)

        # creating label with image
        image_label = tk.Label(image_viewer_frame, image=image_, bg="white")
        image_label.image = image_
        image_label.pack(anchor="center", ipady=200)

    def create_menu(self) -> None:
        menu = tk.Menu(root)
        root.option_add("*tearOff", tk.FALSE)
        debug_menu = tk.Menu()
        root.option_add("*tearOff", tk.FALSE)

        file_menu = tk.Menu()
        file_save_as_menu = tk.Menu()

        file_save_as_menu.add_command(label="Export as .GPIC", command=lambda: self.export_file(path))
        file_save_as_menu.add_command(label="Export as .PNG", command=lambda: self.export_file_as_png(path))

        file_menu.add_command(label="New", command=Gui.re_create_window)
        file_menu.add_cascade(label="Export", menu=file_save_as_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=sys.exit)

        debug_menu.add_command(label="show_image_now", command=self.Debug.show_image)
        debug_menu.add_command(label="show_original_image", command=self.Debug.show_original)
        debug_menu.add_command(label="export_logs", command=lambda: self.Debug.export_logs())

        menu.add_cascade(label="File", menu=file_menu)
        menu.add_cascade(label="Debug", menu=debug_menu)
        menu.configure(activeborderwidth=5)

        root.config(menu=menu)

    # exports image as .png
    def export_file_as_png(self, img_path) -> None:

        export_path = self.Get_windows.get_folder([("png", "*.png"), ("All files", "*.*")], ".png")
        if export_path == "":
            pass
        else:
            with open(img_path, "rb") as img:
                img = img.read()
                img = io.BytesIO(img)
                img.seek(0)
                img = work_with_gppic.open_image(img)
                img.save(export_path, format="PNG")

            logger.info("image has been exported as PNG!")
            showinfo("Exported!", "Image was successfully exported as PNG!")

    # exports image in .gppic format
    def export_file(self, img_path) -> None:
        export_path = self.Get_windows.get_folder([("gpic", "*.gpic"), ("All files", "*.*")], ".gppic")
        if export_path == "":
            pass
        else:
            with open(export_path, "wb") as f:
                with open(img_path, "rb") as img:
                    img = io.BytesIO(img.read())
                    f.write(img.getvalue())

            logger.info("file has been successfully exported!")
        showinfo("Exported!", "Image was successfully exported as GPIC!")

    class Get_windows:

        # creates window for choosing folder to save something
        @staticmethod
        def get_folder(filetypes, defaultextension) -> str:
            initialfile = ''.join(path.split("/")[-1:])

            export_path = filedialog.asksaveasfilename(
                defaultextension=defaultextension,
                initialdir="/",
                initialfile=''.join(initialfile.split(".")[:-1]),
                filetypes=filetypes,
                title="Save file as"
            )
            return export_path

        # creates window for choosing path and returns it
        @staticmethod
        def get_path(filetypes) -> str:
            path = filedialog.askopenfilename(
                title="Choose file",
                filetypes=filetypes
            )
            return path

    class Debug:

        def __init__(self, gui_instance):
            self.Gui = gui_instance

        @staticmethod
        def show_image():
            with open(path, "rb") as file:
                file = file.read()
                file_image = io.BytesIO(file)
                file_image.seek(0)
                img = work_with_gppic.open_image(file_image)
                img.show()

        @staticmethod
        def show_original():
            os.startfile(path)

        @staticmethod
        def export_logs():
            def export_logger():
                buffer_handler.flush()
                log_buffer.seek(0)
                return log_buffer.read()

            now_time = time.ctime(time.time()).replace(" ", "_").replace(":", ".")
            with open(f"./{now_time}.log", "w", encoding="UTF-8") as log:
                log.write(export_logger())

            showinfo("GPIC", f"logs were successfully exported at path: {os.path.abspath(f"{now_time}.log")}")

def main():
    global buffer_handler, log_buffer
    global work_with_gppic
    global path
    gui = Gui()
    gpic = Work_gpic()
    work_with_gppic = gpic

    def create_logger():
        coloredlogs.DEFAULT_FIELD_STYLES = {
            'asctime': {'color': 'green'},
            'levelname': {'color': 'green'},
            'name': {'color': 'blue'}
        }

        coloredlogs.DEFAULT_LEVEL_STYLES = {
            'critical': {'bold': True, 'color': 'red'},
            'debug': {'color': 'white'},
            'error': {'color': 'red'},
            'info': {'color': 'white'},
            'notice': {'color': 'magenta'},
            'spam': {'color': 'green', 'faint': True},
            'success': {'bold': True, 'color': 'green'},
            'verbose': {'color': 'blue'},
            'warning': {'color': 'yellow'}
        }

        coloredlogs.install(
            level=logging.DEBUG,
            logger=logger,
            fmt='%(asctime)s : %(levelname)s : %(message)s'
        )

        log_buffer = io.StringIO()
        buffer_handler = logging.StreamHandler(log_buffer)
        buffer_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        buffer_handler.setFormatter(formatter)
        logger.addHandler(buffer_handler)

        return buffer_handler, log_buffer

    buffer_handler, log_buffer = create_logger()

    gui.create_window()
    path = gui.Get_windows.get_path([("GPIC", "*.GPIC"), ("Txt files", "*.txt"), ("All files", "*.*")])
    with open(path, "rb") as img:
        file = gpic.open_image(img)
    gui.create_image_viewer(file)

    root.mainloop()




if __name__ == "__main__":
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    main()