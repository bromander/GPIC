import time
from PIL import Image, ImageTk, ImageOps
import numpy
from tkinter import filedialog
import tkinter as tk
from tkinter.messagebox import showerror, showinfo
import lzma
import io
import logging
import struct
import os
import scipy
from scipy.fftpack import dct, idct
from typing import Optional, List, Tuple, BinaryIO
import coloredlogs
import sys
import traceback
import brotli
import toml

VERSION = "1.6.0"

# pigar generate - create requirements.txt
# pyinstaller --onefile --add-data "main/GPIC_logo.ico;." --icon=main/GPIC_logo.ico --noconsole --collect-all scipy main/conventer.py
# pyinstaller --onefile --add-data "main/GPIC_logo.ico;." --icon=main/GPIC_logo.ico --noconsole --collect-all scipy main/image_loader.py


def catch_errors(func):
    '''
    if error occurs in the code shows error window to user and logging error to console
    '''
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])
            last_call = tb[-1]
            logger.error(f'{type(e).__name__} on line {last_call.lineno}: {e}')
            showerror(type(e).__name__, f'{type(e).__name__} on line {last_call.lineno}: {e}')
            Gui.Debug.export_logs('ERROR')
            sys.exit()
    return wrapper

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)


class Work_with_gppic:

    def __init__(self, compression_dct_force: int = 10,
                 compression_quant_force: int = 1,
                 compression_type: int = 1,
                 array_data_type: int = 0,
                 dct_blocksize: int = 8,
                 brightnes: int = 0,
                 sharpness: int = 0,
                 format_version: int = 2):
        '''
        :param compression_dct_force: compression force during DCT compression
        :param compression_quant_force: compression force during quantization compression
        :param compression_type: if == 1: rounds the value, if == 2: rounds upward, if == 3: rounds down
        :param array_data_type: defines the data type of the array
        :param dct_blocksize: size of the blocks into which the image will be divided during DCT compression
        '''

        self.Work_TOML.check_file()
        toml_data = self.Work_TOML.get_toml_data()["settings"]

        self.file_path: Optional[str] = None
        self.compression_dct_force = compression_dct_force
        self.compression_quant_force = compression_quant_force
        self.compression_type = toml_data["compression_type"]  # 0=floor,1=round,2=ceil
        self.array_data_type = array_data_type
        self.dct_blocksize = dct_blocksize
        self.brightnes = brightnes
        self.kernel = numpy.array([
                        [0, -1,  0],
                        [-1,  5, -1],
                        [0, -1,  0]
                        ])
        self.sharpness = sharpness
        self.format_version = toml_data["file_version"]

        self.size_img = 0
        self.size_img_uncompress = 0
        self.size_img_uncompress_DCT = 0
        self.RMSE = 0

    class Work_DCT:
        @staticmethod
        def _dct2(block: numpy.ndarray) -> numpy.ndarray:
            '''
            applies the DCT compression method
            :param block: 1 block - array of pixels
            '''
            return dct(dct(block.T, norm='ortho').T, norm='ortho')

        @staticmethod
        def _idct2(block: numpy.ndarray) -> numpy.ndarray:
            """
            applies the IDCT compression method
            :param block: 1 block - array of pixels
            """
            return idct(idct(block.T, norm='ortho').T, norm='ortho')

        @staticmethod
        def _blockify(arr: numpy.ndarray, block_size: int = 8) -> Tuple[numpy.ndarray, Tuple[int, int]]:
            '''
            divides an array into blocks
            :param arr: array of pixels
            :param block_size: size of block
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
        def _unblockify(blocks: numpy.ndarray, orig_shape: Tuple[int, int], block_size: int = 8):
            """
            divides an array of blocks into collected array
            :param blocks: array of blocks
            :param orig_shape: original image size
            :param block_size: size of block
            :return: array of pixels
            """
            n_v, n_h, _, _ = blocks.shape
            padded = blocks.swapaxes(1, 2).reshape(n_v * block_size, n_h * block_size)
            h, w = orig_shape
            return padded[:h, :w]

    class Work_TOML:

        @staticmethod
        def check_file(filename="data.toml"):
            try:
                toml.load(filename)
            except FileNotFoundError:
                logger.debug("Creating TOML file...")
                data = {"settings": {"file_version": 2, "compression_type": 1}}
                with open(filename, "w", encoding="utf-8") as f:
                    toml.dump(data, f)

        @staticmethod
        def get_toml_data(filename="data.toml"):
            return toml.load(filename)

        @staticmethod
        def edit_toml_data(key: any, data: any, filename="data.toml"):

            file_data = toml.load(filename)
            file_data["settings"][key] = data

            with open(filename, "w", encoding="utf-8") as f:
                toml.dump(file_data, f)


    @staticmethod
    @catch_errors
    def extract_pixels(path: str) -> Optional[List[List[Tuple[int, int, int]]]]:
        '''
        returns pixels array from inage
        :param path: path to file
        :return: matrix with data of pixels
        '''
        global IMAGE_FORMAT
        if path is None:
            raise ValueError("'path' not found")
        try:
            with Image.open(path) as img:
                IMAGE_FORMAT = img.format
                if img.mode != "RGB":
                    img = img.convert("RGB")
                width, height = img.size
                logger.info(f"Picture size: {width}×{height}")

                logger.debug("Getting pixel data…")
                pixels = list(img.getdata())
            logger.debug("Loading pixel matrix…")
            pixel_matrix: List[List[Tuple[int, int, int]]] = [
                pixels[i * width:(i + 1) * width]
                for i in range(height)
            ]
            return pixel_matrix
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            return None

    @staticmethod
    def _quantize(data: numpy.ndarray, force: int, method: int) -> numpy.ndarray:
        '''
        applies the quantization method
        :param data: array of pixels
        :param force: force of compression
        :param method: if == 1: rounds the value, if == 2: rounds upward, if == 3: rounds down
        :return:
        '''

        data = data.astype(numpy.int32)
        if force != 1:
            mask = data != 0
            if method == 1:
                data[mask] = numpy.round(data[mask] / force) * force
            elif method == 2:
                data[mask] = numpy.ceil(data[mask] / force) * force
            elif method == 0:
                data[mask] = numpy.floor(data[mask] / force) * force
        return data

    @staticmethod
    def _down(data: numpy.ndarray) -> numpy.ndarray:
        """
        Shifts pixels
        :param data: array of pixels
        """
        mask = data != 0
        if min(data[mask]) > 0:
            if min(data[mask]) > 5:
                data[mask] -= 5
            else:
                data[mask] -= min(data[mask])
        return data

    def _dtype(self) -> None:
        """
        :return: numpy data type
        """
        if self.array_data_type == 0:
            return numpy.uint8
        elif self.array_data_type == 1:
            return numpy.int16
        elif self.array_data_type == 2:
            return numpy.int32
        elif self.array_data_type == 3:
            return numpy.float16
        elif self.array_data_type == 4:
            return numpy.float32

    def _sharpen(self, img: numpy.ndarray, alpha:float, sigma:float=1.0) -> numpy.ndarray:
        """
        appends sharpness to array of pixels
        :param img: array of pixels
        :param sigma: force of sharpness
        """
        blurred = scipy.ndimage.gaussian_filter(img, sigma=sigma)

        details = img.astype(float) - blurred.astype(float)

        sharpened = img.astype(float) + alpha * details

        sharpened = numpy.clip(sharpened, 0, 255)
        return sharpened

    @catch_errors
    def convert_to_gpic_file(self, pixel_matrix: list) -> io.BytesIO:
        '''
        converts an array of pixels
        :param pixel_matrix: convents a pixel matrix into an array of pixels in GPIC format
        :return: buffered image data
        '''

        height = len(pixel_matrix)
        width = len(pixel_matrix[0]) if height else 0
        buf = io.BytesIO()

        logger.debug(f"Array type: {self._dtype()}")
        logger.debug("Creating greyscale...")
        pixels = numpy.array(pixel_matrix, dtype=self._dtype())
        gray = numpy.dot(pixels, [0.2126, 0.7152, 0.0722])

        # Visual effects
        logger.debug("Applying visual effects...")
        gray += self.brightnes  # editing brightness
        gray = self._sharpen(gray, self.sharpness / 10)  # editing sharpness
        gray = self._down(gray)

        self.size_img_uncompress_DCT = len(gray.tobytes())

        if self.format_version == 3:

            logger.debug("Using DCT cumPression...")
            logger.debug(f"DCT block size: {self.dct_blocksize}")

            blocks, orig = self.Work_DCT._blockify(gray, block_size=self.dct_blocksize)

            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    blocks[i, j] = self.Work_DCT._dct2(blocks[i, j])

            dct_mat = blocks.swapaxes(1, 2).reshape(blocks.shape[0] * self.dct_blocksize, blocks.shape[1] * self.dct_blocksize)
            dct_mat = dct_mat[:orig[0], :orig[1]]

            logger.debug("Using quantization...")
            dct_mat_quantize = self._quantize(dct_mat, self.compression_dct_force, self.compression_type) #using quantization

            restored = dct_mat_quantize.astype(numpy.float16)

            restored = self._down(restored)

            self.size_img_uncompress = len(restored.tobytes())

            logger.debug("Using brotli compression...")

            compressed = brotli.compress(restored.tobytes(), quality=5)

            self.size_img = len(compressed)

            # Write sizes and data
            # Header: signature + dimensions
            buf.write(b"\x89GPC\n")  # CSIGN
            buf.write(b"O")  # CDAT
            buf.write(struct.pack('>I', 3)) #version
            buf.write(struct.pack('>I', width))
            buf.write(struct.pack('>I', height))
            buf.write(struct.pack('>I', self.size_img))
            buf.write(b"A")  # CPIX
            buf.write(compressed)
            buf.write(b"TIMAG") # CTXT
            buf.write(b"E") # CEND
            buf.seek(0)

            self.RMSE = numpy.mean((dct_mat - dct_mat_quantize) ** 2)
            logger.debug(f"RMSE: {self.RMSE}")

            logger.debug(f"file size: {self.size_img}bytes")
            del compressed, restored, gray, pixels, dct_mat, blocks, dct_mat_quantize
            return buf

        elif self.format_version == 2:

            logger.debug("Using DCT cumPression...")
            logger.debug(f"DCT block size: {self.dct_blocksize}")

            blocks, orig = self.Work_DCT._blockify(gray, block_size=self.dct_blocksize)

            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    blocks[i, j] = self.Work_DCT._dct2(blocks[i, j])

            dct_mat = blocks.swapaxes(1, 2).reshape(blocks.shape[0] * self.dct_blocksize, blocks.shape[1] * self.dct_blocksize)
            dct_mat = dct_mat[:orig[0], :orig[1]]
            logger.debug("Using quantization...")
            dct_mat_quantize = self._quantize(dct_mat, self.compression_dct_force, self.compression_type) #using quantization

            restored = dct_mat_quantize.astype(numpy.float16)

            restored = self._down(restored)

            self.size_img_uncompress = len(restored.tobytes())

            logger.debug("Using lzma compression...")

            compressed = lzma.compress(restored.tobytes(), preset=9)

            self.size_img = len(compressed)

            # Write sizes and data
            # Header: signature + dimensions
            buf.write(b"\x89GPC\n")  # CSIGN
            buf.write(b"O")  # CDAT
            buf.write(struct.pack('>I', 2)) #version
            buf.write(struct.pack('>I', width))
            buf.write(struct.pack('>I', height))
            buf.write(struct.pack('>I', self.size_img))
            buf.write(b"A")  # CPIX
            buf.write(compressed)
            buf.write(b"TIMAG") # CTXT
            buf.write(b"E") # CEND
            buf.seek(0)

            self.RMSE = numpy.mean((dct_mat - dct_mat_quantize) ** 2)
            logger.debug(f"RMSE: {self.RMSE}")

            logger.debug(f"file size: {self.size_img}bytes")
            del compressed, restored, gray, pixels, dct_mat, blocks, dct_mat_quantize
            return buf

        elif self.format_version == 1:
            gray_a = gray.copy()

            self.size_img_uncompress = len(gray_a.tobytes())

            logger.debug("Using lzma compression...")
            compressed = lzma.compress(gray_a.tobytes(), preset=9)

            self.size_img = len(compressed)

            # Write sizes and data
            # Header: signature + dimensions
            buf.write(b"\x89GPC\n")  # CSIGN
            buf.write(b"O")  # CDAT
            buf.write(struct.pack('>I', 1)) #version
            buf.write(struct.pack('>I', width))
            buf.write(struct.pack('>I', height))
            buf.write(struct.pack('>I', self.size_img))
            buf.write(b"A")  # CPIX
            buf.write(compressed)
            buf.write(b"TIMAG") # CTXT
            buf.write(b"E") # CEND
            buf.seek(0)

            self.RMSE = numpy.mean((gray - gray_a) ** 2)
            logger.debug(f"RMSE: {self.RMSE}")

            logger.debug(f"file size: {self.size_img}bytes")
            del gray, pixels, compressed
            return buf

        elif self.format_version == 0:

            self.size_img_uncompress = len(gray.tobytes())

            self.size_img = len(gray.tobytes())

            # Write sizes and data
            # Header: signature + dimensions
            buf.write(b"\x89GPC\n")  # CSIGN
            buf.write(b"O")  # CDAT
            buf.write(struct.pack('>I', 0)) #version
            buf.write(struct.pack('>I', width))
            buf.write(struct.pack('>I', height))
            buf.write(struct.pack('>I', self.size_img))
            buf.write(b"A")  # CPIX
            buf.write(gray.tobytes())
            buf.write(b"TIMAG") # CTXT
            buf.write(b"E") # CEND
            buf.seek(0)

            self.RMSE = 0
            logger.debug(f"RMSE: {self.RMSE}")

            logger.debug(f"file size: {self.size_img}bytes")
            del gray, pixels
            return buf

        else:
            raise ValueError("File format not found")

    @catch_errors
    def convert_to_gpic(self, pixel_matrix: list):
        '''
        converts an array of pixels
        :param pixel_matrix: convents a pixel matrix into an array of pixels in GPIC format
        :return: buffered image data
        '''

        height = len(pixel_matrix)
        width = len(pixel_matrix[0]) if height else 0

        logger.debug(f"Array type: {self._dtype()}")
        logger.debug("Creating greyscale...")
        pixels = numpy.array(pixel_matrix, dtype=self._dtype())
        gray = (0.2126 * pixels[..., 0] +
                0.7152 * pixels[..., 1] +
                0.0722 * pixels[..., 2])
        del pixels

        # Visual effects
        logger.debug("Applying visual effects...")
        gray += self.brightnes  # editing brightness
        gray = self._sharpen(gray, self.sharpness / 10)  # editing sharpness
        gray = self._down(gray)

        self.size_img_uncompress_DCT = len(gray.tobytes())

        if self.format_version == 3:

            logger.debug("Using DCT cumPression...")
            logger.debug(f"DCT block size: {self.dct_blocksize}")

            blocks, orig = self.Work_DCT._blockify(gray, block_size=self.dct_blocksize)

            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    blocks[i, j] = self.Work_DCT._dct2(blocks[i, j])

            dct_mat = blocks.swapaxes(1, 2).reshape(blocks.shape[0] * self.dct_blocksize,
                                                    blocks.shape[1] * self.dct_blocksize)
            dct_mat = dct_mat[:orig[0], :orig[1]]

            logger.debug("Using quantization...")
            dct_mat_quantize = self._quantize(dct_mat, self.compression_dct_force,
                                              self.compression_type)  # using quantization

            restored = dct_mat_quantize.astype(numpy.float16)

            restored = self._down(restored)

            self.size_img_uncompress = len(restored.tobytes())

            raw = numpy.frombuffer(restored, dtype=numpy.float16).reshape((height, width))

            logger.debug("Unblockifying and using IDCT...")
            blocks2, _ = self.Work_DCT._blockify(raw, block_size=self.dct_blocksize)
            for i in range(blocks2.shape[0]):
                for j in range(blocks2.shape[1]):
                    blocks2[i, j] = self.Work_DCT._idct2(blocks2[i, j])

            restored = self.Work_DCT._unblockify(blocks2, (height, width), block_size=self.dct_blocksize)
            restored = numpy.clip(numpy.rint(restored), 0, 255).astype(numpy.uint8)

            pixels = numpy.stack([restored, restored, restored], axis=-1)  # creating image

            logger.debug("Using brotli compression...")

            compressed = brotli.compress(raw, quality=6)

            self.size_img = len(compressed)

            self.RMSE = numpy.mean((dct_mat - dct_mat_quantize) ** 2)
            logger.debug(f"RMSE: {self.RMSE}")

            logger.debug(f"file size: {self.size_img}bytes")
            del compressed, restored, dct_mat, blocks, dct_mat_quantize

        elif self.format_version == 2:

            logger.debug("Using DCT cumPression...")
            logger.debug(f"DCT block size: {self.dct_blocksize}")

            blocks, orig = self.Work_DCT._blockify(gray, block_size=self.dct_blocksize)

            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    blocks[i, j] = self.Work_DCT._dct2(blocks[i, j])

            dct_mat = blocks.swapaxes(1, 2).reshape(blocks.shape[0] * self.dct_blocksize,
                                                    blocks.shape[1] * self.dct_blocksize)
            dct_mat = dct_mat[:orig[0], :orig[1]]

            logger.debug("Using quantization...")
            dct_mat_quantize = self._quantize(dct_mat, self.compression_dct_force, self.compression_type)  # using quantization

            restored = dct_mat_quantize.astype(numpy.float16)

            restored = self._down(restored)

            self.size_img_uncompress = len(restored.tobytes())

            raw = numpy.frombuffer(restored, dtype=numpy.float16).reshape((height, width))

            logger.debug("Unblockifying and using IDCT...")
            blocks2, _ = self.Work_DCT._blockify(raw, block_size=self.dct_blocksize)
            for i in range(blocks2.shape[0]):
                for j in range(blocks2.shape[1]):
                    blocks2[i, j] = self.Work_DCT._idct2(blocks2[i, j])

            restored = self.Work_DCT._unblockify(blocks2, (height, width), block_size=self.dct_blocksize)
            restored = numpy.clip(numpy.rint(restored), 0, 255).astype(numpy.uint8)

            pixels = numpy.stack([restored, restored, restored], axis=-1)  # creating image

            logger.debug("Using lzma compression...")

            compressed = lzma.compress(raw, preset=9)

            self.size_img = len(compressed)

            self.RMSE = numpy.mean((dct_mat - dct_mat_quantize) ** 2)
            logger.debug(f"RMSE: {self.RMSE}")

            logger.debug(f"file size: {self.size_img}bytes")
            del compressed, restored, dct_mat, blocks, dct_mat_quantize

        elif self.format_version == 1:

            self.size_img_uncompress = len(gray.tobytes())

            logger.debug("Using lzma compression...")

            compressed = lzma.compress(gray.tobytes(), preset=9)
            self.size_img = len(compressed)
            self.RMSE = numpy.mean((gray - gray) ** 2)

            logger.debug(f"RMSE: {self.RMSE}")
            logger.debug(f"file size: {self.size_img}bytes")

            restored = numpy.clip(numpy.rint(gray), 0, 255).astype(numpy.uint8)
            pixels = numpy.stack([restored, restored, restored], axis=-1)  # creating image
            del compressed

        elif self.format_version == 0:

            self.size_img_uncompress = len(gray.tobytes())
            self.size_img = len(gray.tobytes())
            self.RMSE = 0

            logger.debug(f"RMSE: {self.RMSE}")
            logger.debug(f"file size: {self.size_img}bytes")

            restored = numpy.clip(numpy.rint(gray), 0, 255).astype(numpy.uint8)
            pixels = numpy.stack([restored, restored, restored], axis=-1)  # creating image

        else:
            raise ValueError("File format not found")

        del gray
        return {"pixels" : pixels, "version" : self.format_version, "type" : "IMAG"}

    @catch_errors
    def open_image(self, pixels) -> Image.Image:
        '''
        turns the gpic format into a regular image
        :param file_obj: buffered gpic image data
        :return: pillow Image.image object
        '''

        pixels_raw = pixels["pixels"]

        if pixels["version"] == (0 or 1):
            gui.Create_widgets.Disable_wigets.toggle_dct_compression_slider(False)

        if pixels_raw is None:
            raise ValueError("Pixels is None")

        img = Image.fromarray(pixels_raw, mode='RGB')

        gui.root.title(f"Gpic converter | {IMAGE_FORMAT} | {img.size[0]} X {img.size[1]} | {pixels["type"]}")
        logger.debug(f"Image type: {pixels["type"]}")

        return ImageOps.exif_transpose(img)


class Gui:
    def __init__(self):
        self.Create_widgets = self.Create_widgets(self)
        self.On_triggers = self.On_triggers(self)
        self.Debug = self.Debug(self)

        self.root: Optional[tk.Tk] = None

    #creates main root window
    def create_window(self) -> None:
        '''
        creates main window
        '''
        root = tk.Tk()
        root.iconbitmap(resource_path("GPIC_logo.ico"))
        root.title("Gpic converter | loading...")
        root.geometry("1000x800")
        root.resizable(False, False)

        #creating button
        btn_view = tk.Button(text="Update Image", command=self.On_triggers.on_button_view_update_image)
        btn_view.pack(anchor="nw")
        btn_view.place(x=15, y=15)
        self.root = root


    def re_create_window(self) -> None:
        self.root.destroy()
        main()


    class Create_widgets:

        def __init__(self, gui_instance):
            self.Gui = gui_instance
            self.Disable_wigets = self.Disable_wigets(self)

            self.dct_compression_slider: Optional[tk.Scale] = None
            self.image_label: Optional[tk.Label] = None
            self.size_looker_label: Optional[tk.Label] = None
            self.slider_brightness: Optional[tk.Scale] = None
            self.slider_sharpness: Optional[tk.Scale] = None
            self.slider_compression: Optional[tk.Scale] = None
            self.image_viewer_frame: Optional[tk.Frame] = None
            self.version_label: Optional[tk.Label] = None
            self.loading_label: Optional[tk.Label] = None
            self.loading_viewer_frame: Optional[tk.Frame] = None

            self.edit_compr_type_menu_var: Optional[tk.IntVar] = None
            self.edit_format_version_var: Optional[tk.IntVar] = None

        def create_main_widgets(self, image=None):
            """
            Creates main widgets
            """
            if image is None:
                self._create_version_label()
                self._create_menu()
                self._create_image_brightness_slider()
                self._create_image_sharpness_slider()
                self._create_dct_compression_slider()
                self._create_size_looker_label()
                self._create_loading_label("Waiting...")
            else:
                self.loading_label.destroy()
                self.loading_viewer_frame.destroy()

                self.Gui.Create_widgets.size_looker_label.destroy()
                self._create_image_viewer(image)
                self._create_size_looker_label()
                #del image


        def _create_loading_label(self, text) -> None:
            """
            Creates loading screen on center of window
            :param text: Text
            """

            # If user presses "Update Image" button
            if work_with_gppic.file_path is not None:
                image = Image.open(work_with_gppic.file_path).convert("L").convert("RGBA")
                overlay = Image.new("RGBA", image.size, (0, 0, 0, 125)) #creating new inage-overlay for shade
                image = Image.alpha_composite(image, overlay)
                image.thumbnail((550, 550), Image.Resampling.LANCZOS)

                image = ImageTk.PhotoImage(image)

                loading_viewer_frame = tk.Frame(self.Gui.root, bg="white", bd=5, relief=tk.GROOVE)
                loading_viewer_frame.pack(anchor="center", pady=100)
                loading_viewer_frame.configure(width=530, height=530)
                loading_viewer_frame.pack_propagate(False)

                loading_label = tk.Label(loading_viewer_frame, image=image, bg="white", text=text, font=("Arial", 60), compound="center", fg="white")
                loading_label.image = image
                loading_label.pack(anchor="center", ipady=200)

                self.loading_label = loading_viewer_frame
                self.loading_viewer_frame = loading_label

            # if the user uploaded the image for the first time
            else:

                loading_viewer_frame = tk.Frame(self.Gui.root, bg="white", bd=5, relief=tk.GROOVE)
                loading_viewer_frame.pack(anchor="center", pady=100)
                loading_viewer_frame.configure(width=530, height=530)
                loading_viewer_frame.pack_propagate(False)

                loading_label = tk.Label(loading_viewer_frame, text=text, bg="white", font=("Arial", 60))
                loading_label.pack(anchor="center", ipady=200)
                self.loading_label = loading_label
                self.loading_viewer_frame = loading_viewer_frame

        # creates text label with data of image sizes
        def _create_size_looker_label(self) -> None:
            if work_with_gppic.file_path is not None:
                size = self.Gui.format_size(os.path.getsize(work_with_gppic.file_path))
            else:
                size = self.Gui.format_size(0)
            size_looker_label = tk.Label(self.Gui.root,
                                         text=f"Original image size: {size}\n"
                                              f"Now file size: {self.Gui.format_size(work_with_gppic.size_img)} ± 5Kb\n"
                                              f"RMSE: {round(work_with_gppic.RMSE, 2)}",
                                         font=("Arial", 10))
            size_looker_label.pack(anchor="sw")
            size_looker_label.place(y=720, x=5)
            self.size_looker_label = size_looker_label

        #create slider for DCT compression
        def _create_dct_compression_slider(self) -> None:
            """
            Creates slider for compression
            """
            compression_frame = tk.Frame(self.Gui.root, bg="#FFADB0", bd=5, relief=tk.GROOVE)
            compression_frame.pack(anchor="nw", fill=tk.NONE, expand=False)
            compression_frame.place(x=10, y=60, width=110, height=460)

            slider_compression_label = tk.Label(compression_frame, text="Quantization\ncompression", font=("Arial", 12),
                                                bg="#FFADB0")

            slider_compression_label.pack(anchor="center")
            slider_compression_label.place(x=1)

            slider_compression = tk.Scale(
                compression_frame,
                bg="#FFADB0",
                bd=3,
                from_=1,
                to=255,
                orient=tk.VERTICAL,
                length=600,
                command=self.Gui.On_triggers.on_dct_slider_compression,
            )

            slider_compression.pack()
            slider_compression.place(height=390, y=45, x=25)
            slider_compression.set(work_with_gppic.compression_dct_force)

            self.dct_compression_slider = slider_compression

        def _create_image_brightness_slider(self) -> None:
            """
            Creates slider for brightness
            """
            brightness_frame = tk.Frame(self.Gui.root, bg="#AFEEEE", bd=5, relief=tk.GROOVE)
            brightness_frame.pack(anchor="nw", fill=tk.NONE, expand=False)
            brightness_frame.place(x=880, y=60, width=110, height=320)

            slider_brightness_label = tk.Label(brightness_frame, text="  brightness", font=("Arial", 12),
                                                bg="#AFEEEE")

            slider_brightness_label.pack(anchor="center")
            slider_brightness_label.place(x=1)

            slider_brightness = tk.Scale(
                brightness_frame,
                bg="#AFEEEE",
                bd=3,
                from_=100,
                to=-100,
                orient=tk.VERTICAL,
                length=600,
                command=self.Gui.On_triggers.on_bright_slider,
            )

            slider_brightness.pack()
            slider_brightness.place(height=270, y=30, x=25)
            slider_brightness.set(0)

            self.slider_brightness = slider_brightness

        def _create_image_sharpness_slider(self) -> None:
            """
            Creates slider for sharpness
            """
            sharpness_frame = tk.Frame(self.Gui.root, bg="#C0C0C0", bd=5, relief=tk.GROOVE)
            sharpness_frame.pack(anchor="nw", fill=tk.NONE, expand=False)
            sharpness_frame.place(x=880, y=400, width=110, height=320)

            slider_sharpness_label = tk.Label(sharpness_frame, text="  sharpness", font=("Arial", 12),
                                                bg="#C0C0C0")

            slider_sharpness_label.pack(anchor="center")
            slider_sharpness_label.place(x=1)

            slider_sharpness = tk.Scale(
                sharpness_frame,
                bg="#C0C0C0",
                bd=3,
                from_=0,
                to=100,
                orient=tk.VERTICAL,
                length=600,
                command=self.Gui.On_triggers.on_sharp_slider,
            )

            slider_sharpness.pack()
            slider_sharpness.place(height=270, y=30, x=25)
            slider_sharpness.set(0)

            self.slider_sharpness = slider_sharpness

        def _create_image_viewer(self, image) -> None:
            """
            creates image preview window
            """

            #creating frame of preview window
            image_viewer_frame = tk.Frame(self.Gui.root, bg="white", bd=5, relief=tk.GROOVE)
            image_viewer_frame.pack(anchor="center", pady=100)
            image_viewer_frame.configure(width=530, height=530)
            image_viewer_frame.pack_propagate(False)

            image.thumbnail((550, 550), Image.Resampling.LANCZOS)

            image_ = ImageTk.PhotoImage(image)

            #creating label with image
            image_label = tk.Label(image_viewer_frame, image=image_, bg="white")
            image_label.image = image_
            image_label.pack(anchor="center", ipady=200)

            self.image_viewer_frame = image_viewer_frame
            self.image_label = image_label

        def _create_version_label(self) -> None:
            version_label = tk.Label(self.Gui.root, text=f"GPIC Conventer V{VERSION} ", font=("Arial", 10))
            version_label.pack(anchor="se")
            self.version_label = version_label

        def _create_menu(self) -> None:
            menu = tk.Menu(self.Gui.root)
            self.Gui.root.option_add("*tearOff", tk.FALSE)

            file_menu = tk.Menu()
            file_save_as_menu = tk.Menu()
            edit_menu = tk.Menu()
            debug_menu = tk.Menu()
            edit_format_version = tk.Menu()
            edit_compr_type_menu = tk.Menu()

            file_save_as_menu.add_command(label="Export as .GPIC", command=self.Gui.export_file)
            file_save_as_menu.add_command(label="Export as .PNG", command=self.Gui.export_file_as_png)

            self.edit_compr_type_menu_var = tk.IntVar(value=work_with_gppic.compression_type)
            edit_compr_type_menu.add_radiobutton(label="round ceil", command=lambda: self.Gui.On_triggers.on_edit_compression_type(), variable=self.edit_compr_type_menu_var, value=2)
            edit_compr_type_menu.add_radiobutton(label="round Default", command=lambda: self.Gui.On_triggers.on_edit_compression_type(), variable=self.edit_compr_type_menu_var, value=1)
            edit_compr_type_menu.add_radiobutton(label="round floor", command=lambda: self.Gui.On_triggers.on_edit_compression_type(), variable=self.edit_compr_type_menu_var, value=0)

            self.edit_format_version_var = tk.IntVar(value=work_with_gppic.format_version)
            edit_format_version.add_radiobutton(label="3", command=lambda: self.Gui.On_triggers.on_edit_format_version(), variable=self.edit_format_version_var, value=3)
            edit_format_version.add_radiobutton(label="2", command=lambda: self.Gui.On_triggers.on_edit_format_version(), variable=self.edit_format_version_var, value=2)
            edit_format_version.add_radiobutton(label="1", command=lambda: self.Gui.On_triggers.on_edit_format_version(), variable=self.edit_format_version_var, value=1)
            edit_format_version.add_radiobutton(label="0", command=lambda: self.Gui.On_triggers.on_edit_format_version(), variable=self.edit_format_version_var, value=0)

            file_menu.add_command(label="New", command=self.Gui.re_create_window)
            file_menu.add_cascade(label="Export", menu=file_save_as_menu)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=sys.exit)

            edit_menu.add_cascade(label="Compression type", menu=edit_compr_type_menu)
            edit_menu.add_cascade(label="Format version", menu=edit_format_version)

            debug_menu.add_command(label="image_data", command=self.Gui.Debug.get_image_data)
            debug_menu.add_command(label="show_image_now", command=self.Gui.Debug.show_image)
            debug_menu.add_command(label="show_image_compress_off", command=self.Gui.Debug.show_image_virgin)
            debug_menu.add_command(label="show_original_image", command=self.Gui.Debug.show_original)
            debug_menu.add_checkbutton(label="export_logs", command=lambda: self.Gui.Debug.export_logs())

            menu.add_cascade(label="File", menu=file_menu)
            menu.add_cascade(label="Edit", menu=edit_menu)
            menu.add_cascade(label="Debug", menu=debug_menu)
            menu.configure(activeborderwidth=5)

            self.Gui.root.config(menu=menu)

        class Disable_wigets:

            def __init__(self, Create_widgets):
                self.Create_widgets = Create_widgets

            def toggle_dct_compression_slider(self, state:bool):
                scale = self.Create_widgets.dct_compression_slider

                if not state:
                    scale.config(state="disabled")
                else:
                    scale.config(state="normal")


    class Get_windows:

        @staticmethod
        def get_folder(filetypes, defaultextension) -> str:
            '''
            creates window for choosing folder
            :param filetypes: filetypes
            :param defaultextension: default file-type
            :return: path
            '''
            initialfile = ''.join(work_with_gppic.file_path.split("/")[-1:])

            export_path = filedialog.asksaveasfilename(
                defaultextension=defaultextension,
                initialdir="/",
                initialfile=''.join(initialfile.split(".")[:-1]),
                filetypes=filetypes,
                title="Save file as"
            )
            return export_path


        @staticmethod
        def get_path(filetypes) -> str:
            '''
            creates window for choosing path
            :param filetypes: filetypes
            :return: path to file
            '''
            path = filedialog.askopenfilename(
                title="Choose file",
                filetypes=filetypes
            )
            return path


    class On_triggers:

        def __init__(self, gui_instance):
            self.Gui = gui_instance

        # updates image in preview window
        def on_button_view_update_image(self) -> None:
            logger.debug("\n")

            tim = time.time()
            # re-creating image with new settings
            self.Gui.Create_widgets.dct_compression_slider.destroy()
            self.Gui.Create_widgets._create_dct_compression_slider()
            self.Gui.Create_widgets.image_label.destroy()
            self.Gui.Create_widgets.image_viewer_frame.destroy()
            self.Gui.Create_widgets.loading_viewer_frame.destroy()
            self.Gui.Create_widgets.loading_label.destroy()

            self.Gui.Create_widgets._create_loading_label("Loading...")
            self.Gui.Create_widgets.loading_label.update()

            pixel_matrix = work_with_gppic.extract_pixels(work_with_gppic.file_path)
            file_image = work_with_gppic.convert_to_gpic(pixel_matrix)
            file = work_with_gppic.open_image(file_image)

            self.Gui.Create_widgets.size_looker_label.destroy()
            self.Gui.Create_widgets.loading_viewer_frame.destroy()
            self.Gui.Create_widgets.loading_label.destroy()

            self.Gui.Create_widgets._create_image_viewer(file)
            self.Gui.Create_widgets._create_size_looker_label()

            del pixel_matrix, file_image
            logger.info(f"DONE for {round(time.time() - tim, 3)}s")


        # edits DST CUMpression value in Work_with_gppic class
        @staticmethod
        def on_dct_slider_compression(value) -> None:
            work_with_gppic.compression_dct_force = int(value)

        @staticmethod
        def on_bright_slider(value) -> None:
            work_with_gppic.brightnes = int(value)

        @staticmethod
        def on_sharp_slider(value) -> None:
            work_with_gppic.sharpness = int(value)

        def on_edit_compression_type(self) -> None:
            work_with_gppic.compression_type = self.Gui.Create_widgets.edit_compr_type_menu_var.get()
            logger.debug(f"Compression type has been edited to {self.Gui.Create_widgets.edit_compr_type_menu_var.get()}")
            work_with_gppic.Work_TOML.edit_toml_data("compression_type", self.Gui.Create_widgets.edit_compr_type_menu_var.get())

        def on_edit_format_version(self) -> None:
            work_with_gppic.format_version = self.Gui.Create_widgets.edit_format_version_var.get()
            logger.debug(f"Format version has been edited to {self.Gui.Create_widgets.edit_format_version_var.get()}")
            work_with_gppic.Work_TOML.edit_toml_data("file_version",  self.Gui.Create_widgets.edit_format_version_var.get())


    class Debug:

        def __init__(self, gui_instance):
            self.Gui = gui_instance

        @staticmethod
        def get_image_data():
            '''
            shows info-window with info of image
            '''

            dct_compression_forse_data  = work_with_gppic.compression_dct_force
            quantization_compression_forse_data = work_with_gppic.compression_quant_force
            compression_type_data = work_with_gppic.compression_type

            dct_blocksize = work_with_gppic.dct_blocksize

            showinfo(title="get_image_data", message=f"dct_compression_forse : {dct_compression_forse_data}"
                                                     f"\nquantization_compression_forse_data : {quantization_compression_forse_data}"
                                                     f"\ncompression_type : {compression_type_data}"
                                                     f"\nRMSE : {round(work_with_gppic.RMSE, 6)}"
                                                     f"\ndct_blocksize : {dct_blocksize}"
                                                     f"\n\nuncompressed (by DCT & quantization) bytes size:  {work_with_gppic.size_img_uncompress_DCT}"
                                                     f"\nuncompressed (by Lzma) bytes size:  {work_with_gppic.size_img_uncompress}"f""
                                                     f"\nnow bytes size : {work_with_gppic.size_img} ± 5Kb"
                                                     f"\n\nformat_version: {work_with_gppic.format_version}")

        @staticmethod
        def show_image():
            logger.debug("Loading...")

            tim = time.time()

            pixel_matrix = work_with_gppic.extract_pixels(work_with_gppic.file_path)
            img = work_with_gppic.convert_to_gpic(pixel_matrix)
            work_with_gppic.open_image(img).show()

            logger.info(f"DONE for {round(time.time() - tim, 3)}s")

        @staticmethod
        def show_image_virgin():
            '''
            shows image without compression
            '''
            work_with_gppic_virgin = Work_with_gppic(
                compression_dct_force = 1,
                compression_quant_force = 1,
                brightnes = 0,
                sharpness = 0
            )
            pixel_matrix_virgin = work_with_gppic_virgin.extract_pixels(work_with_gppic.file_path)
            file_image_virgin = work_with_gppic_virgin.convert_to_gpic(pixel_matrix_virgin)
            img = work_with_gppic_virgin.open_image(file_image_virgin)
            del pixel_matrix_virgin, file_image_virgin
            img.show()

        @staticmethod
        def show_original():
            os.startfile(work_with_gppic.file_path)

        @staticmethod
        def export_logs(log_type=None):
            def export_logger():
                buffer_handler.flush()
                log_buffer.seek(0)
                return log_buffer.read()
            if log_type == 'ERROR':
                now_time = time.ctime(time.time()).replace(" ", "_").replace(":", ".")
                with open(f"./Crash_Report_{now_time}.log", "w", encoding="UTF-8") as log:
                    log.write(export_logger())
                showinfo("GPIC", f"Crash report were successfully exported at path: {os.path.abspath(f"{now_time}.log")}")
            else:
                now_time = time.ctime(time.time()).replace(" ", "_").replace(":", ".")
                with open(f"./{now_time}.log", "w", encoding="UTF-8") as log:
                    log.write(export_logger())

                showinfo("GPIC", f"logs were successfully exported at path: {os.path.abspath(f"{now_time}.log")}")



    def export_file_as_png(self) -> None:
        '''
        exports image as .png
        :param img: image data
        '''
        export_path = self.Get_windows.get_folder([("png", "*.png"), ("All files", "*.*")], ".png")
        if export_path == "":
            return None
        else:
            pixel_matrix = work_with_gppic.extract_pixels(work_with_gppic.file_path)
            img = work_with_gppic.convert_to_gpic(pixel_matrix)
            img = work_with_gppic.open_image(img)
            img.save(export_path, format="PNG")

            logger.info("image has been exported as PNG!")
            showinfo("Exported!", "Image was successfully exported as PNG!")


    def export_file(self) -> None:
        '''
        exports image in .gppic format
        :param img: image data
        '''
        export_path = self.Get_windows.get_folder([("gpic", "*.gpic"), ("All files", "*.*")], ".gppic")
        if export_path == "":
            pass
        else:
            with open(export_path, "wb") as f:
                pixel_matrix = work_with_gppic.extract_pixels(work_with_gppic.file_path)
                img = work_with_gppic.convert_to_gpic_file(pixel_matrix)
                f.write(img.getvalue())


            logger.info("file has been successfully exported!")
            showinfo("Exported!", "Image was successfully exported as GPIC!")


    @staticmethod
    def format_size(size) -> str:
        '''
        gets bytes and returns them in a beautiful wrapper for user
        :param size:
        '''
        units = {
            1_000_000_000_000: 'Tb',
            1_000_000_000: 'Gb',
            1_000_000: 'Mb',
            1_000: 'Kb',
            1: 'b'
        }

        for threshold, unit in units.items():
            if size >= threshold:
                return f"{size / threshold:.1f}{unit}"


#starts program
@catch_errors
def main():
    global file_image
    global buffer_handler, log_buffer
    global work_with_gppic, gui

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
    gui.Create_widgets.create_main_widgets() # creating main widgets

    path = gui.Get_windows.get_path([("Images", "*.png;*.jpg"), ("Txt files", "*.txt"), ("All files", "*.*")]) # Requesting the path to the image from the user

    work_with_gppic.file_path = path

    if path == "":
        sys.exit()

    tim = time.time()

    #gui.Create_widgets.loading_label.configure(text="loading...") # creating loading screen
    gui.Create_widgets.loading_label.update() # we have to manually refresh the window because we havent started the main loop

    pixel_matrix = work_with_gppic.extract_pixels(path) # getting pixels-matrix from image
    file_image = work_with_gppic.convert_to_gpic(pixel_matrix) # getting image in GPIC format
    gui.Create_widgets.create_main_widgets(work_with_gppic.open_image(file_image)) # creating other widgets
    del file_image, pixel_matrix

    logger.info(f"DONE for {round(time.time() - tim, 3)}s")
    gui.root.mainloop() # Starting main loop


if __name__ == "__main__":
    gui = Gui()
    work_with_gppic = Work_with_gppic()

    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    #logging.disable(logging.CRITICAL)
    main()
