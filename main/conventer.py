import time
from PIL import Image, ImageTk, ImageOps
import numpy
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror, showwarning, showinfo
import lzma
import io
import logging
import struct
import os
import functools
from scipy.fftpack import dct, idct
from typing import Optional, List, Tuple, BinaryIO
import coloredlogs
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

VERSION = "1.4"

# pigar generate - create requirements.txt
# pyinstaller --onefile --add-data "GPIC_logo.ico;." --icon=GPIC_logo.ico --noconsole --collect-all scipy conventer.py
# pyinstaller --onefile --add-data "GPIC_logo.ico;." --icon=GPIC_logo.ico --noconsole --collect-all scipy image_loader.py

#handler that creates small loading window
def loading_screen(func):
    '''
    creates mini-window - loading screen
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def disable_close(): pass

        loading_win = tk.Toplevel(root)
        loading_win.iconbitmap(resource_path("GPIC_logo.ico"))
        loading_win.title("Loading...")
        loading_win.transient(root)
        loading_win.grab_set()
        loading_win.protocol("WM_DELETE_WINDOW", disable_close)

        loading_win.geometry("200x100+{}+{}".format(
            (loading_win.winfo_screenwidth() - 200) // 2,
            (loading_win.winfo_screenheight() - 100) // 2
        ))

        tk.Label(loading_win, text="Loading...").pack(anchor="n", pady=10)
        pb = ttk.Progressbar(loading_win, mode="indeterminate")
        pb.pack(expand=True, fill="x", padx=20, pady=20)
        pb.start(2) #makes the progressbar always move

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)

            def check():
                if future.done():
                    loading_win.destroy()
                else:
                    loading_win.after(50, check)

            loading_win.after(50, check)
            root.wait_window(loading_win)
            return future.result()

    return wrapper

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
            exit()
    return wrapper

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

#Class for working with gppic/other images files
class Work_with_gppic:

    def __init__(self, compression_dct_force: int = 10, compression_quant_force: int = 1, compression_type: int = 1, array_data_type: int = 0, dct_blocksize: int = 8):
        '''
        :param compression_dct_force: compression force during DCT compression
        :param compression_quant_force: compression force during quantization compression
        :param compression_type: if == 1: rounds the value, if == 2: rounds upward, if == 3: rounds down
        :param array_data_type: defines the data type of the array
        :param dct_blocksize: size of the blocks into which the image will be divided during DCT compression
        '''
        self.compression_dct_force = compression_dct_force
        self.compression_quant_force = compression_quant_force
        self.compression_type = compression_type  # 0=floor,1=round,2=ceil
        self.array_data_type = array_data_type
        self.dct_blocksize = dct_blocksize

    @staticmethod
    @catch_errors
    def extract_pixels_from_png(path: str) -> Optional[List[List[Tuple[int, int, int]]]]:
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
    def _dct2(block: numpy.ndarray) -> numpy.ndarray:
        '''
        applies the DCT compression method
        :param block: 1 block - array of pixels
        '''
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

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
    def _unblockify(blocks: numpy.ndarray, orig_shape: Tuple[int, int], block_size: int = 8):
        n_v, n_h, _, _ = blocks.shape
        padded = blocks.swapaxes(1, 2).reshape(n_v * block_size, n_h * block_size)
        h, w = orig_shape
        return padded[:h, :w]

    @staticmethod
    def _quantize(data: numpy.ndarray, force: int, method: int) -> numpy.ndarray:
        '''
        applies the quantization method
        :param data: array of pixels
        :param force: force of compression
        :param method: if == 1: rounds the value, if == 2: rounds upward, if == 3: rounds down
        :return:
        '''

        def _quantize(force, data, method):
            data =  data.astype(numpy.int32)
            if force != 1:
                mask = data != 0
                if method == 1:
                    data[mask] = numpy.round(data[mask] / force) * force
                elif method == 2:
                    data[mask] = numpy.ceil(data[mask] / force) * force
                elif method == 0:
                    data[mask] = numpy.floor(data[mask] / force) * force
            return data

        return _quantize(force, data, method)

    @staticmethod
    def _down(data: numpy.ndarray):
        mask = data != 0
        if min(data[mask]) > 0:
            if min(data[mask]) > 5:
                data[mask] -= 5
            else:
                data[mask] -= min(data[mask])
        return data

    def _dtype(self):
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


    #converts list with png pixels to .gppic file in ram
    #@loading_screen
    @catch_errors
    def convert_to_gppic(self, pixel_matrix: list) -> io.BytesIO:
        '''
        converts an array of pixels
        :param pixel_matrix: convents a pixel matrix into an array of pixels in GPIC format
        :return: buffered image data
        '''
        global size_img, size_img_uncompress, size_img_uncompress_DCT, RMSE

        height = len(pixel_matrix)
        width = len(pixel_matrix[0]) if height else 0
        buf = io.BytesIO()

        # Header: signature + dimensions
        buf.write(b"\x89GPC\nO")
        buf.write(struct.pack('>I', width))
        buf.write(struct.pack('>I', height))
        buf.write(b"A")

        logger.debug("Using greyscale...")
        logger.debug(f"Array type: {self._dtype()}")
        pixels = numpy.array(pixel_matrix, dtype=self._dtype())
        gray = (0.299 * pixels[..., 0] +
                0.587 * pixels[..., 1] +
                0.114 * pixels[..., 2])


        gray = self._down(gray)

        size_img_uncompress_DCT = len(gray.tobytes())

        logger.debug("Using DCT cumPression...")
        logger.debug(f"DCT block size: {self.dct_blocksize}")
        blocks, orig = self._blockify(gray, block_size=self.dct_blocksize)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                blocks[i, j] = self._dct2(blocks[i, j])
        dct_mat = blocks.swapaxes(1, 2).reshape(blocks.shape[0] * self.dct_blocksize, blocks.shape[1] * self.dct_blocksize)
        dct_mat = dct_mat[:orig[0], :orig[1]]
        logger.debug("Using quantization...")
        dct_mat_quantize = self._quantize(dct_mat, self.compression_dct_force, self.compression_type) #using quantization

        restored = dct_mat_quantize.astype(numpy.float16)

        restored = self._down(restored)

        size_img_uncompress = len(restored.tobytes())

        logger.debug("Using lzma compression...")

        compressed = lzma.compress(restored.tobytes(), preset=9)

        size_img = len(compressed)

        # Write sizes and data
        buf.write(struct.pack('>I', size_img))
        buf.write(compressed)
        buf.write(b"TIMAGE")
        buf.seek(0)

        RMSE = numpy.mean((dct_mat - dct_mat_quantize) ** 2)

        logger.debug(f"file size: {size_img}bytes")
        del compressed, restored, gray, pixels
        return buf

    @staticmethod
    def idct2(block: numpy.ndarray) -> numpy.ndarray:
        if block.ndim != 2 or block.shape[0] != block.shape[1]:
            raise ValueError(f"idct2 waiting square block, got {block.shape}")
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def unblockify(blocks: numpy.ndarray,
                   orig_shape: Tuple[int, int],
                   block_size: int = 8) -> numpy.ndarray:
        logger.debug("unblocking array...")
        h, w = orig_shape
        n_v, n_h, b_h, b_w = blocks.shape

        if b_h != block_size or b_w != block_size:
            raise ValueError(f"block size {b_h}x{b_w} not relevant {block_size}")
        full = (
            blocks
            .swapaxes(1, 2)
            .reshape(n_v * block_size, n_h * block_size)
        )
        return full[:h, :w]

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

                logger.debug("Decompressing...")
                raw = lzma.decompress(comp_data)

                inv = numpy.frombuffer(raw, dtype=numpy.float16).reshape((height, width))

                logger.debug("Blockifying and using IDCT...")
                blocks2, _ = self._blockify(inv, block_size=self.dct_blocksize)
                for i in range(blocks2.shape[0]):
                    for j in range(blocks2.shape[1]):
                        blocks2[i, j] = self._idct2(blocks2[i, j])

                restored = self._unblockify(blocks2, (height, width), block_size=self.dct_blocksize)
                restored = numpy.clip(numpy.rint(restored), 0, 255).astype(numpy.uint8)

                pixels = numpy.stack([restored, restored, restored], axis=-1) #creating image

            elif chunk_type == b"T":
                title = data[offset:offset + 4]
                offset += 4
                root.title(f"Gpic converter | {IMAGE_FORMAT} | {title.decode('utf-8')}")
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


#Class for encoding int/str to a byte object
class ToBytes:
    @staticmethod
    def to_bytes_int(number, length) -> bytes:
        '''
        encodes int to a byte object
        :param number: number
        :param length: bytes len
        :return: number in bytes
        '''
        if number is None:
            raise ValueError("attribute 'number' not found.")
        if not isinstance(number, int):
            raise TypeError("'number' shood be integer.")
        if length <= 0:
            raise ValueError("'length' must be >0.")
        return number.to_bytes(length, byteorder='big')

    # encodes str to a byte object
    @staticmethod
    def to_bytes_str(text) -> bytes:
        '''
        encodes str to a byte object
        :param text: text
        :return: string in bytes
        '''
        if text is None:
            raise ValueError("attribute 'text' not found.")
        if not isinstance(text, str):
            raise TypeError("'text' should be str.")
        try:
            return text.encode('ascii')
        except UnicodeEncodeError:
            raise ValueError("String contains invalid ASCII characters..")


#class for creating and working with GUIs
class Gui:
    def __init__(self):
        self.Create_widgets = self.Create_widgets(self)
        self.On_triggers = self.On_triggers(self)
        self.Debug = self.Debug(self)

    #creates main root window
    def create_window(self) -> None:
        '''
        creates main window
        '''
        global root

        root = tk.Tk()
        root.iconbitmap(resource_path("GPIC_logo.ico"))
        root.title("Gpic converter | loading...")
        root.geometry("1000x800")
        root.resizable(False, False)


        #creating buttons
        btn_view = tk.Button(text="Update Image", command=self.On_triggers.on_button_view_update_image)
        btn_view.pack(anchor="nw")
        btn_view.place(x=15, y=15)

    @staticmethod
    def re_create_window() -> None:
        root.destroy()
        main()


    class Create_widgets:

        def __init__(self, gui_instance):
            self.Gui = gui_instance

        def create_main_widgets(self, image):
            self.create_menu()
            self.create_version_label()
            self.create_image_viewer(image)
            self.create_dct_compression_slider()
            #self.create_quantization_compression_slider()
            self.create_size_looker_label()

        # creates text label with data of image sizes
        def create_size_looker_label(self) -> None:
            global size_looker_label

            size_looker_label = tk.Label(root,
                                         text=f"Original image size: {self.Gui.format_size(os.path.getsize(path))}\n"
                                              f"Now file size: {self.Gui.format_size(size_img)} ± 5Kb\n"
                                              f"RMSE: {round(RMSE, 2)}",
                                         font=("Arial", 10))
            size_looker_label.pack(anchor="sw")

        #create slider for DCT compression
        def create_dct_compression_slider(self) -> None:
            compression_frame = tk.Frame(root, bg="#FFADB0", bd=5, relief=tk.GROOVE)
            compression_frame.pack(anchor="nw", fill=tk.NONE, expand=False)
            compression_frame.place(x=10, y=60, width=110, height=460)

            slider_compression_label = tk.Label(compression_frame, text="DCT\ncompression", font=("Arial", 12),
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
            slider_compression.place(height=360, y=45, x=25)
            slider_compression.set(10)

            # create slider for quantization compression

        #create slider for quantization compression
        def create_quantization_compression_slider(self) -> None:
                compression_frame = tk.Frame(root, bg="lightblue", bd=5, relief=tk.GROOVE)
                compression_frame.pack(anchor="nw", fill=tk.NONE, expand=False)
                compression_frame.place(x=10, y=395, width=110, height=320)

                slider_compression_label = tk.Label(compression_frame, text="quantization\ncompression", font=("Arial", 12),
                                                    bg="lightblue")

                slider_compression_label.pack(anchor="center")
                slider_compression_label.place(x=1)

                slider_compression = tk.Scale(
                    compression_frame,
                    bg="lightblue",
                    bd=3,
                    from_=1,
                    to=120,
                    orient=tk.VERTICAL,
                    length=300,
                    command=self.Gui.On_triggers.on_quantization_slider_compression,
                )

                slider_compression.pack()
                slider_compression.place(height=250, y=45, x=25)

        #creates image preview window
        @staticmethod
        def create_image_viewer(image) -> None:
            global image_label
            global image_viewer_frame

            #creating frame of preview window
            image_viewer_frame = tk.Frame(root, bg="white", bd=5, relief=tk.GROOVE)
            image_viewer_frame.pack(anchor="center", pady=100)
            image_viewer_frame.configure(width=530, height=530)
            image_viewer_frame.pack_propagate(False)

            image.thumbnail((550, 550), Image.Resampling.LANCZOS)

            image_ = ImageTk.PhotoImage(image)

            #creating label with image
            image_label = tk.Label(image_viewer_frame, image=image_, bg="white")
            image_label.image = image_
            image_label.pack(anchor="center", ipady=200)

        def create_version_label(self):
            version_label = tk.Label(root, text=f"GPIC Conventer V{VERSION} ", font=("Arial", 10))
            version_label.pack(anchor="se")

        def create_menu(self) -> None:
            global edit_compr_type_menu_var, edit_array_data_type_var, edit_dct_block_size_var, show_console_var
            menu = tk.Menu(root)
            root.option_add("*tearOff", tk.FALSE)

            file_menu = tk.Menu()
            file_save_as_menu = tk.Menu()
            edit_menu = tk.Menu()
            debug_menu = tk.Menu()
            edit_array_data_type = tk.Menu()
            edit_compr_type_menu = tk.Menu()
            edit_dct_block_size = tk.Menu()

            file_save_as_menu.add_command(label="Export as .GPIC", command=lambda: self.Gui.export_file(file_image))
            file_save_as_menu.add_command(label="Export as .PNG", command=lambda: self.Gui.export_file_as_png(file_image))

            edit_compr_type_menu_var = tk.IntVar(value=1)
            edit_compr_type_menu.add_radiobutton(label="round ceil", command=lambda: self.Gui.On_triggers.on_edit_compression_type(), variable=edit_compr_type_menu_var, value=2)
            edit_compr_type_menu.add_radiobutton(label="round Default", command=lambda: self.Gui.On_triggers.on_edit_compression_type(), variable=edit_compr_type_menu_var, value=1)
            edit_compr_type_menu.add_radiobutton(label="round floor", command=lambda: self.Gui.On_triggers.on_edit_compression_type(), variable=edit_compr_type_menu_var, value=0)

            file_menu.add_command(label="New", command=Gui.re_create_window)
            file_menu.add_cascade(label="Export", menu=file_save_as_menu)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=sys.exit)

            edit_menu.add_cascade(label="Compression type", menu=edit_compr_type_menu)
            #edit_menu.add_cascade(label="Array data type", menu=edit_array_data_type)
            #edit_menu.add_cascade(label="DCT block size", menu=edit_dct_block_size)

            debug_menu.add_command(label="image_data", command=self.Gui.Debug.get_image_data)
            debug_menu.add_command(label="show_image_now", command=self.Gui.Debug.show_image)
            debug_menu.add_command(label="show_image_compress_off", command=self.Gui.Debug.show_image_virgin)
            debug_menu.add_command(label="show_original_image", command=self.Gui.Debug.show_original)
            debug_menu.add_checkbutton(label="export_logs", command=lambda: self.Gui.Debug.export_logs())

            edit_array_data_type_var = tk.IntVar(value=0)
            edit_array_data_type.add_radiobutton(label="uint8", command=lambda: self.Gui.On_triggers.on_edit_array_data_type(), variable=edit_array_data_type_var, value=0)
            edit_array_data_type.add_separator()
            edit_array_data_type.add_radiobutton(label="int16", command=lambda: self.Gui.On_triggers.on_edit_array_data_type(), variable=edit_array_data_type_var, value=1)
            edit_array_data_type.add_radiobutton(label="int32", command=lambda: self.Gui.On_triggers.on_edit_array_data_type(), variable=edit_array_data_type_var, value=2)
            edit_array_data_type.add_separator()
            edit_array_data_type.add_radiobutton(label="float16", command=lambda: self.Gui.On_triggers.on_edit_array_data_type(), variable=edit_array_data_type_var, value=3)
            edit_array_data_type.add_radiobutton(label="float32", command=lambda: self.Gui.On_triggers.on_edit_array_data_type(), variable=edit_array_data_type_var, value=4)

            edit_dct_block_size_var = tk.IntVar(value=8)
            edit_dct_block_size.add_radiobutton(label="2", command=lambda: self.Gui.On_triggers.on_edit_dct_block_size(), variable=edit_dct_block_size_var, value=2)
            edit_dct_block_size.add_radiobutton(label="4", command=lambda: self.Gui.On_triggers.on_edit_dct_block_size(), variable=edit_dct_block_size_var, value=4)
            edit_dct_block_size.add_radiobutton(label="8", command=lambda: self.Gui.On_triggers.on_edit_dct_block_size(), variable=edit_dct_block_size_var, value=8)
            edit_dct_block_size.add_radiobutton(label="16", command=lambda: self.Gui.On_triggers.on_edit_dct_block_size(), variable=edit_dct_block_size_var, value=16)


            menu.add_cascade(label="File", menu=file_menu)
            menu.add_cascade(label="Edit", menu=edit_menu)
            menu.add_cascade(label="Debug", menu=debug_menu)
            menu.configure(activeborderwidth=5)


            root.config(menu=menu)


    class Get_windows:

        @staticmethod
        def get_folder(filetypes, defaultextension) -> str:
            '''
            creates window for choosing folder
            :param filetypes: filetypes
            :param defaultextension: default file-type
            :return: path
            '''
            initialfile = ''.join(path.split("/")[-1:])

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
            global file_image

            print()
            # re-creating image with new settings
            pixel_matrix = work_with_gppic.extract_pixels_from_png(path)
            file_image = work_with_gppic.convert_to_gppic(pixel_matrix)
            file = work_with_gppic.open_image(file_image)
            image_label.destroy()

            size_looker_label.destroy()
            image_viewer_frame.destroy()

            self.Gui.Create_widgets.create_image_viewer(file)
            self.Gui.Create_widgets.create_size_looker_label()


        # edits DST CUMpression value in Work_with_gppic class
        @staticmethod
        def on_dct_slider_compression(value) -> None:
            global work_with_gppic
            work_with_gppic = Work_with_gppic(int(value),
                                              work_with_gppic.compression_quant_force,
                                              work_with_gppic.compression_type,
                                              work_with_gppic.array_data_type,
                                              work_with_gppic.dct_blocksize
                                              )

        # edits quantization CUMpression value in Work_with_gppic class
        @staticmethod
        def on_quantization_slider_compression(value) -> None:
            global work_with_gppic
            work_with_gppic = Work_with_gppic(work_with_gppic.compression_dct_force,
                                              int(value), work_with_gppic.compression_type,
                                              work_with_gppic.array_data_type,
                                              work_with_gppic.dct_blocksize
                                              )


        @staticmethod
        def on_edit_compression_type() -> None:
            global work_with_gppic
            work_with_gppic = Work_with_gppic(
                work_with_gppic.compression_dct_force,
                work_with_gppic.compression_quant_force,
                edit_compr_type_menu_var.get(),
                work_with_gppic.array_data_type,
                work_with_gppic.dct_blocksize

            )
            logger.debug(f"Compression type has been edited to {edit_compr_type_menu_var.get()}")

        @staticmethod
        def on_edit_array_data_type() -> None:
            global work_with_gppic
            work_with_gppic = Work_with_gppic(
                work_with_gppic.compression_dct_force,
                work_with_gppic.compression_quant_force,
                work_with_gppic.compression_type,
                edit_array_data_type_var.get(),
                work_with_gppic.dct_blocksize

            )
            logger.debug(f"Array data type has been edited to {edit_array_data_type_var.get()}")

        @staticmethod
        def on_edit_dct_block_size() -> None:
            global work_with_gppic
            work_with_gppic = Work_with_gppic(
                work_with_gppic.compression_dct_force,
                work_with_gppic.compression_quant_force,
                work_with_gppic.compression_type,
                work_with_gppic.array_data_type,
                edit_dct_block_size_var.get()

            )
            logger.debug(f"DCT block size has been edited to {edit_dct_block_size_var.get()}")


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
                                                     f"\nRMSE : {round(RMSE, 6)}"
                                                     f"\ndct_blocksize : {dct_blocksize}"
                                                     f"\n\nuncompressed (by DCT & quantization) bytes size:  {size_img_uncompress_DCT}"
                                                     f"\nuncompressed (by Lzma) bytes size:  {size_img_uncompress}"f""
                                                     f"\nnow bytes size : {size_img} ± 5Kb")

        @staticmethod
        def show_image():
            file_image.seek(0)
            img = work_with_gppic.open_image(file_image)
            img.show()

        @staticmethod
        def show_image_virgin():
            '''
            shows image without compression
            '''
            work_with_gppic_virgin = Work_with_gppic()
            pixel_matrix_virgin = work_with_gppic_virgin.extract_pixels_from_png(path)
            file_image_virgin = work_with_gppic_virgin.convert_to_gppic(pixel_matrix_virgin)
            file_image_virgin.seek(0)
            img = work_with_gppic_virgin.open_image(file_image_virgin)
            img.show()

        @staticmethod
        def show_original():
            os.startfile(path)

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



    def export_file_as_png(self, img) -> None:
        '''
        exports image as .png
        :param img: image data
        '''
        export_path = self.Get_windows.get_folder([("png", "*.png"), ("All files", "*.*")], ".png")
        if export_path == "":
            pass
        else:
            img.seek(0)
            img = work_with_gppic.open_image(img)
            img.save(export_path, format="PNG")


            logger.info("image has been exported as PNG!")
            showinfo("Exported!", "Image was successfully exported as PNG!")


    def export_file(self, img) -> None:
        '''
        exports image in .gppic format
        :param img: image data
        '''
        export_path = self.Get_windows.get_folder([("gpic", "*.gpic"), ("All files", "*.*")], ".gppic")
        if export_path == "":
            pass
        else:
            with open(export_path, "wb") as f:
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
    global work_with_gppic
    global path
    global file_image
    global buffer_handler, log_buffer

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

    gui = Gui()
    gui.create_window()

    work_with_gppic = Work_with_gppic()

    path = gui.Get_windows.get_path([("Images", "*.png;*.jpg"), ("Txt files", "*.txt"), ("All files", "*.*")])


    if path == "":
        sys.exit()

    pixel_matrix = work_with_gppic.extract_pixels_from_png(path)
    file_image = work_with_gppic.convert_to_gppic(pixel_matrix)
    gui.Create_widgets.create_main_widgets(work_with_gppic.open_image(file_image))

    logger.info("DONE")
    root.mainloop()


if __name__ == "__main__":
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    #logging.disable(logging.CRITICAL)
    main()
