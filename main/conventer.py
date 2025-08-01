from PIL import Image, ImageTk, ImageOps
import numpy
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror, showwarning, showinfo
import zlib
import io
import logging
import struct
import os
from concurrent.futures import ThreadPoolExecutor
import functools
from scipy.fftpack import dct, idct
from typing import Optional, List, Tuple, BinaryIO


# pigar generate - create requirements.txt

#handler that creates small loading window
def loading_screen(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        #if user tries to close loading window instead of close will be called this function
        def disable_close():
            pass

        loading_win = tk.Toplevel(root)
        loading_win.title("Loading...")
        loading_win.transient(root)
        loading_win.grab_set() #blocks user interaction with other windows of the program
        loading_win.protocol("WM_DELETE_WINDOW", disable_close) #if user tries to close loading window instead of close will be called this function

        loading_win.update_idletasks()

        w = loading_win.winfo_reqwidth() #gets loading window width
        h = loading_win.winfo_reqheight() #gets loading window height
        sw = loading_win.winfo_screenwidth() #gets screen width
        sh = loading_win.winfo_screenheight() #gets screen height
        x = (sw - w) // 2 #getting x and y centers
        y = (sh - h) // 2
        loading_win.geometry(f"{200}x{100}+{x}+{y}") #plasing window on screen

        #setting window data:
        label_loading = tk.Label(loading_win, text="Loading...")
        label_loading.pack(anchor="n", pady=10)
        pb = ttk.Progressbar(loading_win, mode="indeterminate") #mode "determinate" - progress just go to left/right. Mode "indeterminate" - progress will do ping - pong XD
        pb.pack(expand=True, fill="x", padx=20, pady=20)
        pb.start(2) #makes the progressbar always move

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)

            def check():
                if future.done():
                    loading_win.destroy() #destoys loading window after compliting the function
                else:
                    loading_win.after(50, check) #checks every 20ms "Is function complete?" (I think it is a little bit not optimized at all but the main thing is that it works)

            loading_win.after(50, check)
            root.wait_window(loading_win)
            return future.result()

    return wrapper


#Class for working with gppic/other images files
class Work_with_gppic:

    def __init__(self, compression_dct_force: int = 1, compression_quant_force: int = 1, compression_type: int = 0):
        self.compression_dct_force = compression_dct_force
        self.compression_quant_force = compression_quant_force
        self.compression_type = compression_type  # 0=floor,1=round,2=ceil

    @staticmethod
    def extract_pixels_from_png(path: str) -> Optional[List[List[Tuple[int, int, int]]]]:
        if path is None:
            raise ValueError("'path' not found")
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                width, height = img.size
                logging.info(f"Picture size: {width}×{height}")

                logging.debug("Getting pixel data…")
                pixels = list(img.getdata())

                logging.debug("Loading pixel matrix…")
                pixel_matrix: List[List[Tuple[int, int, int]]] = [
                    pixels[i * width:(i + 1) * width]
                    for i in range(height)
                ]
                return pixel_matrix
        except FileNotFoundError:
            logging.error(f"File not found: {path}")
            return None

    #returns list with all pixels from png file. Example: [(0, 0, 0), (49, 35, 0), (42, 42, 8), (37, 40, 9)]
    @staticmethod
    def _dct2(block: numpy.ndarray) -> numpy.ndarray:
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def _idct2(block: numpy.ndarray) -> numpy.ndarray:
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def _blockify(arr: numpy.ndarray, block_size: int = 8) -> Tuple[numpy.ndarray, Tuple[int, int]]:
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
    def _unblockify(blocks: numpy.ndarray, orig_shape: Tuple[int, int], block_size: int = 8) -> numpy.ndarray:
        n_v, n_h, _, _ = blocks.shape
        padded = blocks.swapaxes(1, 2).reshape(n_v * block_size, n_h * block_size)
        h, w = orig_shape
        return padded[:h, :w]

    @staticmethod
    def _quantize(data: numpy.ndarray, force: int, method: int) -> numpy.ndarray:
        if force != 1:
            if int(method) == 1:
                data[data != 0] = numpy.round(
                    data[
                        data != 0] / force) * force
            elif int(method) == 2:
                data[data != 0] = numpy.ceil(
                    data[
                        data != 0] / force) * force
            elif int(method) == 0:
                data[data != 0] = numpy.floor(
                    data[
                        data != 0] / force) * force
        return data

    #converts list with png pixels to .gppic file in ram
    @loading_screen
    def convert_to_gppic(self, pixel_matrix: list) -> io.BytesIO:
        global size_img, size_img_uncompress, size_img_uncompress_DCT
        # Validate parameters
        for name, val in [('DCT force', self.compression_dct_force),
                          ('quant force', self.compression_quant_force)]:
            if not (0 <= val < 256):
                raise ValueError(f"Invalid {name}: {val}")
        if self.compression_type not in {0, 1, 2}:
            raise ValueError(f"Invalid compression type: {self.compression_type}")

        height = len(pixel_matrix)
        width = len(pixel_matrix[0]) if height else 0
        buf = io.BytesIO()

        # Header: signature + dimensions
        buf.write(b"\x89GPC\nO")
        buf.write(struct.pack('>I', width))
        buf.write(struct.pack('>I', height))
        buf.write(b"A")

        # Convert to grayscale
        pixels = numpy.array(pixel_matrix, dtype=numpy.float32)
        gray = (0.299 * pixels[..., 0] +
                0.587 * pixels[..., 1] +
                0.114 * pixels[..., 2]).astype(numpy.float32)

        size_img_uncompress_DCT = gray.nbytes

        # DCT compression
        blocks, orig = self._blockify(gray)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                blocks[i, j] = self._dct2(blocks[i, j])
        dct_mat = blocks.swapaxes(1, 2).reshape(blocks.shape[0] * 8, blocks.shape[1] * 8)
        dct_mat = dct_mat[:orig[0], :orig[1]]
        dct_mat = self._quantize(dct_mat, self.compression_dct_force, self.compression_type)

        raw = dct_mat.astype(numpy.float32).tobytes()

        # Inverse DCT reconstruction
        inv = numpy.frombuffer(raw, dtype=numpy.float32).reshape(orig[0], orig[1])
        blocks2, _ = self._blockify(inv)
        for i in range(blocks2.shape[0]):
            for j in range(blocks2.shape[1]):
                blocks2[i, j] = self._idct2(blocks2[i, j])
        restored = self._unblockify(blocks2, orig)
        restored = numpy.clip(numpy.rint(restored), 0, 255).astype(numpy.uint8)

        # Quantization post-inverse
        restored = self._quantize(restored, self.compression_quant_force, self.compression_type).astype(numpy.uint8)

        size_img_uncompress = len(restored.tobytes())

        # Deflate compression
        compressed = zlib.compress(restored.tobytes())
        size_img = len(compressed)

        # Write sizes and data
        buf.write(struct.pack('>I', size_img))
        buf.write(compressed)
        buf.write(b"TIMAGE")
        buf.seek(0)
        return buf

    @staticmethod
    def idct2(block: numpy.ndarray) -> numpy.ndarray:
        if block.ndim != 2 or block.shape[0] != block.shape[1]:
            raise ValueError(f"idct2 ожидает квадратный блок, получили shape={block.shape}")
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def unblockify(blocks: numpy.ndarray,
                   orig_shape: Tuple[int, int],
                   block_size: int = 8) -> numpy.ndarray:
        logging.debug("unblocking array...")
        h, w = orig_shape
        n_v, n_h, b_h, b_w = blocks.shape

        if b_h != block_size or b_w != block_size:
            raise ValueError(f"Размер блоков {b_h}x{b_w} не соответствует block_size={block_size}")
        full = (
            blocks
            .swapaxes(1, 2)
            .reshape(n_v * block_size, n_h * block_size)
        )
        return full[:h, :w]

    def open_image(self, file_obj: BinaryIO) -> Image.Image:
        global gui
        data = file_obj.read()
        offset = 5  # пропускаем первые 5 байт
        width = height = None
        pixels = None

        logging.debug("reading image...")

        while offset < len(data):
            chunk_type = data[offset:offset + 1]
            offset += 1

            if chunk_type == b'O':
                # читаем два uint32 BE
                if offset + 8 > len(data):
                    raise ValueError("Недостаточно данных для размера изображения")
                width, height = struct.unpack('>II', data[offset:offset + 8])
                offset += 8

            elif chunk_type == b'A':
                if width is None or height is None:
                    raise ValueError("Размеры изображения не заданы перед данными пикселей")
                if offset + 4 > len(data):
                    raise ValueError("Недостаточно данных для длины сжатого блока")
                (comp_len,) = struct.unpack('>I', data[offset:offset + 4])
                offset += 4
                if offset + comp_len > len(data):
                    raise ValueError("Недостаточно данных для сжатого блока")
                comp_data = data[offset:offset + comp_len]
                offset += comp_len

                # декомпрессия
                raw = zlib.decompress(comp_data)
                expected = width * height
                if len(raw) < expected:
                    raise ValueError(f"Недостаточно пиксельных данных: ожидается {expected}, получено {len(raw)}")

                # строим 2D-массив и дублируем в RGB
                arr = numpy.frombuffer(raw[:expected], dtype=numpy.uint8).reshape((height, width))
                pixels = numpy.stack([arr, arr, arr], axis=-1)

            elif chunk_type == b"T":
                title = data[offset:offset + 4]
                offset += 4
                root.title(title.decode('utf-8'))

            elif chunk_type == b'E':
                break

            else:
                # пропуск неизвестного чанка: читаем длину и смещаем
                if offset + 4 > len(data):
                    raise ValueError(f"Некорректный чанк {chunk_type!r}: нет длины")
                (length,) = struct.unpack('>I', data[offset:offset + 4])
                offset += 4
                logging.debug(f"Skipping unknown chunk {chunk_type!r} of length {length}")
                offset += length

        if pixels is None:
            raise ValueError("Не найден блок пикселей (чанк 'A')")

        img = Image.fromarray(pixels, mode='RGB')
        return ImageOps.exif_transpose(img)


#Class for encoding int/str to a byte object
class ToBytes:
    #encodes int to a byte object
    @staticmethod
    def to_bytes_int(number, length) -> bytes:
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
        global root

        root = tk.Tk()
        root.title("loading...")
        root.geometry("1000x800")


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
            self.create_image_viewer(image)
            self.create_dct_compression_slider()
            self.create_quantization_compression_slider()
            self.create_size_looker_label()

        # creates text label with data of image sizes
        def create_size_looker_label(self) -> None:
            global size_looker_label

            size_looker_label = tk.Label(root,
                                         text=f"Original image size: {self.Gui.format_size(os.path.getsize(path))}\n"
                                              f"Now file size: {self.Gui.format_size(size_img)}",
                                         font=("Arial", 10))
            size_looker_label.pack(anchor="sw")

        #create slider for DCT compression
        def create_dct_compression_slider(self) -> None:
            compression_frame = tk.Frame(root, bg="#FFADB0", bd=5, relief=tk.GROOVE)
            compression_frame.pack(anchor="nw", fill=tk.NONE, expand=False)
            compression_frame.place(x=10, y=60, width=110, height=320)

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
                length=300,
                command=self.Gui.On_triggers.on_dct_slider_compression,
            )

            slider_compression.pack()
            slider_compression.place(height=250, y=45, x=25)

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
            image_viewer_frame.configure(width=550, height=550)
            image_viewer_frame.pack_propagate(False)

            image.thumbnail((550, 550), Image.Resampling.LANCZOS)

            image_ = ImageTk.PhotoImage(image)

            #creating label with image
            image_label = tk.Label(image_viewer_frame, image=image_, bg="white")
            image_label.image = image_
            image_label.pack(anchor="center", ipady=200)

        def create_menu(self) -> None:
            global edit_compr_type_menu_var
            menu = tk.Menu(root)
            root.option_add("*tearOff", tk.FALSE)

            file_menu = tk.Menu()
            file_save_as_menu = tk.Menu()
            edit_menu = tk.Menu()
            debug_menu = tk.Menu()
            edit_compr_type_menu = tk.Menu()

            file_save_as_menu.add_command(label="Export as .GPPIC", command=lambda: self.Gui.export_file(file_image))
            file_save_as_menu.add_command(label="Export as .PNG", command=lambda: self.Gui.export_file_as_png(file_image))

            edit_compr_type_menu_var = tk.IntVar(value=1)
            edit_compr_type_menu.add_radiobutton(label="Lighter", command=lambda: self.Gui.On_triggers.on_edit_compression_type(), variable=edit_compr_type_menu_var, value=2)
            edit_compr_type_menu.add_radiobutton(label="Grayer", command=lambda: self.Gui.On_triggers.on_edit_compression_type(), variable=edit_compr_type_menu_var, value=1)
            edit_compr_type_menu.add_radiobutton(label="Darker", command=lambda: self.Gui.On_triggers.on_edit_compression_type(), variable=edit_compr_type_menu_var, value=0)

            file_menu.add_command(label="New", command=Gui.re_create_window)
            file_menu.add_cascade(label="Export", menu=file_save_as_menu)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=exit)

            edit_menu.add_cascade(label="Compression type", menu=edit_compr_type_menu)

            debug_menu.add_command(label="image_data", command=self.Gui.Debug.get_image_data)
            debug_menu.add_command(label="show_image", command=self.Gui.Debug.show_image)
            debug_menu.add_command(label="show_image_compress_off", command=self.Gui.Debug.show_image_virgin)
            debug_menu.add_command(label="show_original_image", command=self.Gui.Debug.show_original)

            menu.add_cascade(label="File", menu=file_menu)
            menu.add_cascade(label="Edit", menu=edit_menu)
            menu.add_cascade(label="Debug", menu=debug_menu)
            menu.configure(activeborderwidth=5)


            root.config(menu=menu)


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
            work_with_gppic = Work_with_gppic(int(value), work_with_gppic.compression_quant_force, work_with_gppic.compression_type)

        # edits quantization CUMpression value in Work_with_gppic class
        @staticmethod
        def on_quantization_slider_compression(value) -> None:
            global work_with_gppic
            work_with_gppic = Work_with_gppic(work_with_gppic.compression_dct_force, int(value), work_with_gppic.compression_type)


        @staticmethod
        def on_edit_compression_type() -> None:
            global work_with_gppic
            work_with_gppic = Work_with_gppic(work_with_gppic.compression_dct_force, work_with_gppic.compression_quant_force, edit_compr_type_menu_var.get())
            logging.debug(f"Compression type has been edited for {edit_compr_type_menu_var.get()}")


    class Debug:

        def __init__(self, gui_instance):
            self.Gui = gui_instance

        @staticmethod
        def get_image_data():
            dct_compression_forse_data  = work_with_gppic.compression_dct_force
            quantization_compression_forse_data = work_with_gppic.compression_quant_force
            compression_type_data = work_with_gppic.compression_type

            showinfo(title="get_image_data", message=f"dct_compression_forse : {dct_compression_forse_data}"
                                                     f"\nquantization_compression_forse_data : {quantization_compression_forse_data}"
                                                     f"\ncompression_type : {compression_type_data}\n"
                                                     f"\nnow bytes size : {size_img}"
                                                     f"\nuncompressed (by DCT & quantization) bytes size:  {size_img_uncompress_DCT}"
                                                     f"\nuncompressed (by Deflate) bytes size:  {size_img_uncompress}")

        @staticmethod
        def show_image():
            file_image.seek(0)
            img = work_with_gppic.open_image(file_image)
            img.show()

        @staticmethod
        def show_image_virgin():
            work_with_gppic_virgin = Work_with_gppic()
            pixel_matrix_virgin = work_with_gppic_virgin.extract_pixels_from_png(path)
            file_image_virgin = work_with_gppic_virgin.convert_to_gppic(pixel_matrix_virgin)
            file_image_virgin.seek(0)
            img = work_with_gppic_virgin.open_image(file_image_virgin)
            img.show()

        @staticmethod
        def show_original():
            os.startfile(path)




    #exports image as .png
    def export_file_as_png(self, img) -> None:

        export_path = self.Get_windows.get_folder([("png", "*.png"), ("All files", "*.*")], ".png")
        if export_path == "":
            pass
        else:
            img.seek(0)
            img = work_with_gppic.open_image(img)
            img.save(export_path, format="PNG")


            logging.info("image has been exported as PNG!")


    #exports image in .gppic format
    def export_file(self, img) -> None:
        export_path = self.Get_windows.get_folder([("gppic", "*.gppic"), ("All files", "*.*")], ".gppic")
        if export_path == "":
            pass
        else:
            with open(export_path, "wb") as f:
                f.write(img.getvalue())


            logging.info("file has been successfully exported!")


    #gets bytes and returns them in a beautiful wrapper for user
    @staticmethod
    def format_size(size) -> str:
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
def main():
    global work_with_gppic
    global path
    global file_image


    gui = Gui()
    gui.create_window()

    work_with_gppic = Work_with_gppic()

    path = gui.Get_windows.get_path([("Изображения", "*.png;*.jpg"), ("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")])


    if path == "":
        exit()

    pixel_matrix = work_with_gppic.extract_pixels_from_png(path)
    file_image = work_with_gppic.convert_to_gppic(pixel_matrix)
    gui.Create_widgets.create_main_widgets(work_with_gppic.open_image(file_image))

    logging.info("DONE")
    root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(funcName)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    main()