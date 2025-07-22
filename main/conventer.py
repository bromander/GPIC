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

    def __init__(self, compression_dct_force, compression_quantization_force, compression_type):
        self.compression_dct_force = compression_dct_force
        self.compression_quantization_force = compression_quantization_force
        self.compression_type = compression_type

    #returns list with all pixels from png file. Example: [(0, 0, 0), (49, 35, 0), (42, 42, 8), (37, 40, 9)]
    @staticmethod
    @loading_screen
    def extract_pixels_from_png(path) -> list:
        if path == None:
            raise ValueError("'path' not found")
        else:
            with Image.open(path) as img:

                img = img.convert("RGB")

                width, height = img.size
                logging.info(f"Picture size: {width}x{height}")

                logging.debug("Getting pixel data...")
                pixels = list(img.getdata())

                logging.debug("Loading pixel matrix...")
                pixel_matrix = [
                    pixels[i * width:(i + 1) * width] for i in range(height)
                ]
                return pixel_matrix

    @staticmethod
    def dct2(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def blockify(arr, block_size=8):
        logging.debug("blocking array...")
        h, w = arr.shape
        pad_h = (-h) % block_size
        pad_w = (-w) % block_size
        arr_padded = numpy.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant')
        H, W = arr_padded.shape
        return (arr_padded.reshape(H // block_size, block_size, W // block_size, block_size).swapaxes(1, 2)), (h, w)

    #converts list with png pixels to .gppic file in ram
    @loading_screen
    def convert_to_Gppic(self, pixel_matrix) -> io.BytesIO:
        global size_img
        global size_img_uncompress

        '''
                                logging.debug("unpacking main data...")
                        bytes_len = struct.unpack('>I', img[index + 1:index + 5])[0]
                        index += 5
                        logging.debug("decompressing main data...")
                        decompressed = zlib.decompress(img[index:index + bytes_len])

                        logging.debug("creating array...")
                        dct_data = numpy.frombuffer(decompressed, dtype=numpy.float32)
                        dct_data = dct_data.reshape(size[1], size[0])

                        blocks, _ = self.blockify(dct_data, 8)
                        logging.debug("unusing DCT method...")
                        for i in range(blocks.shape[0]):
                            for j in range(blocks.shape[1]):
                                blocks[i, j] = self.idct2(blocks[i, j])

                        restored = self.unblockify(blocks, size, 8)

                        pixels_gray = numpy.clip(numpy.rint(restored), 0, 255).astype(numpy.uint8)

                        logging.debug("Creating image...")
                        pixels = numpy.stack([pixels_gray] * 3, axis=-1)

                        index += bytes_len

                '''

        if self.compression_dct_force not in range(0, 256):
            raise ValueError(f"invalid value of compression_dct_force: {self.compression_dct_force}")
        elif self.compression_quantization_force not in range(0, 256):
            raise ValueError(f"invalid value of compression_dct_force: {self.compression_quantization_force}")
        elif int(self.compression_type) not in range(0, 3):
            raise ValueError(f"invalid value of compression type: {self.compression_type}")

        logging.debug("writing main data...")

        sizeX, sizeY = len(pixel_matrix[0]), len(pixel_matrix)
        img = io.BytesIO()
        img.write(b"\x89")
        img.write(ToBytes.to_bytes_str("GPC\n"))
        img.write(ToBytes.to_bytes_str("O"))
        img.write(ToBytes.to_bytes_int(sizeX, 4))
        img.write(ToBytes.to_bytes_int(sizeY, 4))
        img.write(ToBytes.to_bytes_str("A"))

        logging.debug("creating array...")

        pixels_array = numpy.array(pixel_matrix)
        gray_pixels = (0.299 * pixels_array[..., 0] +
                       0.587 * pixels_array[..., 1] +
                       0.114 * pixels_array[..., 2]).astype(numpy.float32)

        blocks, orig_shape = self.blockify(gray_pixels, 8)

        logging.debug("using DCT method...")

        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                blocks[i, j] = self.dct2(blocks[i, j])

        dct_data = (blocks.swapaxes(1, 2).reshape(blocks.shape[0] * 8, blocks.shape[1] * 8))[:orig_shape[0], :orig_shape[1]]

        logging.debug("compressing by quantization DCT...")
        if self.compression_dct_force != 1:
            if int(self.compression_type) == 1:
                dct_data[dct_data != 0] = numpy.round(dct_data[dct_data != 0] / self.compression_dct_force) * self.compression_dct_force
            elif int(self.compression_type) == 2:
                dct_data[dct_data != 0] = numpy.ceil(dct_data[dct_data != 0] / self.compression_dct_force) * self.compression_dct_force
            elif int(self.compression_type) == 0:
                dct_data[dct_data != 0] = numpy.floor(dct_data[dct_data != 0] / self.compression_dct_force) * self.compression_dct_force


        gray_pixels_b = dct_data.astype(numpy.float32).tobytes()
        size_img_uncompress = len(gray_pixels_b)

        dct_data = numpy.frombuffer(gray_pixels_b, dtype=numpy.float32)
        dct_data = dct_data.reshape(sizeX, sizeY)

        blocks, _ = self.blockify(dct_data, 8)
        logging.debug("unusing DCT method...")
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                blocks[i, j] = self.idct2(blocks[i, j])

        restored = self.unblockify(blocks, [sizeX, sizeY], 8)

        gray_pixels_b = numpy.clip(numpy.rint(restored), 0, 255).astype(numpy.uint8)

        logging.debug("compressing by quantization after DCT...")
        if self.compression_quantization_force != 1:
            if int(self.compression_type) == 1:
                gray_pixels_b[gray_pixels_b != 0] = numpy.round(
                    gray_pixels_b[
                        gray_pixels_b != 0] / self.compression_quantization_force) * self.compression_quantization_force
            elif int(self.compression_type) == 2:
                gray_pixels_b[gray_pixels_b != 0] = numpy.ceil(
                    gray_pixels_b[
                        gray_pixels_b != 0] / self.compression_quantization_force) * self.compression_quantization_force
            elif int(self.compression_type) == 0:
                gray_pixels_b[gray_pixels_b != 0] = numpy.floor(
                    gray_pixels_b[
                        gray_pixels_b != 0] / self.compression_quantization_force) * self.compression_quantization_force


        logging.debug("compressing by deflate...")

        compressed = zlib.compress(gray_pixels_b)
        size_img = len(compressed)

        img.write(ToBytes.to_bytes_int(size_img, 4))
        img.write(compressed)

        img.write(ToBytes.to_bytes_str("T"))
        img.write(ToBytes.to_bytes_str("IMG"))
        img.write(ToBytes.to_bytes_str("E"))

        img.seek(0)
        return img

    @staticmethod
    def idct2(block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def unblockify(blocks, orig_shape, block_size=8):
        logging.debug("unblocking array...")
        h, w = orig_shape
        n_v, n_h, _, _ = blocks.shape
        full = (blocks.swapaxes(1, 2)
                .reshape(n_v * block_size, n_h * block_size))
        return full[:h, :w]

    #opens .gppic file and returns it like png image
    @loading_screen
    def open_image(self, img) -> Image.Image:
        pixels = []
        size = []
        logging.debug("reading image...")
        img = img.read()
        index = 5

        while True:
            now_elem = img[index:index + 1]

            if now_elem == b"O":
                size = [struct.unpack('>I', img[index + 1:index + 5])[0],
                        struct.unpack('>I', img[index + 5:index + 9])[0]]
                index += 8

            if now_elem == b"A":
                bytes_len = struct.unpack('>I', img[index + 1:index + 5])[0]  # gets len of bytes(pixels)
                index += 5

                decompressed = zlib.decompress(img[index:index + bytes_len])  # decompressing pixels data by Deflate
                img = img.replace(img[index:index + bytes_len],
                                  decompressed)  # replasing compressed data to decompressed data

                pixels = numpy.zeros((size[1], size[0], 3), dtype=numpy.uint8)

                data = img[index + 1:index + 1 + size[0] * size[1]]

                pixels[:, :, 0] = numpy.frombuffer(data, dtype=numpy.uint8).reshape(size[1], size[0])

                pixels[:, :, 1] = pixels[:, :, 0]
                pixels[:, :, 2] = pixels[:, :, 0]
                pixels[pixels > 255] = 255
                pixels[pixels < 0] = 0

                index += size[0] * size[1]

            if now_elem == b"E":
                break

            index += 1

        original_image = Image.fromarray(pixels)
        return ImageOps.exif_transpose(original_image)



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
        root.title("gppic conventer")
        root.geometry("1000x800")


        #creating buttons
        btn_view = tk.Button(text="Update Image", command=self.On_triggers.on_button_view_update_image)
        btn_view.pack(anchor="nw")
        btn_view.place(x=15, y=15)


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
                from_=10,
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

            file_menu.add_command(label="New")
            file_menu.add_cascade(label="Export", menu=file_save_as_menu)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=exit)

            edit_menu.add_cascade(label="Compression type", menu=edit_compr_type_menu)

            debug_menu.add_command(label="image_data", command=self.Gui.Debug.get_image_data)
            debug_menu.add_command(label="show_image", command=self.Gui.Debug.show_image)
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
            file_image = work_with_gppic.convert_to_Gppic(pixel_matrix)
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
            work_with_gppic = Work_with_gppic(int(value), work_with_gppic.compression_quantization_force, work_with_gppic.compression_type)

        # edits quantization CUMpression value in Work_with_gppic class
        @staticmethod
        def on_quantization_slider_compression(value) -> None:
            global work_with_gppic
            work_with_gppic = Work_with_gppic(work_with_gppic.compression_dct_force, int(value), work_with_gppic.compression_type)


        @staticmethod
        def on_edit_compression_type() -> None:
            global work_with_gppic
            work_with_gppic = Work_with_gppic(work_with_gppic.compression_dct_force, work_with_gppic.compression_quantization_force, edit_compr_type_menu_var.get())
            logging.debug(f"Compression type has been edited for {edit_compr_type_menu_var.get()}")


    class Debug:

        def __init__(self, gui_instance):
            self.Gui = gui_instance

        @staticmethod
        def get_image_data():
            dct_compression_forse_data  = work_with_gppic.compression_dct_force
            quantization_compression_forse_data = work_with_gppic.compression_quantization_force
            compression_type_data = work_with_gppic.compression_type

            showinfo(title="get_image_data", message=f"dct_compression_forse : {dct_compression_forse_data}"
                                                     f"\nquantization_compression_forse_data : {quantization_compression_forse_data}"
                                                     f"\ncompression_type : {compression_type_data}\n"
                                                     f"\nbytes size : {size_img}"
                                                     f"\nuncompressed bytes size:  {size_img_uncompress}")

        @staticmethod
        def show_image():
            file_image.seek(0)
            img = work_with_gppic.open_image(file_image)
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

    work_with_gppic = Work_with_gppic(14, 1, 1) #default CUMpression force - 1, default CUMpression type = 1
    gui = Gui()

    gui.create_window()

    path = gui.Get_windows.get_path([("Изображения", "*.png;*.jpg"), ("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")])


    if path == "":
        exit()

    pixel_matrix = work_with_gppic.extract_pixels_from_png(path)
    file_image = work_with_gppic.convert_to_Gppic(pixel_matrix)

    gui.Create_widgets.create_main_widgets(work_with_gppic.open_image(file_image))

    logging.info("DONE")
    root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(funcName)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    main()