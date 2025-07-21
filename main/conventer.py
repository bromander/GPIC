from PIL import Image, ImageTk, ImageOps
import numpy
from tkinter import filedialog
import tkinter as tk
import zlib
import io
import logging
import struct
import os

# pigar generate - create requirements.txt



#Class for working with gppic/other images files
class Work_with_gppic:

    def __init__(self, compression_force):
        self.compression_force = compression_force

    #returns list with all pixels from png file. Example: [(0, 0, 0), (49, 35, 0), (42, 42, 8), (37, 40, 9)]
    @staticmethod
    def extract_pixels_from_png(path) -> list:
        if path == None:
            raise ValueError("Attribute 'path' not found")
        else:
            with Image.open(path) as img:

                img = img.convert("RGB")

                width, height = img.size
                logging.info(f"Picture size: {width}x{height}")

                logging.info("Getting pixel data...")
                pixels = list(img.getdata())

                logging.info("Loading pixel matrix...")
                pixel_matrix = [
                    pixels[i * width:(i + 1) * width] for i in range(height)
                ]
                return pixel_matrix

    #converts list with png pixels to .gppic file in ram
    def convert_to_Gppic(self, pixel_matrix) -> io.BytesIO:
        global size_img

        if self.compression_force not in range(0, 255):
            raise ValueError(f"invalid value: {self.compression_force}. Acceptable values: from 1 to 255 inclusive")
        else:
            logging.info("compression force: " + str(self.compression_force))

        sizeX = len(pixel_matrix[0])
        sizeY = len(pixel_matrix)

        #creates file
        img = io.BytesIO()
        img.seek(0)

        img.write(b"\x89")  # non readable ascii symbol
        img.write(ToBytes.to_bytes_str("GPC\n"))  # name

        img.write(ToBytes.to_bytes_str("O"))  # main chunk
        img.write(ToBytes.to_bytes_int(sizeX, 4))  # X size of image
        img.write(ToBytes.to_bytes_int(sizeY, 4))  # Y size of image

        img.write(ToBytes.to_bytes_str("A"))  # start of required chunks
        pixels_array = numpy.array(pixel_matrix)

        #creating list of numbers - values of pixel
        gray_pixels = (0.299 * pixels_array[..., 0] + 0.587 * pixels_array[..., 1] + 0.114 * pixels_array[
                ..., 2]).astype(numpy.uint8)

        #compressing
        if self.compression_force != 1:
            gray_pixels[gray_pixels > 10] = numpy.floor(gray_pixels[gray_pixels > 10] / self.compression_force) * self.compression_force


        compressed = zlib.compress(gray_pixels.tobytes())

        size_img = len(compressed)

        img.write(ToBytes.to_bytes_int(size_img, 4)) # writing len of bytes (pixel data)



        logging.info(f"bytes size: {size_img}")

        img.write(compressed)  # compressing pixels data by using the Deflate compression type and writing this to file


        img.write(ToBytes.to_bytes_str("T"))  # type of image
        img.write(ToBytes.to_bytes_str("IMG"))  # common image

        img.write(ToBytes.to_bytes_str("E"))  # End.
        img.seek(0)
        return img

    #opens .gppic file and returns it like png image
    def open_image(self, img) -> Image.Image:
        pixels = []
        size = []
        img = img.read()

        index = 5

        while True:

            now_elem = img[index:index + 1]

            if now_elem == b"O": # Getting main chunk data
                size = [struct.unpack('>I', img[index + 1:index + 5])[0],
                        struct.unpack('>I', img[index + 5:index + 9])[0]]
                index += 8

            if now_elem == b"A": # Getting required chunk data (pixel data)
                bytes_len = struct.unpack('>I', img[index + 1:index + 5])[0] #gets len of bytes(pixels)
                index += 5

                decompressed = zlib.decompress(img[index:index + bytes_len]) #decompressing pixels data by Deflate
                img = img.replace(img[index:index + bytes_len], decompressed) #replasing compressed data to decompressed data

                pixels = numpy.zeros((size[1], size[0], 3), dtype=numpy.uint8)

                data = img[index + 1:index + 1 + size[0] * size[1]]

                pixels[:, :, 0] = numpy.frombuffer(data, dtype=numpy.uint8).reshape(size[1], size[0])

                pixels[:, :, 1] = pixels[:, :, 0]
                pixels[:, :, 2] = pixels[:, :, 0]

                index += size[0] * size[1]

            if now_elem == b"E": # Breaking cycle on E chunk
                break

            index += 1

        original_image = Image.fromarray(pixels) #creating from list of pixels image in png format
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
            raise TypeError("'text' shood be str.")
        try:
            return text.encode('ascii')
        except UnicodeEncodeError:
            raise ValueError("String contains invalid ASCII characters..")


#class for creating and working with GUIs
class Gui:
    def __init__(self):
        self.Create_widgets = self.Create_widgets(self)
        self.On_triggers = self.On_triggers(self)

    #creates main root window
    def create_window(self, image) -> None:
        global root

        root = tk.Tk()
        root.title("gppic conventer")
        root.geometry("800x600")


        #creating buttons
        btn_view = tk.Button(text="Update Image", command=self.On_triggers.on_button_view_update_image)
        btn_view.pack(anchor="nw")
        btn_view.place(x=15, y=15)

        btn_view = tk.Button(text="Export Image", command=lambda: self.export_file(file_image))
        btn_view.pack(anchor="nw")
        btn_view.place(x=115, y=15)

        btn_view = tk.Button(text="Export Image .PNG", command=lambda: self.export_file_as_png(file_image))
        btn_view.pack(anchor="nw")
        btn_view.place(x=210, y=15)

        #btn_view = tk.Button(text="File", command=None)
        #btn_view.pack(anchor="nw")
        #btn_view.place(width=60, height=30)

        self.Create_widgets.create_main_widgets(image)

        root.mainloop()


    class Create_widgets:

        def __init__(self, gui_instance):
            self.Gui = gui_instance

        def create_main_widgets(self, image):
            self.create_image_viewer(image)
            self.create_compression_slider()
            self.create_size_looker_label()
            #self.create_up_sliders()

        # creates text label with data of image sizes
        def create_size_looker_label(self) -> None:
            global size_looker_label

            size_looker_label = tk.Label(root,
                                         text=f"Original image size: {self.Gui.format_size(os.path.getsize(path))}\n"
                                              f"Now file size: {self.Gui.format_size(size_img)}",
                                         font=("Arial", 10))
            size_looker_label.pack(anchor="sw")

        #create slider for compression
        def create_compression_slider(self) -> None:
            compression_frame = tk.Frame(root, bg="lightblue", bd=5, relief=tk.GROOVE)
            compression_frame.pack(anchor="nw", fill=tk.NONE, expand=False)
            compression_frame.place(x=10, y=80, width=110, height=320)

            slider_compression_label = tk.Label(compression_frame, text="compression", font=("Arial", 12),
                                                bg="lightblue")
            slider_compression_label.pack(anchor="center")
            slider_compression_label.place(y=5)

            slider_compression = tk.Scale(
                compression_frame,
                bg="lightblue",
                bd=3,
                from_=1,  # Минимальное значение
                to=120,  # Максимальное значение
                orient=tk.VERTICAL,  # Ориентация ползунка (HORIZONTAL или VERTICAL)
                length=300,  # Длина ползунка
                command=self.Gui.On_triggers.on_slider_compression  # Функция обратного вызова
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
            image_viewer_frame.configure(width=350, height=350)
            image_viewer_frame.pack_propagate(False)

            image.thumbnail((350, 350), Image.Resampling.LANCZOS)

            image_ = ImageTk.PhotoImage(image)

            #creating label with image
            image_label = tk.Label(image_viewer_frame, image=image_)
            image_label.image = image_
            image_label.pack(anchor="center")

        def create_up_sliders(self) -> None:
            file_frame = tk.Frame(root, bg="white", bd=0.5, relief=tk.SOLID)
            file_frame.pack(anchor="nw", fill=tk.NONE, expand=False)
            file_frame.place(x=10, y=80, width=110, height=320)


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


        # edits CUMpression value in Work_with_gppic class
        @staticmethod
        def on_slider_compression(value) -> None:
            global work_with_gppic
            work_with_gppic = Work_with_gppic(int(value))




    #edits CUMpression value in Work_with_gppic class
    @staticmethod
    def on_slider_compression(value) -> None:
        global work_with_gppic
        work_with_gppic = Work_with_gppic(int(value))


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

    work_with_gppic = Work_with_gppic(1) #default CUMpression force - 1
    gui = Gui()
    path = gui.Get_windows.get_path([("Изображения", "*.png;*.jpg"), ("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")])


    if path == "":
        exit()

    pixel_matrix = work_with_gppic.extract_pixels_from_png(path)
    file_image = work_with_gppic.convert_to_Gppic(pixel_matrix)

    gui.create_window(work_with_gppic.open_image(file_image))

    logging.info("DONE")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()