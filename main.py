import cv2
import numpy as np
from tkinter import filedialog, Button, Label, Tk
from PIL import Image, ImageTk
from skimage import filters, color, exposure, transform
import matplotlib.pyplot as plt


class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Processor App")
        self.master.geometry("1000x800")
        self.master.configure(bg="#f0f0f0")

        self.create_widgets()

    def create_widgets(self):
        button_font = ("Helvetica", 12, "bold")

        btn_open_image = Button(self.master, text="Open Image", command=self.open_img, bg="#4CAF50", fg="white",
                                font=button_font)
        btn_open_image.grid(column=0, row=0, columnspan=2, sticky='nsew', padx=10, pady=10)

        btn_sobel = Button(self.master, text="Sobel", command=lambda: self.apply_filter(filters.sobel), bg="#2196F3",
                           fg="white", font=button_font)
        btn_sobel.grid(column=0, row=1, sticky='nsew', padx=10, pady=5)

        btn_roberts = Button(self.master, text="Roberts", command=lambda: self.apply_filter(filters.roberts),
                             bg="#2196F3", fg="white", font=button_font)
        btn_roberts.grid(column=1, row=1, sticky='nsew', padx=10, pady=5)

        btn_gabor = Button(self.master, text="Gabor", command=self.apply_gabor, bg="#2196F3", fg="white",
                           font=button_font)
        btn_gabor.grid(column=0, row=2, sticky='nsew', padx=10, pady=5)

        btn_histogram = Button(self.master, text="Histogram Eq.", command=self.apply_histogram_eq, bg="#FF9800",
                               fg="white", font=button_font)
        btn_histogram.grid(column=1, row=2, sticky='nsew', padx=10, pady=5)

        btn_resize = Button(self.master, text="Resize", command=self.apply_resize, bg="#FF9800", fg="white",
                            font=button_font)
        btn_resize.grid(column=0, row=3, sticky='nsew', padx=10, pady=5)

        btn_rotate = Button(self.master, text="Rotate", command=self.apply_rotate, bg="#FF9800", fg="white",
                            font=button_font)
        btn_rotate.grid(column=1, row=3, sticky='nsew', padx=10, pady=5)

        btn_open_video = Button(self.master, text="Open Video", command=self.open_video, bg="#f44336", fg="white",
                                font=button_font)
        btn_open_video.grid(column=0, row=4, columnspan=2, sticky='nsew', padx=10, pady=10)

        self.panel = Label(self.master, bg="#dddddd")
        self.panel.grid(column=0, row=5, columnspan=2, rowspan=4, sticky='nsew', padx=10, pady=10)

        for i in range(2):
            self.master.grid_columnconfigure(i, weight=1)
        for i in range(10):
            self.master.grid_rowconfigure(i, weight=1)

    def open_img(self):
        filename = filedialog.askopenfilename(title='Open')
        if filename:
            img = Image.open(filename)
            img = img.resize((400, 250), Image.LANCZOS)
            self.image = np.asarray(img)
            imgtk = ImageTk.PhotoImage(img)
            self.panel.configure(image=imgtk)
            self.panel.image = imgtk

    def display(self, original, result):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(original)
        ax[0].set_title("Original Image")
        ax[1].imshow(result)
        ax[1].set_title("Processed Image")
        plt.show()

    def apply_filter(self, filter_func):
        gray = color.rgb2gray(self.image)
        filtered_image = filter_func(gray)
        self.display(self.image, filtered_image)

    def apply_gabor(self):
        gray = color.rgb2gray(self.image)
        filt_real, _ = filters.gabor(gray, frequency=1)
        self.display(self.image, filt_real)

    def apply_histogram_eq(self):
        eq_image = exposure.equalize_hist(self.image)
        self.display(self.image, eq_image)

    def apply_resize(self):
        resized_image = transform.resize(self.image, (self.image.shape[0] // 2, self.image.shape[1] // 2),
                                         anti_aliasing=True)
        self.display(self.image, resized_image)

    def apply_rotate(self):
        rotated_image = transform.rotate(self.image, 45)
        self.display(self.image, rotated_image)

    def open_video(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        vid = cv2.VideoCapture(0)
        while True:
            ret, frame = vid.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                center = (x + w // 2, y + h // 2)
                radius = int(round((w + h) * 0.25))
                cv2.circle(frame, center, radius, (255, 0, 0), 2)

            cv2.imshow("Video", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    app = ImageProcessorApp(root)
    root.mainloop()