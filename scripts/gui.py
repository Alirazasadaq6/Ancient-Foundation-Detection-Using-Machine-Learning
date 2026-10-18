import customtkinter as ctk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk
from model_unet import load_model as load_unet_model
from model_deeplab import load_model as load_deeplab_model
from scripts.detect import run_detection
import threading
import os
import tkinter as tk

# Try to import drag-and-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    messagebox.showerror("Import Error", "Please install tkinterdnd2 package:\npip install tkinterdnd2")
    raise

# Set light mode and color theme for the app
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

# Tooltip class – shows text when mouse hovers over a widget
class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.waittime = 500
        self.wraplength = 300
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    # Show the tooltip near the mouse cursor
    def showtip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         wraplength=self.wraplength)
        label.pack(ipadx=1)

    # Hide the tooltip
    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()

# Main GUI class
class AncientFoundationGUI:
    def __init__(self, root):
        # Set up window and screen size
        self.root = root
        self.root.title("Ancient Foundation Detector")

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Create layout frames
        padding_outer = 20
        padding_inner = 10
        half_width = (self.screen_width - 3 * padding_outer) // 2

        # Input image frame
        self.left_frame = ctk.CTkFrame(root, width=half_width, height=self.screen_height - 150)
        self.left_frame.grid(row=0, column=0, padx=(padding_outer, padding_outer//2), pady=padding_outer)

        # Output image frame
        self.right_frame = ctk.CTkFrame(root, width=half_width, height=self.screen_height - 150)
        self.right_frame.grid(row=0, column=1, padx=(padding_outer//2, padding_outer), pady=padding_outer)

        # Input image label
        self.input_label = ctk.CTkLabel(self.left_frame, text="Input Image (Drag & Drop Supported)",
                                        font=ctk.CTkFont(size=16, weight="bold"))
        self.input_label.pack(pady=(5, 0))

        # Input image area
        self.left_canvas = ctk.CTkLabel(self.left_frame, text="Drag & Drop Image Here",
                                        width=half_width - 2*padding_inner,
                                        height=self.screen_height - 200, fg_color="white", justify="center")
        self.left_canvas.pack(padx=padding_inner, pady=(5, padding_inner))

        self.left_canvas.drop_target_register(DND_FILES)
        self.left_canvas.dnd_bind('<<Drop>>', self.handle_drop)

        # Output image label
        self.output_label = ctk.CTkLabel(self.right_frame, text="Output Image",
                                        font=ctk.CTkFont(size=16, weight="bold"))
        self.output_label.pack(pady=(5, 0))

        # Output image area
        self.right_canvas = ctk.CTkLabel(self.right_frame, text="", width=half_width - 2*padding_inner,
                                         height=self.screen_height - 200, fg_color="white")
        self.right_canvas.pack(padx=padding_inner, pady=(5, padding_inner))

        # Button panel at the bottom
        self.buttons_frame = ctk.CTkFrame(root)
        self.buttons_frame.grid(row=1, column=0, columnspan=6, pady=(0, 30), sticky='n')

        # Responsive layout settings
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Model selection Radio button
        btn_width = 180
        btn_height = 50
        corner_radius = 15

        self.model_var = ctk.StringVar(value="unet")

        self.radio_unet = ctk.CTkRadioButton(self.buttons_frame, text="UNet", variable=self.model_var, value="unet")
        self.radio_unet.grid(row=0, column=0, padx=(10,5), pady=10, sticky='w')

        self.radio_deeplab = ctk.CTkRadioButton(self.buttons_frame, text="DeepLab", variable=self.model_var, value="deeplab")
        self.radio_deeplab.grid(row=0, column=1, padx=(5,15), pady=10, sticky='w')

        CreateToolTip(self.radio_unet, "UNet: Best for pixel-wise segmentation, good accuracy on small datasets.")
        CreateToolTip(self.radio_deeplab, "DeepLab: Uses atrous convolution, better at capturing context in large images.")

        # Upload image button
        self.upload_btn = ctk.CTkButton(self.buttons_frame, text="Upload Image", command=self.upload_image,
                                        width=btn_width, height=btn_height, corner_radius=corner_radius,
                                        fg_color="#007BFF", hover_color="#0056b3")
        self.upload_btn.grid(row=0, column=2, padx=15)

        # Detect button
        self.detect_btn = ctk.CTkButton(self.buttons_frame, text="Detect Image", command=self.start_detection_thread,
                                        width=btn_width, height=btn_height, corner_radius=corner_radius,
                                        fg_color="#FF8C00", hover_color="#cc7000")
        self.detect_btn.grid(row=0, column=3, padx=15)

        # Save result button
        self.save_btn = ctk.CTkButton(self.buttons_frame, text="Save Output", command=self.save_output,
                                      width=btn_width, height=btn_height, corner_radius=corner_radius,
                                      fg_color="#28A745", hover_color="#1e7e34")
        self.save_btn.grid(row=0, column=4, padx=15)

        # Exit button
        self.exit_btn = ctk.CTkButton(self.buttons_frame, text="Exit", command=self.exit_app,
                                      width=btn_width, height=btn_height, corner_radius=corner_radius,
                                      fg_color="#dc3545", hover_color="#a71d2a")
        self.exit_btn.grid(row=0, column=5, padx=15)

        # Store models, images, and progress bar
        self.models_cache = {}
        self.input_image = None
        self.input_display_photo = None
        self.output_image = None
        self.output_display_photo = None
        self.progress_popup = None
        self.progress_bar = None

    # Drag & Drop image handler
    def handle_drop(self, event):

        # Validate and load dropped image
        files = self.root.tk.splitlist(event.data)
        if not files:
            return
        file_path = files[0]
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            messagebox.showerror("Invalid file", "Please drop a valid image file (.jpg, .jpeg, .png).")
            return
        try:
            img = Image.open(file_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            self.input_image = img
            canvas_width = (self.screen_width - 3 * 20) // 2 - 2 * 10
            canvas_height = self.screen_height - 200
            img_resized = img.resize((canvas_width, canvas_height))
            self.input_display_photo = ImageTk.PhotoImage(img_resized)
            self.left_canvas.configure(image=self.input_display_photo, text="")
            self.right_canvas.configure(image=None)
            self.output_image = None
            self.output_display_photo = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dropped file:\n{e}")

    # Upload button logic
    def upload_image(self):
        # Load selected image from file dialog
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            img = Image.open(file_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            self.input_image = img
            canvas_width = (self.screen_width - 3 * 20) // 2 - 2 * 10
            canvas_height = self.screen_height - 200
            img_resized = img.resize((canvas_width, canvas_height))
            self.input_display_photo = ImageTk.PhotoImage(img_resized)
            self.left_canvas.configure(image=self.input_display_photo, text="")
            self.right_canvas.configure(image=None)
            self.output_image = None
            self.output_display_photo = None

    # Start detection in a new thread
    def start_detection_thread(self):
        # Run detection without freezing the GUI
        if self.input_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return

        # Clear previous output image display by setting empty transparent image
        empty_img = Image.new('RGBA', (1, 1), (255, 255, 255, 0))
        empty_photo = ImageTk.PhotoImage(empty_img)
        self.output_display_photo = empty_photo
        self.right_canvas.configure(image=self.output_display_photo)

        self.output_image = None

        self.show_loading_popup()
        threading.Thread(target=self.detect_image).start()

    # Show a loading popup during processing
    def show_loading_popup(self):
        if self.progress_popup and self.progress_popup.winfo_exists():
            return
        self.progress_popup = Toplevel(self.root)
        self.progress_popup.title("Processing...")
        self.progress_popup.geometry("300x100")
        self.progress_popup.resizable(False, False)
        self.progress_popup.grab_set()

        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 150
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 50
        self.progress_popup.geometry(f"+{x}+{y}")

        label = ctk.CTkLabel(self.progress_popup, text="Detecting... Please wait.")
        label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.progress_popup, mode='indeterminate')
        self.progress_bar.pack(pady=10, padx=20, fill='x')
        self.progress_bar.start()

    # Close the loading popup
    def close_loading_popup(self):
        if self.progress_popup and self.progress_popup.winfo_exists():
            self.progress_bar.stop()
            self.progress_popup.destroy()

    # Load the model only once and reuse from cache
    def load_model_if_needed(self, model_name):
        print(f"Loading model: {model_name}")
        if model_name in self.models_cache:
            print("Model loaded from cache")
            return self.models_cache[model_name]

        if model_name == "unet":
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models",
                "ancient_foundation_unet.pth")
            model, device = load_unet_model(model_path)
        elif model_name == "deeplab":
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models",
                "ancient_foundation_deeplab.pth")
            model, device = load_deeplab_model(model_path)
        else:
            messagebox.showerror("Error", f"Model '{model_name}' not supported.")
            return None, None

        print("Model loaded successfully")
        self.models_cache[model_name] = (model, device)
        return model, device

    # Run detection and show result in GUI
    def detect_image(self):
        model_name = self.model_var.get()
        print(f"Starting detection with model: {model_name}")
        model, device = self.load_model_if_needed(model_name)
        if model is None:
            print("Model loading failed!")
            self.close_loading_popup()
            return

        try:
            output_pil = run_detection(self.input_image, model, device)
            print("Detection completed successfully")
        except Exception as e:
            print(f"Detection error: {e}")
            self.root.after(0, self.close_loading_popup)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed:\n{e}"))
            return

        self.output_image = output_pil
        canvas_width = (self.screen_width - 3 * 20) // 2 - 2 * 10
        canvas_height = self.screen_height - 200
        output_resized = output_pil.resize((canvas_width, canvas_height))
        self.output_display_photo = ImageTk.PhotoImage(output_resized)


        def update_gui():
            self.right_canvas.configure(image=self.output_display_photo)
            self.close_loading_popup()

        self.root.after(0, update_gui)

    # Save output image to file
    def save_output(self):
        if self.output_image is None:
            messagebox.showwarning("Warning", "No output image to save!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg;*.jpeg")])
        if file_path:
            self.output_image.save(file_path)
            messagebox.showinfo("Saved", f"Output image saved at:\n{file_path}")

    # Exit the app
    def exit_app(self):
        self.root.quit()
        self.root.destroy()

# Run the app
if __name__ == "__main__":
    root = TkinterDnD.Tk()
    try:
        root.iconbitmap("../icons/icon.ico")
    except Exception as e:
        print(f"iconbitmap error: {e}")
    try:
        icon_img = ImageTk.PhotoImage(file="../icons/icon.ico")
        root.iconphoto(True, icon_img)
    except Exception as e:
        print(f"iconphoto error: {e}")

    root.state('zoomed')
    app = AncientFoundationGUI(root)
    root.mainloop()