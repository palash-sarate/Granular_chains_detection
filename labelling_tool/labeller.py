import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import json
import os
import csv
import cv2
import threading  # Import threading module
import cupy as cp  # Import cupy for GPU computations
# import math

class ParticleLabelingApp:
    def __init__(self, master):
        self.master = master
        self.isDev = True
        master.title("Particle Connection Labeler")
        self.resize_scale = 0.2
        self.zoom_level = 1.0

        # Load Image and JSON
        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        # Create buttons for user input
        button_frame = tk.Frame(master)
        button_frame.pack()

        self.connected_button = tk.Button(button_frame, text="Connected", command=self.mark_connected)
        self.connected_button.pack(side=tk.LEFT)

        self.not_connected_button = tk.Button(button_frame, text="Not Connected", command=self.mark_not_connected)
        self.not_connected_button.pack(side=tk.LEFT)

        # Create a frame for the canvas and scrollbars
        canvas_frame = tk.Frame(master)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Configure the grid
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Create canvas
        self.canvas = tk.Canvas(canvas_frame)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        # Bind Ctrl+Scroll to the zoom function
        self.canvas.bind("<Control-MouseWheel>", self.zoom)
        # Add vertical scrollbar
        v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(yscrollcommand=v_scroll.set)

        # Add horizontal scrollbar
        h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky='ew')
        self.canvas.configure(xscrollcommand=h_scroll.set)

        # Label status
        self.status = tk.Label(master, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Variables
        self.image = None
        self.particles = []
        self.current_pair = None
        self.pair_index = 0

        # Initialize list to store labels
        self.labels = []

        if self.isDev:
            self.load_image()

    def load_image(self):
        if self.isDev:
            image_path = "E:\\hopper\\Images\\N=24\\theta=60\\wd=10cm\\A0010609.tif"
        else:
            image_path = filedialog.askopenfilename()
        if not image_path:
            return

        # Load and resize image for display
        masked_image_path = image_path.replace('.tif', '_masked.tif')
        img = cv2.imread(masked_image_path)
        img = cv2.resize(img, None, fx=self.resize_scale, fy=self.resize_scale, interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Tkinter
        self.original_image = Image.fromarray(img_rgb)
        self.resized_image = self.original_image  # Initialize resized image
        self.display_image = ImageTk.PhotoImage(image=self.resized_image)

        self.canvas.config(width=img.shape[1], height=img.shape[0])
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)

        self.canvas.update()

        # After displaying the image on the canvas
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        # Load Particle locations
        csv_path = image_path.replace('.tif', '_trackpy.csv')
        print(csv_path)
        try:
            if os.path.exists(csv_path):
                self.particles = []
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.particles.append((float(row['x']), float(row['y'])))
                self.start_find_pairs_thread()  # Start the find_pairs method in a new thread
                # self.show_next_pair()  # Show the first pair after processing
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            self.status.config(text="Warning: CSV file not found.")
        except json.JSONDecodeError:
            self.status.config(text="Warning: Error decoding CSV file.")

        # Load mask
        # mask_path = image_path.replace('.tif', '_mask.tif')
        # if os.path.exists(mask_path):
        #     self.mask = Image.open(mask_path)
        #     self.mask = ImageTk.PhotoImage(self.mask)
        
    def zoom(self, event):
        # Determine the zoom factor based on scroll direction
        if event.delta > 0:
            self.zoom_level *= 1.1  # Zoom in
        else:
            self.zoom_level *= 0.9  # Zoom out

        # Limit the zoom level to reasonable bounds
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0))

        # Calculate the new size
        width, height = self.original_image.size
        new_size = (int(width * self.zoom_level), int(height * self.zoom_level))

        # Resize the image
        self.resized_image = self.original_image.resize(
            new_size, resample=Image.Resampling.LANCZOS
        )
        self.display_image = ImageTk.PhotoImage(self.resized_image)

        # Update the image on the canvas
        self.canvas.itemconfig(self.image_id, image=self.display_image)

        # Adjust the scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def start_find_pairs_thread(self):
        thread = threading.Thread(target=self.find_pairs)
        thread.start()

    def find_pairs(self):
        max_distance = 50  # Maximum pixel distance to consider
        pairs = []
        checked_pairs = set()

        # total_pairs = len(self.particles) * (len(self.particles) - 1) // 2
        processed_pairs = 0

        particles_array = cp.array(self.particles)  # Convert particles to GPU array

        for i in range(10):# range(len(self.particles)):
            distances = cp.sqrt(cp.sum((particles_array[i] - particles_array[i+1:]) ** 2, axis=1))
            close_particles = cp.where(distances <= max_distance)[0]

            for idx in close_particles:
                idx = int(idx)
                j = i + 1 + idx  # Adjust index because we sliced the array
                if (i, j) in checked_pairs or (j, i) in checked_pairs:
                    continue

                pairs.append((tuple(self.particles[i]), tuple(self.particles[j])))
                checked_pairs.add((i, j))

                processed_pairs += 1
                if processed_pairs % 100 == 0:  # Update status every 100 pairs
                    self.status.config(text=f"Processed {processed_pairs} pairs")
                    self.master.update_idletasks()

        self.pairs = pairs  # Save the pairs to self.pairs
        self.pair_index = 0  # Reset pair index
        self.show_next_pair()  # Show the first pair after processing

    def show_next_pair(self):
        if self.pair_index >= len(self.pairs):
            self.status.config(text="All pairs labeled.")
            return

        self.current_pair = self.pairs[self.pair_index]
        self.draw_pair(self.current_pair)
        self.pair_index += 1

    def draw_pair(self, pair):
        (x1, y1), (x2, y2) = pair
        # Convert particle coordinates to canvas coordinates
        x1 = x1 * self.resize_scale
        y1 = y1 * self.resize_scale
        x2 = x2 * self.resize_scale
        y2 = y2 * self.resize_scale
        # Clear previous overlays
        if hasattr(self, 'line_id'):
            self.canvas.delete(self.line_id)
        if hasattr(self, 'zoomed_image_id'):
            self.canvas.delete(self.zoomed_image_id)
    
        # Draw line between particles on the canvas (overlay)
        self.line_id = self.canvas.create_line(x1, y1, x2, y2, fill='red', width=2)
    
        # Calculate midpoint between particles
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
    
        # Define zoom level and crop radius
        zoom_scale = 2  # Adjust zoom level as needed
        crop_radius = 50  # Adjust crop radius as needed
    
        # Calculate bounding box for cropping, centered at midpoint
        min_x = max(0, int(mid_x - crop_radius))
        min_y = max(0, int(mid_y - crop_radius))
        max_x = min(self.image.width, int(mid_x + crop_radius))
        max_y = min(self.image.height, int(mid_y + crop_radius))
    
        # Crop and zoom the image
        cropped_image = self.image.crop((min_x, min_y, max_x, max_y))
        zoomed_size = (int(cropped_image.width * zoom_scale), int(cropped_image.height * zoom_scale))
        zoomed_image = cropped_image.resize(zoomed_size, resample=Image.Resampling.LANCZOS)
    
        # Draw line between particles on the zoomed image
        draw_zoom = ImageDraw.Draw(zoomed_image)
        adj_x1 = (x1 - min_x) * zoom_scale
        adj_y1 = (y1 - min_y) * zoom_scale
        adj_x2 = (x2 - min_x) * zoom_scale
        adj_y2 = (y2 - min_y) * zoom_scale
        draw_zoom.line((adj_x1, adj_y1, adj_x2, adj_y2), fill="red", width=2)
    
        # Create circular mask
        mask = Image.new('L', zoomed_image.size, 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.ellipse((0, 0, zoomed_image.width, zoomed_image.height), fill=255)
        zoomed_image.putalpha(mask)
    
        # Convert to Tkinter image
        self.zoomed_image_tk = ImageTk.PhotoImage(zoomed_image)
    
        # Add zoomed image to canvas
        self.zoomed_image_id = self.canvas.create_image(200, 200, image=self.zoomed_image_tk, anchor=tk.CENTER)
    
        # Make the overlay movable
        self.canvas.tag_bind(self.zoomed_image_id, '<ButtonPress-1>', self.on_zoomed_press)
        self.canvas.tag_bind(self.zoomed_image_id, '<B1-Motion>', self.on_zoomed_motion)
    
        self.status.config(text=f"Labeling pair {self.pair_index} of {len(self.pairs)}")
    
    def on_zoomed_press(self, event):
        self.zoomed_start_x = event.x
        self.zoomed_start_y = event.y
        self.canvas.tag_raise(self.zoomed_image_id)
    
    def on_zoomed_motion(self, event):
        dx = event.x - self.zoomed_start_x
        dy = event.y - self.zoomed_start_y
        self.canvas.move(self.zoomed_image_id, dx, dy)
        self.zoomed_start_x = event.x
        self.zoomed_start_y = event.y

    def mark_connected(self):
        # Store label for the current pair as connected
        self.labels.append((self.current_pair, 'connected'))
        self.show_next_pair()

    def mark_not_connected(self):
        # Store label for the current pair as not connected
        self.labels.append((self.current_pair, 'not connected'))
        self.show_next_pair()

if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleLabelingApp(root)
    root.mainloop()
