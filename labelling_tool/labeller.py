import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import json
import os
import csv
import cv2
import threading  # Import threading module
import cupy as cp  # Import cupy for GPU computations
import numpy as np
# import math
import pandas as pd
from Tooltips import CanvasTooltip, Tooltip 

class ParticleLabelingApp:
    def __init__(self, master):
        self.finding_pairs = False
        
        self.master = master
        self.isDev = True
        master.title("Particle Connection Labeler")
        self.zoom_level = 0.5

        self.overlay_position = (200, 200)
        self.dragging_overlay = False
        
        self.create_sideBar()
        self.create_topBar()
        self.create_Canvas()

        # Variables
        self.original_image = None
        self.resized_image = None
        self.display_image = None
        self.image_id = None
        self.current_pair = None
        self.pair_index = -1
        self.click_radius = 5  # Radius for clickable particle locations in pixels

        # Initialize list to store labels
        self.labels = []
        self.labels_per_image = 100
        self.selected_particles = []
        self.deleted_particles = []

        # Bind the on_quit method to the window close event
        self.master.protocol("WM_DELETE_WINDOW", self.on_quit)
        
        self.load_metadata()
        self.load_image()
        self.schedule_save_data()

    def create_topBar(self):
        # Create a frame for the top bar
        top_bar_frame = tk.Frame(self.master)
        top_bar_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Create a frame for the status label and pack it to the left
        status_frame = tk.Frame(top_bar_frame, width=200)
        status_frame.pack(side=tk.LEFT, fill=tk.X)
        
        # Status label
        self.status = tk.Label(status_frame, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create a frame for the buttons and pack it to the center
        button_frame = tk.Frame(top_bar_frame)
        button_frame.pack(side=tk.TOP)
        
        # Load Image and JSON
        self.load_button = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT)

        self.connected_button = tk.Button(button_frame, text="<<", command=self.show_previous_pair)
        self.connected_button.pack(side=tk.LEFT)

        self.not_connected_button = tk.Button(button_frame, text=">>", command=self.show_next_pair)
        self.not_connected_button.pack(side=tk.LEFT)

        self.connected_button = tk.Button(button_frame, text="Connected", command=self.mark_connected)
        self.connected_button.pack(side=tk.LEFT)

        self.not_connected_button = tk.Button(button_frame, text="Not Connected", command=self.mark_not_connected)
        self.not_connected_button.pack(side=tk.LEFT)
        
        self.not_connected_button = tk.Button(button_frame, text="Pass", command=self.show_next_pair)
        self.not_connected_button.pack(side=tk.LEFT)
        
        self.not_connected_button = tk.Button(button_frame, text="Save Data", command=self.save_data)
        self.not_connected_button.pack(side=tk.LEFT)
    
    def toggle_switch(self, state, button, callback, on, off):
        state.set(not state.get())
        if state.get():
            button.config(image=on)
        else:
            button.config(image=off)
        callback()
        
    def create_sideBar(self):
        on = tk.PhotoImage(file = "./labelling_tool/on.png")
        off = tk.PhotoImage(file = "./labelling_tool/off.png")
        widget_width = 20  # Set a fixed width for all widgets
        # Create a frame on the left side
        self.left_frame = tk.Frame(self.master)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, pady=50)
        
        visibility_label = tk.Label(self.left_frame, text="Show Particle Locations", width=widget_width)
        visibility_label.pack()            
        # Initialize show_locations as a BooleanVar
        self.show_locations = tk.BooleanVar(value=False)  # Start with particles visible
        # Create a checkbox to toggle particle visibility
        self.visibility_toggle = tk.Button(
            self.left_frame,image = off,
            command=lambda: self.toggle_switch(self.show_locations,
                                       self.visibility_toggle, 
                                       self.show_hide_locations, 
                                       on, off)          
        )
        self.visibility_toggle.pack(anchor='nw', pady=(0, 5))
        Tooltip("Toggle particle visibility", self.visibility_toggle)
        
        chains_visibility_label = tk.Label(self.left_frame, text="Show Chains", width=widget_width)
        chains_visibility_label.pack()            
        # Initialize show_locations as a BooleanVar
        self.draw_chains_toggle = tk.BooleanVar(value=False)  # Start with particles visible
        # Create a checkbox to toggle particle visibility
        self.chains_visibility_toggle = tk.Button(
            self.left_frame,image = off,
            command=lambda: self.toggle_switch(self.draw_chains_toggle,
                                       self.chains_visibility_toggle, 
                                       self.draw_chains, 
                                       on, off)          
        )
        self.chains_visibility_toggle.pack(anchor='nw', pady=(0, 5))
        Tooltip("Toggle chains visibility",self.chains_visibility_toggle)
        
        add_label = tk.Label(self.left_frame,text="Add Particle Location")
        add_label.pack() 
        # Initialize add location as a BooleanVar
        self.adding_locations = tk.BooleanVar(value=False)  # Start with particles visible
        # Create a checkbox to toggle add functionality
        self.add_location_toggle = tk.Button(
            self.left_frame,image = off,
            command=lambda: self.toggle_switch(self.adding_locations,
                                       self.add_location_toggle, 
                                       self.toggle_adding_locations, 
                                       on, off),            
        )
        self.add_location_toggle.pack(anchor='nw', pady=(0, 5))
        Tooltip("Once on click on the canvas to add a particle",self.add_location_toggle)
        
        # Create a checkbox to toggle delete functionality
        self.delete_location_toggle = tk.Button(
            self.left_frame, 
            command=self.delete_locations,
            text="Delete Selection", 
            width=widget_width
            )
        self.delete_location_toggle.pack(anchor='nw', pady=(0, 5))
        Tooltip("Deleted selected particles",self.delete_location_toggle)
        
        # clear selection button
        self.clear_selection_button = tk.Button(self.left_frame,
                                                text="Clear Selection", 
                                                command=self.clear_particle_selection, width=widget_width)
        self.clear_selection_button.pack(anchor='nw', pady=(0, 5))
        Tooltip("Clear the selected particle",self.clear_selection_button)
        
        # mark currentlly selected particle as a chain
        self.mark_chain_button = tk.Button(self.left_frame,
                                             text="Mark Chain",
                                            command=self.mark_chain, width=widget_width)
        self.mark_chain_button.pack(anchor='nw', pady=(0, 5))
        Tooltip("Mark the selected particles as a chain",self.mark_chain_button)
        
        # add text box to enter custom chain id
        add_label = tk.Label(self.left_frame,text="Enter a custom chain id")
        add_label.pack()
        self.custom_chain_id = tk.Entry(self.left_frame, width=widget_width)
        self.custom_chain_id.pack(anchor='nw', pady=(0, 5))
        Tooltip("Enter a custom chain id to mark to current selection",self.custom_chain_id)
        
        # Dropdown to select the chain length
        self.chain_length = tk.IntVar(value=1)
        self.chain_length_label = tk.Label(self.left_frame, text="Set Chain Length", width=widget_width)
        self.chain_length_label.pack(anchor='nw', pady=(0, 5))
        self.chain_length_dropdown = tk.OptionMenu(self.left_frame, self.chain_length, 1, 4, 12, 24, 48)
        self.chain_length_dropdown.config(width=widget_width-3)
        self.chain_length_dropdown.pack(anchor='nw', pady=(0, 5))
        Tooltip("Select the chain length",self.chain_length_dropdown)
    
    def save_metadata(self):
        # Save metadata to a JSON file
        metadata = {
            'image_path': self.image_path,
            'image_name': self.image_name,
            'zoom_level': self.zoom_level,
            'overlay_position': self.overlay_position,
            'labels_per_image': self.labels_per_image
        }
        with open('./labelling_tool/metadata.json', 'w') as f:
            json.dump(metadata, f)
        
    def load_metadata(self):
        # Load metadata from a JSON file
        try:
            with open('./labelling_tool/metadata.json', 'r') as f:
                metadata = json.load(f)
                self.image_path = metadata['image_path']
                self.image_name = metadata['image_name']
                self.zoom_level = metadata['zoom_level']
                self.overlay_position = metadata['overlay_position']
                self.labels_per_image = metadata['labels_per_image']
        except FileNotFoundError:
            self.status.config(text="Metadata file not found.")
        except json.JSONDecodeError:
            self.status.config(text="Error decoding metadata file.")
    
    def on_quit(self):
        self.save_metadata()
        self.master.quit()      
    
    def create_Canvas(self):
        # Create a frame for the canvas and scrollbars
        canvas_frame = tk.Frame(self.master)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Configure the grid
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Create canvas
        self.canvas = tk.Canvas(canvas_frame)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        # Bind Ctrl+Scroll to the zoom function
        self.canvas.bind("<Control-MouseWheel>", self.zoom)
        # Bind mouse events for panning the main image
        self.canvas.bind('<ButtonPress-1>', self.start_pan)
        self.canvas.bind('<B1-Motion>', self.pan_image)
        self.canvas.bind('<ButtonRelease-1>', self.on_pan_stop)
        
        # Add vertical scrollbar
        v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(yscrollcommand=v_scroll.set)

        # Add horizontal scrollbar
        h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky='ew')
        self.canvas.configure(xscrollcommand=h_scroll.set)
        
    def load_image(self):
        if self.isDev:
            self.image_path = "F:\\hopper\\Images\\N=24\\theta=60\\wd=10cm\\A0010609.tif"
        else:
            self.image_path = filedialog.askopenfilename()
        if not self.image_path:
            return
        
        self.image_name = os.path.basename(self.image_path)
        self.status.config(text=f"Loading image: {self.image_name}")
        self.image_ext = os.path.splitext(self.image_path)[1]

        # Load and resize image for display
        masked_image_path = self.image_path.replace(self.image_ext, '_masked.tif')
        img = cv2.imread(masked_image_path)
        img_rsz = cv2.resize(img, None, fx=self.zoom_level, fy=self.zoom_level, interpolation=cv2.INTER_AREA)
        img_rsz_rgb = cv2.cvtColor(img_rsz, cv2.COLOR_BGR2RGB)  # Convert to RGB for Tkinter
        
        self.original_image = Image.fromarray(img)
        self.resized_image = Image.fromarray(img_rsz_rgb)  # Initialize resized image
        self.display_image = ImageTk.PhotoImage(image=self.resized_image)

        self.canvas.config(width=img.shape[1], height=img.shape[0])
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)

        self.canvas.update()

        # After displaying the image on the canvas
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        # Load Particle locations
        csv_path = self.image_path.replace(self.image_ext, '_trackpy.csv')
        pd_csv_path = f"./labelled_data/{self.image_name.replace(self.image_ext, '_data.csv')}"
        print(f"pd_csv: {pd_csv_path}")
        pd_csv_exists = os.path.exists(pd_csv_path)
        
        try:
            if pd_csv_exists:
                # self.particles = pd.DataFrame(columns=['x', 'y','mass','size','ecc','signal','raw_mass','ep','isParticle','chainId','pairIds'])                        
                self.particles = pd.read_csv(pd_csv_path)
                print("Loaded pd_csv")
                print(pd_csv_path)
                self.start_find_pairs_thread()  # Start the find_pairs method in a new thread                
            elif os.path.exists(csv_path):
                print("Pd_csv not found, loading trackpy csv")
                print(csv_path)
                self.read_trackpy_csv(csv_path)
                self.start_find_pairs_thread()  # Start the find_pairs method in a new thread
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            self.status.config(text="Warning: CSV file not found.")
        except json.JSONDecodeError:
            self.status.config(text="Warning: Error decoding CSV file.")

    def read_trackpy_csv(self, csv_path):
        self.particles = pd.read_csv(csv_path)
        self.particles['isParticle'] = True
        self.particles['chainId'] = -1
        self.particles['pairIds'] = [[] for _ in range(len(self.particles))]
        self.particles['nonPairIds'] = [[] for _ in range(len(self.particles))]
        self.particles['id'] = self.particles.index
        # add attributes to the particles dataframe 
        self.particles.attrs['image_path']= self.image_path
        self.particles.attrs['trackpy_csv_path']= csv_path

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
        self.canvas.configure(scrollregion=self.canvas.bbox(self.image_id))
        self.draw_line_over_image()
        self.refresh_locations()
        
    def start_find_pairs_thread(self):
        thread = threading.Thread(target=self.find_pairs)
        thread.start()

    def find_pairs(self):
        max_distance = 50  # Maximum pixel distance to consider
        pairs = []
        checked_pairs = set()
        self.finding_pairs = True
        processed_pairs = 0
        x_y_columns = self.particles[['x', 'y']]
        particles_array = cp.array(x_y_columns.values)  # Convert particles to GPU array

        for i in range(10): #range(len(self.particles)):
            distances = cp.sqrt(cp.sum((particles_array[i] - particles_array[i+1:]) ** 2, axis=1))
            close_particles = cp.where(distances <= max_distance)[0]

            for idx in close_particles:
                idx = int(idx)
                j = i + 1 + idx  # Adjust index because we sliced the array
                if (i, j) in checked_pairs or (j, i) in checked_pairs:
                    continue

                pairs.append((self.particles['id'][i], self.particles['id'][j]))
                checked_pairs.add((i, j))

                processed_pairs += 1
                if processed_pairs % 100 == 0:  # Update status every 100 pairs
                    self.status.config(text=f"Processed {processed_pairs} pairs")
                    self.master.update_idletasks()
        
        self.finding_pairs = False
        self.pairs = pairs  # Save the pairs to self.pairs
        self.pair_index = -1  # Reset pair index
        self.show_next_pair()  # Show the first pair after processing

    def show_next_pair(self):
        if self.pair_index >= (self.labels_per_image - 1):
            self.status.config(text="ENOUGH PAIRS LABELED.")
            # Save labels to a JSON file
            self.save_data()
            return
        if (self.pair_index + 1) >= len(self.pairs):
            self.status.config(text="All pairs labeled.")
            # Save labels to a JSON file
            self.save_data()
            return

        self.pair_index += 1
        self.current_pair = self.pairs[self.pair_index]
        print(f"Pair index: {self.pair_index} of {len(self.pairs) - 1}")        
        self.draw_pair(self.current_pair)

    def show_previous_pair(self):
        if self.pair_index == 0:
            self.status.config(text="No previous pair.")
            return

        self.pair_index -= 1
        self.current_pair = self.pairs[self.pair_index]
        print(f"Pair index: {self.pair_index} of {len(self.pairs) - 1}")            
        self.draw_pair(self.current_pair)

    def draw_line_over_image(self):
        particle_1 = self.particles[self.particles['id'] == self.current_pair[0]]
        particle_2 = self.particles[self.particles['id'] == self.current_pair[1]]
        (x1, y1) = particle_1[['x', 'y']].values[0] 
        (x2, y2) = particle_2[['x', 'y']].values[0]
                        
        zoomed_x1 = x1 * self.zoom_level
        zoomed_y1 = y1 * self.zoom_level
        zoomed_x2 = x2 * self.zoom_level
        zoomed_y2 = y2 * self.zoom_level
        
        # adjust by image location
        zoomed_x1 += self.canvas.coords(self.image_id)[0]
        zoomed_y1 += self.canvas.coords(self.image_id)[1]
        zoomed_x2 += self.canvas.coords(self.image_id)[0]
        zoomed_y2 += self.canvas.coords(self.image_id)[1]
        
        # Clear previous overlays
        if hasattr(self, 'line_id'):
            self.canvas.delete(self.line_id)
            
        # Draw line between particles on the canvas
        self.line_id = self.canvas.create_line(zoomed_x1, zoomed_y1, zoomed_x2, zoomed_y2, fill='red', width=2)
        
        # print(f"Drawing line between particles: ({x1}, {y1}) and ({x2}, {y2})")

    def draw_pair(self, pair):
        if self.finding_pairs:
            return
        
        particle_1 = self.particles[self.particles['id'] == pair[0]]
        particle_2 = self.particles[self.particles['id'] == pair[1]]
        (x1, y1) = particle_1[['x', 'y']].values[0] 
        (x2, y2) = particle_2[['x', 'y']].values[0]

        if hasattr(self, 'overlay_image_id'):
            self.canvas.delete(self.overlay_image_id)
    
        self.draw_line_over_image()
    
        # Calculate midpoint between particles
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
    
        # Define zoom level and crop radius
        overlay_zoom_scale = 2  # Adjust zoom level as needed
        overlay_crop_radius = 50  # Adjust crop radius as needed
    
        # Calculate bounding box for cropping, centered at midpoint
        min_x = max(0, int(mid_x - overlay_crop_radius))
        min_y = max(0, int(mid_y - overlay_crop_radius))
        max_x = min(self.original_image.width, int(mid_x + overlay_crop_radius))
        max_y = min(self.original_image.height, int(mid_y + overlay_crop_radius))
    
        # Crop and zoom the image
        cropped_image = self.original_image.crop((min_x, min_y, max_x, max_y))
        overlay_size = (int(cropped_image.width * overlay_zoom_scale), int(cropped_image.height * overlay_zoom_scale))
        overlay_image = cropped_image.resize(overlay_size, resample=Image.Resampling.LANCZOS)
    
        # Draw line between particles on the zoomed image
        draw_overlay = ImageDraw.Draw(overlay_image)
        adj_x1 = (x1 - min_x) * overlay_zoom_scale
        adj_y1 = (y1 - min_y) * overlay_zoom_scale
        adj_x2 = (x2 - min_x) * overlay_zoom_scale
        adj_y2 = (y2 - min_y) * overlay_zoom_scale
        draw_overlay.line((adj_x1, adj_y1, adj_x2, adj_y2), fill="red", width=2)
    
        # Create circular mask
        mask = Image.new('L', overlay_image.size, 0)
        draw_mask = ImageDraw.Draw(mask)
        
        draw_mask.ellipse((0, 0, overlay_image.width, overlay_image.height), fill=255)
        overlay_image.putalpha(mask)
    
        # Convert to Tkinter image
        self.overlay_image_tk = ImageTk.PhotoImage(overlay_image)
    
        # Add zoomed image to canvas
        self.overlay_image_id = self.canvas.create_image(
            self.overlay_position[0],
            self.overlay_position[1], 
            image=self.overlay_image_tk, 
            anchor=tk.CENTER
            )

        # Make the overlay movable
        self.canvas.tag_bind(self.overlay_image_id, '<ButtonPress-1>', self.on_overlay_press)
        self.canvas.tag_bind(self.overlay_image_id, '<B1-Motion>', self.on_overlay_motion)
        self.canvas.tag_bind(self.overlay_image_id, '<ButtonRelease-1>', self.on_overlay_release)
    
        self.status.config(text=f"Labeling pair {self.pair_index} of {len(self.pairs)}")

    def on_overlay_press(self, event):
        self.dragging_overlay = True
        self.overlay_start_x = event.x
        self.overlay_start_y = event.y
        self.canvas.tag_raise(self.overlay_image_id)
        return 'break'

    def on_overlay_motion(self, event):
        self.dragging_overlay = True
        dx = event.x - self.overlay_start_x
        dy = event.y - self.overlay_start_y
        self.canvas.move(self.overlay_image_id, dx, dy)
        self.overlay_start_x = event.x
        self.overlay_start_y = event.y
        # Update the overlay position
        self.overlay_position = (self.overlay_position[0] + dx, self.overlay_position[1] + dy)
        return 'break'
    
    def on_overlay_release(self, event):
        self.dragging_overlay = False
    
    def start_pan(self, event):
        # Record the current mouse position
        if self.dragging_overlay is False:
            # Remove all particle locations
            self.clear_all_locations() # this makes panning image much faster
            # remove all chain outlines
            self.clear_chains()
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.image_start_x, self.image_start_y = self.canvas.coords(self.image_id)
    
    def pan_image(self, event):
        # Move the image to the new position
        if self.dragging_overlay is False:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            new_x = self.image_start_x + dx
            new_y = self.image_start_y + dy
            
            self.canvas.coords(self.image_id, min(0,new_x), min(0,new_y))
            
    def on_pan_stop(self, event):
        # print("Panning stopped")
        self.draw_line_over_image()
        self.refresh_locations()
        
    def mark_connected(self):
        self.particles.loc[
            self.particles['id'] == self.current_pair[0], 'pairIds'
            ] = self.particles.loc[
                self.particles['id'] == self.current_pair[0], 'pairIds'
                ].apply(lambda lst: lst + [self.current_pair[1]])
        self.particles.loc[
            self.particles['id'] == self.current_pair[1], 'pairIds'
            ] = self.particles.loc[
                self.particles['id'] == self.current_pair[1], 'pairIds'
                ].apply(lambda lst: lst + [self.current_pair[0]]) 

        self.show_next_pair()

    def mark_not_connected(self):
        # Store label for the current pair as not connected
        self.particles.loc[
            self.particles['id'] == self.current_pair[0], 'nonPairIds'
            ] = self.particles.loc[
                self.particles['id'] == self.current_pair[0], 'nonPairIds'
                ].apply(lambda lst: lst + [self.current_pair[1]])
        self.particles.loc[
            self.particles['id'] == self.current_pair[1], 'nonPairIds'
            ] = self.particles.loc[
                self.particles['id'] == self.current_pair[1], 'nonPairIds'
                ].apply(lambda lst: lst + [self.current_pair[0]]) 

        self.show_next_pair()

    def save_data(self):
        # save dataframe
        self.particles.to_csv(f"./labelled_data/{self.image_name.replace(self.image_ext, '_data.csv')}", index=False)
    
    def schedule_save_data(self):
        # Call the save_data method
        self.save_data()
        # Schedule the function to run again after 60 seconds
        threading.Timer(60, self.schedule_save_data).start()
        
    def refresh_locations(self):
        print("Refreshing locations")
        # Remove all particle locations
        self.clear_all_locations()
        # Redraw particle locations
        self.show_hide_locations()
    
    def show_hide_locations(self):
        # Show or hide the particle locations on the canvas
        if self.show_locations.get():
            self.draw_particle_locations()
        else:
            self.clear_all_locations()
            
    def clear_all_locations(self):
        self.canvas.delete('particle_location')
        self.canvas.delete('particle_click_region')
                
    def draw_particle_locations(self, particles = None):
        if particles is None:
            particles = self.particles.copy()
        # Get the visible region of the canvas in canvas coordinates
        x0 = self.canvas.canvasx(0)
        y0 = self.canvas.canvasy(0)
        x1 = self.canvas.canvasx(self.canvas.winfo_width())
        y1 = self.canvas.canvasy(self.canvas.winfo_height())
        
        # Get image coordinates
        image_coords = self.canvas.coords(self.image_id)
        image_x, image_y = image_coords[0], image_coords[1]

        # Precompute zoom level and image offset
        zoom_level = self.zoom_level
        
        # Convert particle positions to canvas coordinates
        cp_canvas_x = cp.asarray(particles['x']) * zoom_level + image_x
        cp_canvas_y = cp.asarray(particles['y']) * zoom_level + image_y
        # filter particles
        cp_isParticle = cp.asarray(particles['isParticle'])
        
        particles['canvas_x'] = cp_canvas_x.get()
        particles['canvas_y'] = cp_canvas_y.get()
        particles['isParticle'] = cp_isParticle.get()

        # Filter particles within the visible region
        visible_particles_mask = (
            (particles['canvas_x'] >= x0) & (particles['canvas_x'] <= x1) &
            (particles['canvas_y'] >= y0) & (particles['canvas_y'] <= y1) &
            (particles['isParticle'] == True)
        )
        visible_particles = particles[visible_particles_mask]
        
        # Draw particle locations on the canvas
        for index, row in visible_particles.iterrows():
            self.draw_particle(index, row['canvas_x'], row['canvas_y'])
        # Draw chains if the toggle is on
        if self.draw_chains_toggle.get():
            self.draw_chains()
        # Raise the overlay image to the top
        self.canvas.tag_raise(self.overlay_image_id)
        
    def draw_particle(self,index, x, y):
        radius = 1.3
        if index in self.selected_particles:
            outline = 'red'
            fill = 'red'
        else:
            outline = 'blue'
            fill = 'blue'
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius, 
            fill=fill, outline=outline,
            tags=('particle_location', f'particle_{index}')
        )
        # Create an invisible oval for the clickable region
        self.canvas.create_oval(
            x - self.click_radius, y - self.click_radius, x + self.click_radius, y + self.click_radius,
            outline='', fill='', tags=(f'clickable_particle_{index}','particle_click_region')
        )
        # Bind click event to the particle
        self.canvas.tag_bind(f'clickable_particle_{index}', '<Button-1>', lambda event, idx=index: self.on_particle_click(event, idx))
        # Bind enter and leave events to change the cursor
        self.canvas.tag_bind(f'clickable_particle_{index}', '<Enter>', lambda event: self.canvas.config(cursor='hand2'))
        self.canvas.tag_bind(f'clickable_particle_{index}', '<Leave>', lambda event: self.canvas.config(cursor=''))
        
    def on_particle_click(self, event, index):
        # Get the item that was clicked
        item = self.canvas.find_withtag(f'particle_{index}')[0]
        # Highlight the selected particle
        self.canvas.itemconfig(item, outline='red', fill='red')
        # Store the selected particle's ID
        # self.selected_particle = item
        if index not in self.selected_particles:
            self.selected_particles.append(index)
        else:
            self.selected_particles.remove(index)
        print(f"Selected particles: {self.selected_particles}")

    def delete_locations(self):
        if hasattr(self, 'selected_particles') and self.selected_particles:
            self.recently_deleted_particles = self.particles[self.particles['id'].isin(self.selected_particles)].copy()
            self.deleted_particles.append(self.recently_deleted_particles)
            self.particles.loc[self.particles['id'].isin(self.selected_particles), 'isParticle'] = False
            self.selected_particles = []
            self.refresh_locations()
        else:
            self.status.config(text="No particles selected.")
    
    def undo_delete(self):
        # Undo the last delete operation
        # but the deleted particles are now restored to the list at the end
        # change the logic to restore the particles at the same index if required
        if hasattr(self, 'deleted_particles'):
            self.particles += self.recently_deleted_particles
            self.refresh_locations()
            self.recently_deleted_particles = []
        else:
            self.status.config(text="Nothing to undo.")

    def toggle_adding_locations(self):
        # print("Toggling adding locations: ", self.adding_locations.get())
        # Add a new particle location
        if self.adding_locations.get():
            self.canvas.bind('<Button-1>', self.add_location)
        else:
            self.canvas.unbind('<Button-1>')
            self.canvas.bind('<ButtonPress-1>', self.start_pan)

    def add_location(self, event):
        # Add a new particle location
        x = (event.x - self.canvas.coords(self.image_id)[0]) / self.zoom_level
        y = (event.y - self.canvas.coords(self.image_id)[1]) / self.zoom_level
        self.particles.append((x, y))
        self.refresh_locations()
        print(f"Added particle at: ({x}, {y})")

    def clear_chains(self):
        self.canvas.delete('chain_outline')
        
    def draw_chains(self):
        # clear all chain outlines
        self.clear_chains()
        if not self.draw_chains_toggle.get():
            return
        # Draw chains on the canvas
        chain_ids = self.particles['chainId'].unique()
        for chain_id in chain_ids:
            if chain_id != -1:
                self.draw_chain(chain_id)
    
    def draw_chain(self, chain_id):
        # Draw a chain on the canvas
        # Get the particles in the chain
        chain_particles = self.particles[self.particles['chainId'] == chain_id]
        # Get the particle locations
        x_y_columns = chain_particles[['x', 'y']]
        particles_array = cp.array(x_y_columns.values)
        # Get the chain color
        chain_color = self.get_chain_color(chain_id)
        # Draw thick outline around the particles in the chain
        for index, (x, y) in enumerate(particles_array):
            canvas_x = x * self.zoom_level + self.canvas.coords(self.image_id)[0]
            canvas_y = y * self.zoom_level + self.canvas.coords(self.image_id)[1]
            radius = 5  # Adjust the radius for the outline
            outline_handle = self.canvas.create_oval(
            canvas_x - radius, canvas_y - radius, canvas_x + radius, canvas_y + radius,
            outline=chain_color, width=3, tags='chain_outline'
            )
            CanvasTooltip(canvas=self.canvas, tag_or_id=outline_handle, text=f"Chain ID: {chain_id}")

    def get_chain_color(self, chain_id):
        """
        Returns the same color string for each chain_id.
        """
        if not hasattr(self, 'chain_colors'):
            self.chain_colors = {}
        if chain_id not in self.chain_colors:
            # Assign a new color, e.g. a hex code
            # In production you might pick from a predefined list or generate a random new color
            self.chain_colors[chain_id] = f"#{(hash(str(chain_id)) & 0xFFFFFF):06x}"
        return self.chain_colors[chain_id]
    
    def mark_chain(self):
        # Mark the selected particles as a chain
        if hasattr(self, 'selected_particles') and self.selected_particles:
            # Get the chain length
            chain_length = self.chain_length.get()
            # get the max of chainId column of self.particles dataframe
            max_chain_id = self.particles['chainId'].max()
            # read the custom chain id
            custom_chain_id = self.custom_chain_id.get()
            # check if selected particles are already in a chain
            chains_ids = self.particles.loc[self.particles['id'].isin(self.selected_particles), 'chainId']
            print(f"chains_ids: {chains_ids}")
            
            # if there are more than one chain_id except -1, do not mark any particles as chain, throw an error
            # Count the number of valid chain_ids
            valid_chain_ids = chains_ids[chains_ids != -1]
            num_valid_chain_ids = valid_chain_ids.nunique()
            print(f"Number of valid chain_ids: {num_valid_chain_ids}")        
            if num_valid_chain_ids > 1:
                self.status.config(text="Selected particles belong to multiple chains.")
                # clear selection
                self.clear_particle_selection()
                return
            
            # if only some chain_id and -1 , mark all particles with chain_id
            if num_valid_chain_ids == 1:
                chain_id = valid_chain_ids.iloc[0]
                print(f"marking all with chain_id: {chain_id}")
                self.particles.loc[self.particles['id'].isin(self.selected_particles), 'chainId'] = chain_id
                # clear selection
                self.clear_particle_selection()
                return
            # if all chain_ids == -1 , mark all particles with new chain_id
            if num_valid_chain_ids == 0:
                if custom_chain_id:
                    new_chain_id = int(custom_chain_id)
                else:
                    if max_chain_id == -1 :
                        new_chain_id = 0
                    else :
                        new_chain_id = max_chain_id + 1
                self.particles.loc[self.particles['id'].isin(self.selected_particles), 'chainId'] = new_chain_id
                # clear custom chain id
                self.custom_chain_id.delete(0, tk.END)
                print("new chain id:",new_chain_id)
                # clear selection
                self.clear_particle_selection()
                return
        else:
            self.status.config(text="No particles selected to mark as chain.")
    
    def clear_particle_selection(self):
        self.selected_particles = []
        print(f"cleared Selected particles: {self.selected_particles}")
        self.refresh_locations()

if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleLabelingApp(root)
    root.mainloop()
