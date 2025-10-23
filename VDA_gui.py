import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import logging
import torch
import numpy as np
import threading
import json
import shutil
import cv2  # Added for PNG saving
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video
from typing import Tuple, Any, cast

from queue import Queue, Empty  # Used for thread-safe communication and its exception

# Define model configurations (kept from original)
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}, # Added vitb
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# --- Tooltip Class for Tkinter (Replaces PySimpleGUI tooltips) ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.id = None
        self.x = self.y = 0
        widget.bind('<Enter>', self.schedule)
        widget.bind('<Leave>', self.hidetip)

    def schedule(self, event=None):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
        self.id = None

    def showtip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tip_window, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self, event=None):
        self.unschedule()
        if self.tip_window:
            self.tip_window.destroy()
        self.tip_window = None

# --- Main Application Class ---
class VideoDepthAnythingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Video Depth Anything')
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.update_queue = Queue()
        self.settings_file = 'config_vda.json'
        
        self.load_settings()
        self.create_variables()
        self.create_widgets()
        
        # Start checking the queue for updates from the thread
        self.after(100, self.check_queue)
        
        # Protocol handler for window closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_settings(self):
        self.saved_values = {}
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                try:
                    self.saved_values = json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"Error loading {self.settings_file}. Using defaults.")
                    self.saved_values = {}

    def save_settings(self):
        current_values = {
            '-INPUT_FOLDER-': self.input_path_var.get(), 
            '-OUTPUT_DIR-': self.output_dir_var.get(),
            '-ENCODER-': self.encoder_var.get(),
            '-INPUT_SIZE-': self.input_size_var.get(),
            '-MAX_RES-': self.max_res_var.get(),
            '-MAX_LEN-': self.max_len_var.get(),
            '-TARGET_FPS-': self.target_fps_var.get(),
            '-FP32-': self.fp32_var.get(),
            '-SAVE_COLOR-': self.save_color_var.get(),
            '-SAVE_NPZ-': self.save_npz_var.get(),
            '-SAVE_EXR-': self.save_exr_var.get(),
            '-CREATE_SRC-': self.create_src_var.get(),
            '-TTA-': self.tta_var.get(),
            '-RESUME-': self.resume_var.get(),
            '-SAVE_PNG-': self.save_png_var.get(),
            '-PNG_16BIT-': self.png_16bit_var.get(),
            '-PNG_COMPRESSION-': self.png_compression_var.get(),
            '-MP4_CRF-': self.mp4_crf_var.get(),
            '-METRIC-': self.metric_var.get(),
            '-INVERT_METRIC-': self.invert_metric_var.get(),
        }
        with open(self.settings_file, 'w') as f:
            json.dump(current_values, f, indent=4)

    def create_variables(self):
        # Define default values using saved settings
        self.input_path_var = tk.StringVar(value=self.saved_values.get('-INPUT_FOLDER-', ''))
        self.output_dir_var = tk.StringVar(value=self.saved_values.get('-OUTPUT_DIR-', ''))
        self.encoder_var = tk.StringVar(value=self.saved_values.get('-ENCODER-', 'vitl'))
        self.input_size_var = tk.StringVar(value=str(self.saved_values.get('-INPUT_SIZE-', 518)))
        self.max_res_var = tk.StringVar(value=str(self.saved_values.get('-MAX_RES-', 1280)))
        
        self.max_len_var = tk.StringVar(value=str(self.saved_values.get('-MAX_LEN-', -1)))
        self.target_fps_var = tk.StringVar(value=str(self.saved_values.get('-TARGET_FPS-', -1)))
        
        self.png_compression_var = tk.IntVar(value=self.saved_values.get('-PNG_COMPRESSION-', 1))
        self.mp4_crf_var = tk.IntVar(value=self.saved_values.get('-MP4_CRF-', 18))
        
        self.fp32_var = tk.BooleanVar(value=self.saved_values.get('-FP32-', False))
        self.save_color_var = tk.BooleanVar(value=self.saved_values.get('-SAVE_COLOR-', False))
        self.save_npz_var = tk.BooleanVar(value=self.saved_values.get('-SAVE_NPZ-', False))
        self.save_exr_var = tk.BooleanVar(value=self.saved_values.get('-SAVE_EXR-', False))
        self.create_src_var = tk.BooleanVar(value=self.saved_values.get('-CREATE_SRC-', False))
        self.tta_var = tk.BooleanVar(value=self.saved_values.get('-TTA-', False))
        self.resume_var = tk.BooleanVar(value=self.saved_values.get('-RESUME-', True))
        self.save_png_var = tk.BooleanVar(value=self.saved_values.get('-SAVE_PNG-', False))
        self.png_16bit_var = tk.BooleanVar(value=self.saved_values.get('-PNG_16BIT-', False))
        self.metric_var = tk.BooleanVar(value=self.saved_values.get('-METRIC-', False))
        self.invert_metric_var = tk.BooleanVar(value=self.saved_values.get('-INVERT_METRIC-', False))

    def create_widgets(self):
        
        # Configure grid column weights for proper resizing
        self.grid_columnconfigure(1, weight=1)
        
        r = 0 # Row counter

        # Title
        title_label = ttk.Label(self, text='Video Depth Anything GUI', font=('Helvetica', 16))
        title_label.grid(row=r, column=0, columnspan=3, pady=10, padx=10, sticky='w')
        r += 1

        # Input Path (Folder/File)
        lbl_in_path = ttk.Label(self, text='Input Path')
        lbl_in_path.grid(row=r, column=0, padx=10, pady=5, sticky='w')
        ToolTip(lbl_in_path, "Select the folder containing video files or a single video file.")
        
        entry_in_path = ttk.Entry(self, textvariable=self.input_path_var)
        entry_in_path.grid(row=r, column=1, padx=5, pady=5, sticky='ew')
        
        # Frame for the two buttons in column 2
        frame_input_buttons = ttk.Frame(self)
        frame_input_buttons.grid(row=r, column=2, padx=10, pady=5, sticky='w')

        btn_browse_folder = ttk.Button(frame_input_buttons, text='Folder', command=self.browse_input_folder)
        btn_browse_folder.pack(side=tk.LEFT, padx=(0, 5))
        
        btn_browse_file = ttk.Button(frame_input_buttons, text='File', command=self.browse_input_file)
        btn_browse_file.pack(side=tk.LEFT)
        r += 1

        # Note (simplified)
        note_label = ttk.Label(self, text='Note: Path can be a folder (processes all videos) or a single video file.', font=('Helvetica', 9))
        note_label.grid(row=r, column=0, columnspan=3, padx=10, pady=(0, 5), sticky='w')
        r += 1

        # Output Directory
        lbl_out_dir = ttk.Label(self, text='Output Directory')
        lbl_out_dir.grid(row=r, column=0, padx=10, pady=5, sticky='w')
        ToolTip(lbl_out_dir, "Specifies the directory where the output depth maps will be saved.")
        
        entry_out_dir = ttk.Entry(self, textvariable=self.output_dir_var)
        entry_out_dir.grid(row=r, column=1, padx=5, pady=5, sticky='ew')
        
        btn_browse_out = ttk.Button(self, text='Browse', command=self.browse_output_dir)
        btn_browse_out.grid(row=r, column=2, padx=10, pady=5)
        r += 1

        # Encoder
        lbl_encoder = ttk.Label(self, text='Encoder')
        lbl_encoder.grid(row=r, column=0, padx=10, pady=5, sticky='w')
        ToolTip(lbl_encoder, "Specifies the encoder to use. Use vits for Small, vitb for Base, and vitl for Large.")
        
        combo_encoder = ttk.Combobox(self, textvariable=self.encoder_var, values=['vits', 'vitb', 'vitl'], state='readonly')
        combo_encoder.grid(row=r, column=1, columnspan=2, padx=10, pady=5, sticky='w')
        r += 1

        # Advanced Settings Frame
        frame_advanced = ttk.LabelFrame(self, text='Advanced Settings', padding="10")
        frame_advanced.grid(row=r, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

        # Input Size
        lbl_in_size = ttk.Label(frame_advanced, text='Input Size')
        lbl_in_size.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ToolTip(lbl_in_size, "Input size for model inference. Default is 518.")
        
        entry_in_size = ttk.Entry(frame_advanced, textvariable=self.input_size_var, width=10)
        entry_in_size.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Max Resolution
        lbl_max_res = ttk.Label(frame_advanced, text='Max Resolution')
        lbl_max_res.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        ToolTip(lbl_max_res, "Maximum resolution for model inference. Default is 1280.")
        
        entry_max_res = ttk.Entry(frame_advanced, textvariable=self.max_res_var, width=10)
        entry_max_res.grid(row=0, column=3, padx=5, pady=5, sticky='w')

        # Max Length 
        lbl_max_len = ttk.Label(frame_advanced, text='Max Length (-1=No Limit)')
        lbl_max_len.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ToolTip(lbl_max_len, "Maximum length of the input video in frames. -1 means no limit.")
        
        entry_max_len = ttk.Entry(frame_advanced, textvariable=self.max_len_var, width=10)
        entry_max_len.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Target FPS
        lbl_target_fps = ttk.Label(frame_advanced, text='Target FPS (-1=Original)')
        lbl_target_fps.grid(row=1, column=2, padx=5, pady=5, sticky='w')
        ToolTip(lbl_target_fps, "Target frames per second for processing. -1 means original FPS.")
        
        entry_target_fps = ttk.Entry(frame_advanced, textvariable=self.target_fps_var, width=10)
        entry_target_fps.grid(row=1, column=3, padx=5, pady=5, sticky='w')
        # The 'r' variable is already incremented later. No need to increment it here.

        r += 1 # Increment r after the Advanced Settings Frame

        # Metric Model Checkbox
        check_metric = ttk.Checkbutton(self, text='Use Metric Model', variable=self.metric_var)
        check_metric.grid(row=r, column=0, columnspan=3, padx=10, pady=5, sticky='w')
        ToolTip(check_metric, "Use metric depth models trained on Virtual KITTI and IRS datasets.")
        
        # Invert Metric Output Checkbox (Column 2)
        check_invert_metric = ttk.Checkbutton(self, text='Invert Metric Output', variable=self.invert_metric_var)
        check_invert_metric.grid(row=r, column=2, columnspan=1, padx=10, pady=5, sticky='w')
        ToolTip(check_invert_metric, "Reverse the grayscale/color mapping for the visual depth map (closer=lighter -> closer=darker).")
        r += 1 # Increment r after the Metric Checkbox
        
        # Checkboxes Row 1 (FP32, TTA)
        check_fp32 = ttk.Checkbutton(self, text='Use fp32 precision', variable=self.fp32_var)
        check_fp32.grid(row=r, column=0, columnspan=2, padx=10, pady=5, sticky='w')
        ToolTip(check_fp32, "Use 32-bit floating point precision for inference. Default is 16-bit.")
        
        check_tta = ttk.Checkbutton(self, text='Enable TTA', variable=self.tta_var)
        check_tta.grid(row=r, column=2, padx=10, pady=5, sticky='w')
        ToolTip(check_tta, "Enable Test-Time Augmentation (horizontal flipping) for potentially improved quality (slower).")
        r += 1

        # Checkboxes Row 2 (Save Color, Create Source)
        check_color = ttk.Checkbutton(self, text='Save color depth map', variable=self.save_color_var)
        check_color.grid(row=r, column=0, columnspan=2, padx=10, pady=5, sticky='w')
        ToolTip(check_color, "Save depth maps with color palette instead of grayscale.")
        
        check_src = ttk.Checkbutton(self, text='Create source clip', variable=self.create_src_var)
        check_src.grid(row=r, column=2, padx=10, pady=5, sticky='w')
        ToolTip(check_src, "Create a source clip alongside the depth map.")
        r += 1

        # Checkboxes Row 3 (NPZ, EXR)
        check_npz = ttk.Checkbutton(self, text='Save depth as npz', variable=self.save_npz_var)
        check_npz.grid(row=r, column=0, columnspan=2, padx=10, pady=5, sticky='w')
        ToolTip(check_npz, "Save depth maps in .npz format.")
        
        check_exr = ttk.Checkbutton(self, text='Save depth as exr', variable=self.save_exr_var)
        check_exr.grid(row=r, column=2, padx=10, pady=5, sticky='w')
        ToolTip(check_exr, "Save depth maps in .exr format.")
        r += 1

        # Checkboxes Row 4 (Save PNG, 16-bit PNG)
        check_save_png = ttk.Checkbutton(self, text='Save depth maps as PNG sequence', variable=self.save_png_var)
        check_save_png.grid(row=r, column=0, columnspan=2, padx=10, pady=5, sticky='w')
        ToolTip(check_save_png, "Save each depth map frame as an 8-bit or 16-bit PNG file.")
        
        check_16bit_png = ttk.Checkbutton(self, text='Use 16-bit PNG', variable=self.png_16bit_var)
        check_16bit_png.grid(row=r, column=2, padx=10, pady=5, sticky='w')
        ToolTip(check_16bit_png, "Save PNGs as 16-bit (default is 8-bit).")
        r += 1

        # PNG Compression and MP4 CRF Sliders
        # PNG Compression
        lbl_png_comp = ttk.Label(self, text='PNG Compression Level')
        lbl_png_comp.grid(row=r, column=0, padx=10, pady=5, sticky='w')
        ToolTip(lbl_png_comp, "(0-9, 0=no compression)")
        
        slider_png_comp = ttk.Scale(self, from_=0, to=9, orient=tk.HORIZONTAL, variable=self.png_compression_var, length=150)
        slider_png_comp.grid(row=r, column=1, padx=5, pady=5, sticky='ew')
        
        # MP4 CRF
        lbl_mp4_crf = ttk.Label(self, text='MP4 CRF')
        lbl_mp4_crf.grid(row=r+1, column=0, padx=10, pady=5, sticky='w')
        ToolTip(lbl_mp4_crf, "(0=lossless; higher numbers lower quality\nDefault=18)")
        
        slider_mp4_crf = ttk.Scale(self, from_=0, to=51, orient=tk.HORIZONTAL, variable=self.mp4_crf_var, length=150)
        slider_mp4_crf.grid(row=r+1, column=1, padx=5, pady=5, sticky='ew')
        
        # Current slider values (Optional: display current value next to slider)
        ttk.Label(self, textvariable=self.png_compression_var).grid(row=r, column=2, padx=5, pady=5, sticky='w')
        ttk.Label(self, textvariable=self.mp4_crf_var).grid(row=r+1, column=2, padx=5, pady=5, sticky='w')
        r += 2

        # Resume Checkbox
        check_resume = ttk.Checkbutton(self, text='Resume', variable=self.resume_var)
        check_resume.grid(row=r, column=0, columnspan=3, padx=10, pady=5, sticky='w')
        ToolTip(check_resume, "Move completed files to \"finished\" folder for easy resuming.")
        r += 1

        # Buttons
        self.btn_start = ttk.Button(self, text='Process', command=self.start_processing)
        self.btn_start.grid(row=r, column=0, padx=10, pady=10, sticky='w')
        
        self.btn_stop = ttk.Button(self, text='Stop', command=self.stop_processing, state=tk.DISABLED)
        self.btn_stop.grid(row=r, column=1, padx=5, pady=10, sticky='w')
        
        btn_exit = ttk.Button(self, text='Exit', command=self.on_closing)
        btn_exit.grid(row=r, column=2, padx=10, pady=10, sticky='e')
        r += 1

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self, orient='horizontal', mode='determinate', length=400)
        self.progress_bar.grid(row=r, column=0, columnspan=3, padx=10, pady=5, sticky='ew')
        r += 1

        # Status Label
        self.status_var = tk.StringVar(value='Status: Idle')
        self.status_label = ttk.Label(self, textvariable=self.status_var)
        self.status_label.grid(row=r, column=0, columnspan=3, padx=10, pady=5, sticky='w')
        r += 1
        
        # Padding
        for child in self.winfo_children():
            child.grid_configure(padx=2, pady=2)
            # Removed the inner loop because frame_input_buttons uses pack for its children,
            # which conflicts with the grid_configure call.
            if isinstance(child, ttk.LabelFrame): 
                # Keep configuration for LabelFrames that use grid internally (like Advanced Settings)
                for sub_child in child.winfo_children():
                    sub_child.grid_configure(padx=2, pady=2)

    def browse_input_folder(self):
        folder_path = filedialog.askdirectory(title="Select Input Folder")
        if folder_path:
            self.input_path_var.set(folder_path) # Changed to input_path_var

    def browse_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Single Video File",
            filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*"))
        )
        if file_path:
            self.input_path_var.set(file_path)

    def browse_output_dir(self):
        folder_path = filedialog.askdirectory(title="Select Output Directory")
        if folder_path:
            self.output_dir_var.set(folder_path)

    def start_processing(self):
        # Validate inputs before starting the thread
        try:
            int(self.input_size_var.get())
            int(self.max_res_var.get())
        except ValueError:
            messagebox.showerror('Input Error', 'Please enter valid integers for Input Size and Max Resolution')
            return

        input_path = self.input_path_var.get()
        output_dir = self.output_dir_var.get()

        if not input_path:
            messagebox.showerror('Input Error', 'Please select an input file or folder')
            return

        if not output_dir:
            messagebox.showerror('Input Error', 'Please select an output directory')
            return
            
        # Determine if the input path is a file or a folder
        is_file = os.path.isfile(input_path)
        
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.stop_event.clear()

        self.set_input_widgets_state('disabled')
        
        # Collect all values to pass to the thread (mimics PySimpleGUI's 'values' dictionary)
        values = {
            '-INPUT_FOLDER-': input_path if not is_file else '', # Only set folder if it's not a file
            '-INPUT_FILE-': input_path if is_file else '',       # Only set file if it is a file
            '-OUTPUT_DIR-': output_dir,
            '-ENCODER-': self.encoder_var.get(),
            '-INPUT_SIZE-': self.input_size_var.get(),
            '-MAX_RES-': self.max_res_var.get(),
            '-MAX_LEN-': self.max_len_var.get(),
            '-TARGET_FPS-': self.target_fps_var.get(),
            '-FP32-': self.fp32_var.get(),
            '-SAVE_COLOR-': self.save_color_var.get(),
            '-SAVE_NPZ-': self.save_npz_var.get(),
            '-SAVE_EXR-': self.save_exr_var.get(),
            '-CREATE_SRC-': self.create_src_var.get(),
            '-TTA-': self.tta_var.get(),
            '-RESUME-': self.resume_var.get(),
            '-SAVE_PNG-': self.save_png_var.get(),
            '-PNG_16BIT-': self.png_16bit_var.get(),
            '-PNG_COMPRESSION-': self.png_compression_var.get(),
            '-METRIC-': self.metric_var.get(),
            '-INVERT_METRIC-': self.invert_metric_var.get(),
            '-MP4_CRF-': self.mp4_crf_var.get(),
            '-INVERT_METRIC-': self.invert_metric_var.get(),
        }

        self.processing_thread = threading.Thread(
            target=self.process_videos_threaded, 
            args=(values, self.stop_event)
        )
        self.processing_thread.daemon = True # Daemonize thread
        self.processing_thread.start()
        self.status_var.set('Status: Starting...')

    def stop_processing(self):
        self.stop_event.set()
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set('Status: Stopping... please wait.')

    def check_queue(self):
        """Poll the queue for updates from the worker thread."""
        try:
            while True:
                key, value = self.update_queue.get_nowait()
                if key == '-SET_MAX-':
                    self.progress_bar.config(maximum=value, value=0)
                elif key == '-PROGRESS_UPDATE-':
                    self.progress_bar.config(value=value)
                elif key == '-STATUS_UPDATE-':
                    self.status_var.set(f'Status: {value}')
                elif key == '-ERROR-':
                    messagebox.showerror('Processing Error', value)
                elif key == '-THREAD_DONE-':
                    self.on_thread_done(value)
                self.update_queue.task_done()
        except Empty:
            pass # No updates yet
        
        # Schedule the next check
        self.after(100, self.check_queue)

    def on_thread_done(self, message):
        """Called when the worker thread finishes."""
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.set_input_widgets_state('normal')
        self.status_var.set(f'Status: Processing {message}')
        self.progress_bar.config(value=self.progress_bar['maximum'])
        self.processing_thread = None

    def on_closing(self):
        """Saves settings and stops the thread before exiting."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            # Wait a moment for the thread to stop gracefully (optional)
            self.processing_thread.join(timeout=1)
        self.save_settings()
        self.destroy()
        
    def process_videos_threaded(self, values, stop_event):
        """
        Original process_videos logic running in a separate thread.
        Uses self.update_queue.put(('KEY', value)) for thread-safe GUI updates.
        """
        
        # Helper function to send updates back to the main thread
        def write_event_value(key, value):
            self.update_queue.put((key, value))

        # Retrieve and validate inputs
        input_folder = values['-INPUT_FOLDER-']
        input_file = values['-INPUT_FILE-']

        # If both are somehow set, prioritize the file for safety.
        if input_file:
            input_folder = ''
        elif input_folder and not os.path.isdir(input_folder):
            # If input_folder is set but is actually a file, treat it as a file
            if os.path.isfile(input_folder):
                input_file = input_folder
                input_folder = ''
            else:
                write_event_value('-ERROR-', f'Input path must be a valid file or directory: {input_folder}')
                return
            
        output_dir = values['-OUTPUT_DIR-']
        encoder = values['-ENCODER-']
        
        try:
            input_size = int(values['-INPUT_SIZE-'])
            max_res = int(values['-MAX_RES-'])
        except ValueError:
            # Should have been caught by start_processing, but good practice to handle here too
            write_event_value('-ERROR-', 'Internal Error: Invalid Input Size or Max Resolution type.')
            return
            
        max_len = values['-MAX_LEN-']
        target_fps = values['-TARGET_FPS-']
        fp32 = values['-FP32-']
        grayscale = not values['-SAVE_COLOR-']
        create_src = values['-CREATE_SRC-']
        save_npz = values['-SAVE_NPZ-']
        save_exr = values['-SAVE_EXR-']
        tta = values['-TTA-']
        resume = values['-RESUME-']
        save_png = values['-SAVE_PNG-']
        png_16bit = values['-PNG_16BIT-']
        png_compression = int(values['-PNG_COMPRESSION-'])
        mp4_crf = int(values['-MP4_CRF-'])
        metric = values['-METRIC-']
        invert_metric = values['-INVERT_METRIC-']

        # Determine files to process
        if input_file:
            files_to_process = [input_file]  # Single file selected
        elif input_folder:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = [
                f for f in os.listdir(input_folder)
                if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(tuple(video_extensions))
            ]
            if not video_files:
                write_event_value('-ERROR-', 'No video files found in the input folder')
                return
            files_to_process = [os.path.join(input_folder, f) for f in video_files]
        else:
            # Should have been caught by start_processing
            write_event_value('-ERROR-', 'Logic Error: No input selected.')
            return

        # Initialize device and model
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        write_event_value('-STATUS_UPDATE-', 'Loading model...')
        
        checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
        # NOTE: video_depth_anything, torch, numpy, etc. must be installed for this to run
        try:
            video_depth_anything = VideoDepthAnything(**model_configs[encoder], metric=metric) # PASS METRIC FLAG
            video_depth_anything.load_state_dict(                                                                                
                torch.load(f'./checkpoints/{checkpoint_name}_{encoder}.pth', map_location='cpu'), strict=True # USE CHECKPOINT_NAME
            )
            video_depth_anything = video_depth_anything.to(DEVICE).eval()
        except FileNotFoundError:
            write_event_value('-ERROR-', f'Checkpoint file for {encoder} not found in ./checkpoints/')
            return
        except Exception as e:
            write_event_value('-ERROR-', f'Error loading model: {str(e)}')
            return
            

        # Process files
        total_files = len(files_to_process)
        write_event_value('-SET_MAX-', total_files)
        for i, file_path in enumerate(files_to_process):
            if stop_event.is_set():
                write_event_value('-THREAD_DONE-', 'Stopped')
                return
            write_event_value('-STATUS_UPDATE-', f'Processing file {i+1} of {total_files}: {os.path.basename(file_path)}')
            
            # Read video frames (upscaling uses Lanczos via dc_utils.py)
            try:                                                                                                    
                frames, actual_target_fps = read_video_frames(file_path, int(max_len), int(target_fps), int(max_res))
            except Exception as e:
                write_event_value('-ERROR-', f'Error reading video frames for {os.path.basename(file_path)}: {str(e)}')
                write_event_value('-PROGRESS_UPDATE-', i + 1)
                continue # Skip to next file
                
            # Convert list of frames to a NumPy array
            frames_array = np.stack(frames, axis=0)
            
            # Infer depth with TTA if enabled
            try:
                if tta:
                    # Original depths
                    depths_original, fps = video_depth_anything.infer_video_depth(
                        frames_array, actual_target_fps, input_size=input_size, device=DEVICE, fp32=fp32
                    )
                    # Flip frames horizontally
                    flipped_frames = [np.flip(frame, axis=1) for frame in frames]
                    flipped_frames_array = np.stack(flipped_frames, axis=0)
                    # Predict depths for flipped frames
                    depths_flipped, _ = video_depth_anything.infer_video_depth(
                        flipped_frames_array, actual_target_fps, input_size=input_size, device=DEVICE, fp32=fp32
                    )
                    # Flip the flipped depths back
                    depths_flipped_back = [np.flip(depth, axis=1) for depth in depths_flipped]
                    # Average the depths and convert to NumPy array
                    depths_list = [(d_orig + d_flip) / 2 for d_orig, d_flip in zip(depths_original, depths_flipped_back)]
                    depths = np.stack(depths_list, axis=0)
                else:
                    depths, fps = video_depth_anything.infer_video_depth(
                        frames_array, actual_target_fps, input_size=input_size, device=DEVICE, fp32=fp32
                    )
            except Exception as e:
                write_event_value('-ERROR-', f'Error during depth inference for {os.path.basename(file_path)}: {str(e)}')
                write_event_value('-PROGRESS_UPDATE-', i + 1)
                continue

            # Save outputs
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            video_name = os.path.basename(file_path)
            base_name_no_ext = os.path.splitext(video_name)[0]
            
            try:
                processed_video_path = os.path.join(output_dir, base_name_no_ext + '_src.mp4')
                depth_vis_path = os.path.join(output_dir, base_name_no_ext + '_depth.mp4')
                
                if create_src:
                    save_video(frames, processed_video_path, fps=fps, crf=mp4_crf)
                
                # Create a copy for visualization. The raw depths should not be modified.
                depths_vis = depths.copy()
                
                # Apply inversion for visualization only if the user checked the 'Invert' box
                if invert_metric:
                    # Invert the depth values (max - current) to flip the visualization
                    # Note: We only do this for the *visualization*, not for npz/exr/png raw data.
                    depths_vis = depths_vis.max() - depths_vis
                
                save_video(depths_vis, depth_vis_path, fps=fps, is_depths=True, grayscale=grayscale, crf=mp4_crf)

                if save_npz:
                    depth_npz_path = os.path.join(output_dir, base_name_no_ext + '_depths.npz')
                    np.savez_compressed(depth_npz_path, depths=depths)

                if save_exr:
                    depth_exr_dir = os.path.join(output_dir, base_name_no_ext + '_depths_exr')
                    os.makedirs(depth_exr_dir, exist_ok=True)
                    # NOTE: OpenEXR dependency must be available
                    import OpenEXR
                    import Imath
                    for j, depth in enumerate(depths):
                        output_exr = f"{depth_exr_dir}/frame_{j:05d}.exr"
                        header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                        header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
                        exr_file = OpenEXR.OutputFile(output_exr, header)
                        exr_file.writePixels({"Z": depth.tobytes()})
                        exr_file.close()

                # Save depth maps as PNG sequence if enabled (8-bit default, 16-bit optional)
                if save_png:
                    depth_png_dir = os.path.join(output_dir, base_name_no_ext + '_depth_png')
                    os.makedirs(depth_png_dir, exist_ok=True)
                    d_min = depths.min()
                    d_max = depths.max()
                    for j, depth in enumerate(depths):
                        if png_16bit:
                            depth_norm = (depth - d_min) / (d_max - d_min) * 65535
                            depth_img = depth_norm.astype(np.uint16)
                        else:
                            depth_norm = (depth - d_min) / (d_max - d_min) * 255
                            depth_img = depth_norm.astype(np.uint8)
                        output_png = f"{depth_png_dir}/frame_{j:05d}.png"
                        compression_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
                        # NOTE: cv2 dependency must be available
                        cv2.imwrite(output_png, depth_img, compression_params)

                # Move files after processing to finished folder if Resume is checked
                if resume:
                    # Determine the source directory of the file being processed
                    file_src_dir = os.path.dirname(file_path)
                    
                    finished_dir = os.path.join(file_src_dir, "finished")
                    if not os.path.exists(finished_dir):
                        os.makedirs(finished_dir)
                    try:
                        shutil.move(file_path, os.path.join(finished_dir, os.path.basename(file_path)))
                    except Exception as e:
                        write_event_value('-ERROR-', f'Failed to move {file_path} to finished folder: {str(e)}')
                        
            except Exception as e:
                write_event_value('-ERROR-', f'Error saving outputs for {os.path.basename(file_path)}: {str(e)}')
                
            write_event_value('-PROGRESS_UPDATE-', i + 1)

        # --- Cleanup ---
        # Explicitly delete the model and free VRAM
        try:
            del video_depth_anything
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            # Also call garbage collector
            import gc
            gc.collect()
        except Exception as e:
            logging.error(f"Error during VRAM cleanup: {e}")

        write_event_value('-THREAD_DONE-', 'Completed')

    def set_input_widgets_state(self, state):
        """Sets the state ('normal' or 'disabled') of all input widgets."""
        if state not in ['normal', 'disabled']:
            return

        # Find all relevant widgets
        widgets_to_control = []
        
        # Collect entry, combobox, scale, and checkbutton widgets
        for child in self.winfo_children():
            if isinstance(child, (ttk.Entry, ttk.Combobox, ttk.Checkbutton, ttk.Scale)):
                widgets_to_control.append(child)
            elif isinstance(child, (ttk.Frame, ttk.LabelFrame)):
                # Also collect widgets inside advanced settings frame
                widgets_to_control.extend(
                    c for c in child.winfo_children() 
                    if isinstance(c, (ttk.Entry, ttk.Combobox, ttk.Checkbutton, ttk.Scale))
                )

        # Explicitly include the input path buttons as they are in a Frame
        # We need to find the frame_input_buttons and get its children
        for child in self.winfo_children():
            if child.winfo_class() == 'Frame' and child.winfo_name().startswith('!frame'): # Crude way to find the anonymous Frame
                 # Assuming the input buttons frame is the first anonymous Frame after the title
                if child.winfo_children():
                    first_child = child.winfo_children()[0]
                    # Check if the first child is a Button (a better heuristic)
                    if isinstance(first_child, ttk.Button) and first_child.cget('text') in ('Folder', 'File'):
                        widgets_to_control.extend(child.winfo_children())
                        break
        
        # Set state
        for widget in widgets_to_control:
            try:
                # Checkbuttons and Scales use .config(state=...), Entries and Comboboxes use .config(state=...)
                # Scales do not have a .cget('state') that always works, so use a try-except
                widget.config(state=state)
            except Exception:
                pass # Ignore widgets that don't support 'state' config

if __name__ == '__main__':

    # Ensure a basic structure for external dependencies if not running in the actual environment
    # This prevents the script from crashing immediately if imports fail, though processing will not work.
    try:
        from video_depth_anything.video_depth import VideoDepthAnything
        from utils.dc_utils import read_video_frames, save_video
    except ImportError:
        # Placeholder/dummy functions if the actual libraries are not installed
        def VideoDepthAnything(*args, **kwargs):
            raise RuntimeError("video_depth_anything library not found. Install required dependencies.")
        def read_video_frames(*args):
            raise RuntimeError("utils.dc_utils library not found. Install required dependencies.")
        def save_video(*args, **kwargs):
            raise RuntimeError("utils.dc_utils library not found. Install required dependencies.")
        
        # Display a warning, but let the GUI start
        logging.warning("Required external libraries (torch, cv2, video_depth_anything, utils) were not found.")
        logging.warning("The GUI will display, but the 'Process' function will likely fail.")


    app = VideoDepthAnythingGUI()
    app.mainloop()