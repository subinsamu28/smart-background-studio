# ─────────────────────────────────────────────
# ✅ Standard Library Imports
# ─────────────────────────────────────────────

import sys        # Provides access to system-specific parameters and functions
import os         # Enables interaction with the operating system (e.g., paths, directories)
import subprocess # Allows running system shell commands and managing external processes

# ─────────────────────────────────────────────
# ✅ Third-Party Library Imports
# ─────────────────────────────────────────────

import cv2        # OpenCV library for computer vision tasks (image and video processing)
import torch      # PyTorch library for deep learning model handling
import vlc        # Python bindings for VLC media player (used for video playback)
import numpy as np # NumPy for numerical operations, used for image arrays
from PIL import Image  # Python Imaging Library (Pillow) for working with image files

# ─────────────────────────────────────────────
# ✅ Ensuring Root Directory is in the Python Path
# ─────────────────────────────────────────────

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root directory to sys.path so Python can import modules from other folders
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ─────────────────────────────────────────────
# ✅ Backend Module Imports (Custom Functions)
# ─────────────────────────────────────────────

# Print statements to indicate imports for debugging/logging purposes
print("🛠️ Calling image processing function: from backend.processor import run_blur_background")
print("🛠️ Calling image processing function: from backend.processor import run_u2net")
print("🛠️ Calling image processing function: from backend.processor import run_solid_background")
print("🛠️ Calling image processing function: from backend.processor import run_background_replace")

# Import image processing functions from the custom backend.processor module
from backend.processor import (
    run_blur_background,      # Applies Gaussian blur to the background of an image
    run_u2net,                # Runs U2-Net model for background segmentation
    run_solid_background,     # Replaces the background with a solid color
    run_background_replace,   # Replaces the background with a chosen image
    preprocess_image,         # Prepares image data for model input
    postprocess_mask,         # Converts model output to usable segmentation mask
    load_model                # Loads the trained deep learning model into memory
)

# ─────────────────────────────────────────────
# ✅ PyQt5 GUI Module Imports
# ─────────────────────────────────────────────

# Importing essential UI widgets used to create the GUI application
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QComboBox, QHBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QColorDialog, QCheckBox, QSlider, QProgressBar, QSpinBox
)

# Importing core functionality from PyQt5 (event system, timing, animations)
from PyQt5.QtCore import (
    Qt, QTimer, QTime, QObject, QRunnable, pyqtSignal, pyqtSlot,
    QThreadPool, QUrl, QPropertyAnimation, QEasingCurve
)

# Importing GUI components for displaying images, colors, and animations
from PyQt5.QtGui import (
    QPixmap, QImage, QPalette, QBrush, QColor, QMovie
)

# Importing PyQt5 multimedia classes for playing videos and audio
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget



# Define a class to hold custom signals for our worker thread
class WorkerSignals(QObject):
    # Signal emitted when the task is complete (no data sent)
    finished = pyqtSignal()
    
    # Signal emitted when the task has a result (passes the result object)
    result = pyqtSignal(object)
    
    # Signal emitted when an error occurs (passes a tuple: exception object and traceback)
    error = pyqtSignal(tuple)


# Define a worker that runs a function in a separate thread
class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        print("🔍 Entered function: __init__")
        super(Worker, self).__init__()
        
        # Store the function to be executed and its arguments
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        
        # Create a signals object to communicate back with the main thread
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        print("🔍 Entered function: run")
        try:
            # Try to execute the function with the provided arguments
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            import traceback
            # If an error occurs, emit the error signal with details
            self.signals.error.emit((e, traceback.format_exc()))
        else:
            # If the function executes successfully, emit the result
            self.signals.result.emit(result)
        finally:
            # Always emit the finished signal to indicate the task is done
            self.signals.finished.emit()


class SmartBackgroundApp(QWidget):
    def __init__(self):
        print("🔍 Entered function: __init__")
        super().__init__()

        # Create a thread pool to manage background tasks without freezing the UI
        self.threadpool = QThreadPool()

        # Set up basic window properties
        self.setWindowTitle("Smart Background Studio")  # Title shown on the window
        self.setGeometry(100, 100, 1200, 700)  # Position and size (x, y, width, height)
        self.setAcceptDrops(True)  # Enable drag-and-drop support
        self.setWindowOpacity(0.0)  # Start with window fully transparent (for fade-in)

        # Initialize app state variables
        self.blur_intensity = 5  # Default blur intensity for background
        self.bg_image_path = None  # Path to the background image
        self.selected_bg_image = None  # The image currently selected for the background
        self.brightness_value = 0  # Brightness adjustment value
        self.recording = False  # Flag to track if recording is in progress
        self.paused = False  # Flag to track if recording is paused
        self.out = None  # Placeholder for video writer object (e.g., cv2.VideoWriter)

        # Variables to handle batch image processing
        self.image_files = []  # List of image file paths for batch processing
        self.current_image_index = -1  # Index of the current image being processed

        # Load the AI model for background processing once during startup
        self.net = load_model()

        # Set up all UI components
        self.init_ui()

        # Start a smooth fade-in effect for the window
        self.fade_in()

    def fade_in(self):
        print("🔍 Entered function: fade_in")
        # Create a property animation for the window's opacity
        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setDuration(1000)  # Animation duration in milliseconds (1 second)
        self.anim.setStartValue(0.0)  # Start fully transparent
        self.anim.setEndValue(1.0)  # End fully opaque
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)  # Smooth transition curve
        self.anim.start()  # Start the animation

    def simulate_progress(self):
        print("🔍 Entered function: simulate_progress")
        # Simulate a progress bar update over time using QTimer
        for i in range(101):
            # Each iteration sets a timer to update the progress bar value
            QTimer.singleShot(i * 10, lambda val=i: self.progress_bar.setValue(val))
            # Use lambda to capture current value of i correctly

    def check_batch_mode(self, mode):
        print("🔍 Entered function: check_batch_mode")
        # Check if the selected mode is "Batch Processing"
        if mode == "Batch Processing":
            # If it is, trigger the batch processing workflow
            self.run_batch_processing()
            
    def init_ui(self):
        print("🔍 Entered function: init_ui")

        # Set up the preview area using QGraphicsView and QGraphicsScene
        self.preview = QGraphicsView()  # Viewport for displaying video or image content
        self.scene = QGraphicsScene()  # Scene to manage visual elements in the preview
        self.preview.setScene(self.scene)
        
        # Style the preview area: transparent background with a subtle border
        self.preview.setStyleSheet("background: transparent; border: 1px solid #444;")

        # Set a placeholder background image for the preview area
        self.preview.setBackgroundBrush(QBrush(QPixmap("assets/preview_bg.png")))

        # Apply a sleek, dark-themed UI style for the whole window
        self.setStyleSheet("background-color: #121212; color: white; font-family: 'Segoe UI';")

        # Label for animated background effect (initially hidden and layered behind other widgets)
        self.bg_animation_label = QLabel(self)
        self.bg_animation_label.setGeometry(self.preview.geometry())  # Match preview size
        self.bg_animation_label.setStyleSheet("background: transparent;")
        self.bg_animation_label.lower()  # Push behind other UI elements

        # Load and assign a looping animation (e.g., a subtle background effect)
        self.bg_movie = QMovie("ui/animations/preview_background.gif")
        self.bg_animation_label.setMovie(self.bg_movie)
        self.bg_animation_label.setVisible(False)  # Start hidden; can be shown when needed

        # Button for selecting an image file
        self.upload_btn = QPushButton("📂 Select Image")
        self.upload_btn.clicked.connect(self.select_image)  # Connect button to file picker
        self.upload_btn.setStyleSheet(self.futuristic_button_style())  # Apply custom styling

        # Checkbox to enable/disable mirroring the video stream
        self.mirror_checkbox = QCheckBox("Mirror Video")
        self.mirror_checkbox.setChecked(False)  # Default is no mirroring

        # Color picker button (initially hidden)
        self.color_picker_btn = QPushButton("Pick Background Color")
        self.color_picker_btn.setStyleSheet(self.futuristic_button_style())  # Apply custom button styling
        self.color_picker_btn.clicked.connect(self.pick_color)  # Connect to method that handles color selection
        self.color_picker_btn.setVisible(False)  # Start hidden until needed

        # Navigation buttons for batch image mode
        self.prev_btn = QPushButton("⬅️ Previous")
        self.prev_btn.clicked.connect(self.show_previous_image)  # Go to previous image
        self.prev_btn.setStyleSheet(self.futuristic_button_style())
        self.prev_btn.setVisible(False)  # Hidden unless in batch processing mode

        self.next_btn = QPushButton("Next ➡️")
        self.next_btn.clicked.connect(self.show_next_image)  # Go to next image
        self.next_btn.setStyleSheet(self.futuristic_button_style())
        self.next_btn.setVisible(False)  # Hidden unless in batch processing mode

        # Default selected background color (RGB format) – white
        self.selected_color = (255, 255, 255)

        # Setup for recording timer functionality
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_time)  # Timer ticks every second or interval to update UI
        self.recording_time = QTime(0, 0, 0)  # Start recording time from 00:00

        # Recording indicator label
        self.recording_label = QLabel("● REC 00:00")
        self.recording_label.setStyleSheet(
            "color: red; font-weight: bold; font-size: 14px; padding: 4px;"
        )  # Style to make it look like a real recorder indicator
        self.recording_label.setVisible(False)  # Hidden by default until recording starts

        # Dropdown menu to choose different operational modes
        self.mode_selector = QComboBox()
        self.mode_selector.addItems([
            "Transparent PNG Output",           # Export with transparent background
            "Solid Color Background",           # Replace background with a flat color
            "Background Replacement",           # Use a custom background image
            "Real-time Webcam",                 # Apply effects to live webcam feed
            "Batch Processing",                 # Process multiple images in a batch
            "Smart Resize + Aspect Lock",       # Resize with aspect ratio constraints
        ])

        # Apply consistent, modern styling to the dropdown
        self.mode_selector.setStyleSheet(self.combo_box_style())

        # React to mode changes: adjust the visible UI controls accordingly
        self.mode_selector.currentTextChanged.connect(self.toggle_ui_controls)

        # Also check if the selected mode activates batch processing logic
        self.mode_selector.currentTextChanged.connect(self.check_batch_mode)

        # Button to initiate the image processing action
        self.run_btn = QPushButton("🚀 Process Image")
        self.run_btn.clicked.connect(self.run_model)  # Connect to the main model execution method
        self.run_btn.setStyleSheet(self.futuristic_button_style())  # Apply custom futuristic styling

        # Animated background label to add a visual flair to the preview area
        self.bg_animation_label = QLabel(self)
        self.bg_animation_label.setGeometry(self.preview.geometry())  # Ensure it overlays exactly behind the preview
        self.bg_animation_label.setStyleSheet("background: transparent;")
        self.bg_animation_label.lower()  # Ensure this animation label stays behind other UI components

        # Set up the looping animation using a GIF file
        self.bg_movie = QMovie("ui/animations/preview_background.gif")  # Load the animation
        self.bg_animation_label.setMovie(self.bg_movie)  # Assign it to the label
        self.bg_movie.start()  # Start the animation immediately

        # Ensure the preview remains visually integrated with the animated background
        self.preview.setStyleSheet("background: transparent; border: 1px solid #444;")

        # Set a default background brush in the preview (used when no live video or image is present)
        self.preview.setBackgroundBrush(QBrush(QPixmap("assets/preview_bg.png")))

        # Checkbox to toggle live mode (e.g., real-time webcam input)
        self.live_toggle = QCheckBox("Live")
        self.live_toggle.setChecked(False)  # Default is off
        self.live_toggle.setVisible(False)  # Hidden unless in a mode that supports live input
        self.live_toggle.stateChanged.connect(self.toggle_live_mode)  # Toggle live functionality

        # Checkbox to mirror video input (horizontal flip)
        self.mirror_checkbox = QCheckBox("Mirror Video")
        self.mirror_checkbox.setChecked(False)  # Default is no mirroring
        self.mirror_checkbox.setVisible(False)  # Only shown when live/video preview is active

        # Dropdown to select background style in supported modes
        self.bg_mode_selector = QComboBox()
        self.bg_mode_selector.addItems([
            "Solid Color",              # Flat colored background
            "Image",                    # Use a static image as background
            "Blur",                     # Apply blur to background
            "Transparent PNG Output"    # Export with a transparent background
        ])
        self.bg_mode_selector.currentTextChanged.connect(
            lambda: self.toggle_ui_controls(self.mode_selector.currentText())
        )  # Ensure the main mode selector updates UI accordingly when this value changes
        self.bg_mode_selector.setVisible(False)  # Hidden by default, shown only in relevant modes

        # Slider to adjust background blur intensity
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setMinimum(1)  # Minimum blur level
        self.blur_slider.setMaximum(20)  # Maximum blur level
        self.blur_slider.setValue(5)  # Default value
        self.blur_slider.setVisible(False)  # Hidden until blur mode is selected
        self.blur_slider.valueChanged.connect(self.change_blur_intensity)  # Connect to handler

        # Button to pick a background image file
        self.bg_image_btn = QPushButton("Select Background Image")
        self.bg_image_btn.setStyleSheet(self.futuristic_button_style())  # Apply modern styling
        self.bg_image_btn.clicked.connect(self.pick_background_image)  # Open file dialog
        self.bg_image_btn.setVisible(False)  # Hidden unless background image mode is active

        # Initialize variables used by image and webcam background modes
        self.bg_image_path = None  # Path to selected background image for webcam
        self.selected_bg_image = None  # Image object for static replacement use

        # Slider to adjust brightness of the foreground
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)  # Minimum brightness adjustment
        self.brightness_slider.setMaximum(100)  # Maximum brightness adjustment
        self.brightness_slider.setValue(0)  # Neutral/default brightness
        self.brightness_slider.setVisible(False)  # Hidden unless needed
        self.brightness_slider.valueChanged.connect(self.change_brightness)  # Connect to brightness handler

        # Start recording button
        self.start_record_btn = QPushButton("Start Recording")
        self.start_record_btn.setVisible(False)  # Hidden by default, shown in relevant modes
        self.start_record_btn.clicked.connect(self.start_recording)  # Trigger start recording logic
        self.start_record_btn.setStyleSheet(self.futuristic_button_style())  # Apply consistent custom styling

        # Pause recording button
        self.pause_record_btn = QPushButton("Pause Recording")
        self.pause_record_btn.setVisible(False)  # Hidden until recording starts
        self.pause_record_btn.clicked.connect(self.pause_recording)  # Pause the recording
        self.pause_record_btn.setStyleSheet(self.futuristic_button_style())

        # Stop recording button
        self.stop_record_btn = QPushButton("Stop Recording")
        self.stop_record_btn.setVisible(False)  # Only shown during active recording
        self.stop_record_btn.clicked.connect(self.stop_recording)  # Stop and finalize recording
        self.stop_record_btn.setStyleSheet(self.futuristic_button_style())

        # Snapshot Button – captures the current frame or preview state
        self.snapshot_btn = QPushButton("📸 Snapshot")
        self.snapshot_btn.setVisible(False)  # Hidden by default; shown in relevant modes
        self.snapshot_btn.clicked.connect(self.take_snapshot)  # Connect to snapshot handler

        # Width input for resizing output image or video
        self.resize_width_input = QSpinBox()
        self.resize_width_input.setRange(100, 2000)  # Allow user to set width between 100px and 2000px
        self.resize_width_input.setValue(800)  # Default width value
        self.resize_width_input.setVisible(False)  # Hidden unless resize mode is active

        # Height input for resizing output
        self.resize_height_input = QSpinBox()
        self.resize_height_input.setRange(100, 2000)  # Allow user to set height between 100px and 2000px
        self.resize_height_input.setValue(800)  # Default height value
        self.resize_height_input.setVisible(False)

        # Checkbox to lock aspect ratio during resizing
        self.aspect_lock_checkbox = QCheckBox("Lock Aspect Ratio")
        self.aspect_lock_checkbox.setChecked(True)  # Aspect ratio is locked by default
        self.aspect_lock_checkbox.setVisible(False)  # Only shown when resizing controls are active

        # Apply consistent futuristic styling to all action buttons
        self.start_record_btn.setStyleSheet(self.futuristic_button_style())
        self.pause_record_btn.setStyleSheet(self.futuristic_button_style())
        self.stop_record_btn.setStyleSheet(self.futuristic_button_style())
        self.snapshot_btn.setStyleSheet(self.futuristic_button_style())

        # ✅ Define the top bar layout to organize main controls horizontally
        top_bar = QHBoxLayout()

        # Image selection button
        top_bar.addWidget(self.upload_btn)

        # Dropdown to choose the processing mode
        top_bar.addWidget(self.mode_selector)

        # Main action button to run processing
        top_bar.addWidget(self.run_btn)

        # Background image picker – shown only in "Background Replacement" mode
        top_bar.addWidget(self.bg_image_btn)

        # Live mode toggle – visible only in "Real-time Webcam" mode
        top_bar.addWidget(self.live_toggle)

        # Selector for background type (e.g. solid, blur, image) – also used in live mode
        top_bar.addWidget(self.bg_mode_selector)

        # Blur strength slider – shown when background mode is set to "Blur"
        top_bar.addWidget(self.blur_slider)

        # Mirror video checkbox – specific to webcam mode
        top_bar.addWidget(self.mirror_checkbox)

        # Color picker for solid background mode – visibility controlled by logic
        top_bar.addWidget(self.color_picker_btn)  # ✅ Keep this one last for layout consistency

        # Brightness control – available in several visual modes
        top_bar.addWidget(self.brightness_slider)

        # Recording buttons – shown in webcam/video modes
        top_bar.addWidget(self.start_record_btn)
        top_bar.addWidget(self.pause_record_btn)
        top_bar.addWidget(self.stop_record_btn)

        # Snapshot button – also for webcam mode
        top_bar.addWidget(self.snapshot_btn)

        # Navigation buttons – for batch mode only
        top_bar.addWidget(self.prev_btn)
        top_bar.addWidget(self.next_btn)

        # Resize controls – only shown in "Smart Resize + Aspect Lock" mode
        top_bar.addWidget(self.resize_width_input)
        top_bar.addWidget(self.resize_height_input)
        top_bar.addWidget(self.aspect_lock_checkbox)

        # ✅ Wrap everything inside the main vertical layout
        layout = QVBoxLayout()
        layout.setSpacing(15)  # Add vertical spacing between layout elements
        layout.setContentsMargins(20, 20, 20, 20)  # Add padding around the layout edges

        # Add the top bar containing all control widgets
        layout.addLayout(top_bar)

        # Add the background mode selector just below the top bar (only visible in certain modes)
        layout.addWidget(self.bg_mode_selector)

        # Preview setup
        self.preview = QGraphicsView()
        self.scene = QGraphicsScene()
        self.preview.setScene(self.scene)
        self.preview.setStyleSheet("background: transparent; border: 1px solid #444;")  # Clean, minimal preview frame
        self.preview.setBackgroundBrush(QBrush(QPixmap("assets/preview_bg.png")))  # Placeholder preview background
        layout.addWidget(self.preview)

        # Ensure controls are set according to the currently selected mode
        self.toggle_ui_controls(self.mode_selector.currentText())

        # Add recording label below the preview (shows timer/status when recording)
        layout.addWidget(self.recording_label)

        # Finalize and apply the full layout to the widget
        self.setLayout(layout)

        # ✅ Add Progress Bar just before setting layout
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # Define range of progress (0% to 100%)
        self.progress_bar.setValue(0)  # Initialize progress value to 0 (no progress yet)

        # Apply custom styling to make the progress bar fit the futuristic UI theme
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #1f1f1f;             # Dark border around the progress bar
                border-radius: 5px;                    # Rounded corners for a smoother look
                background-color: #2c2c2c;             # Dark background for the bar container
                text-align: center;                    # Center the percentage text
                color: white;                          # White text for better visibility
            }
            QProgressBar::chunk {
                background-color: #00f0ff;             # Neon blue fill color for progress
                width: 20px;                           # Width of each chunk in the fill (chunk-based fill)
            }
        """)

        layout.addWidget(self.progress_bar)  # Add the styled progress bar to the main vertical layout

        # Bottom Loading Animation - displayed during long operations for visual feedback
        self.loading_label = QLabel()  # QLabel used to show the animated loader
        self.loading_label.setAlignment(Qt.AlignCenter)  # Center-align the loading animation within the label

        # Load the animated GIF for visual feedback while processing
        self.loading_movie = QMovie("ui/animations/preview_background.gif")  # Path to the loader GIF
        self.loading_label.setMovie(self.loading_movie)  # Attach movie (GIF) to the label

        self.loading_label.setVisible(False)  # Initially hidden; shown only during processing states
        layout.addWidget(self.loading_label)  # Add the loading label to the layout

        # Final application of the complete layout to the QWidget
        self.setLayout(layout)  # This line activates the constructed layout on the UI

        # 🔽 Add the media player setup here

        # Create a media player instance for handling video playback
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # Create a video widget to visually display the video output
        self.video_widget = QVideoWidget()

        # Connect the media player's output to the video display widget
        self.media_player.setVideoOutput(self.video_widget)

        # Initially hide the video widget; it will be shown only when needed
        self.video_widget.setVisible(False)

        # Add the video display widget to the main vertical layout (typically below the preview)
        layout.addWidget(self.video_widget)



    def select_image(self):
        print("🔍 Entered function: select_image")

        # Open a file dialog allowing user to pick an image or video file
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select an Image or Video", 
            "", 
            "Images/Videos (*.png *.jpg *.jpeg *.mp4)"
        )

        if file_path:
            # Store the selected file path
            self.current_image = file_path
            
            # Display the file in the preview area
            self.show_preview(file_path)

    def show_preview(self, file_path):
        print("🔍 Entered function: show_preview")

        # Validate the provided file path
        if not file_path:
            print("❌ No file path provided to show_preview.")
            return

        # Clear any existing content from the preview scene
        self.scene.clear()

        # Hide the video widget in case it was previously visible
        self.video_widget.setVisible(False)

        # Retrieve the currently selected mode from the dropdown
        current_mode = self.mode_selector.currentText()

        # Check if the selected file is a video
        if file_path.lower().endswith(".mp4"):
            # Set the selected video file to the media player
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))

            # Make the video display widget visible
            self.video_widget.setVisible(True)

            # Start playing the video
            self.media_player.play()
        else:
            # Load the image into a QPixmap
            pixmap = QPixmap(file_path)

            # If loading failed, log and exit
            if pixmap.isNull():
                print(f"❌ Failed to load image: {file_path}")
                return

            # Scale the image to fit nicely in the preview (maintaining aspect ratio)
            pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)

            # Add the image to the scene for preview
            self.scene.addItem(QGraphicsPixmapItem(pixmap))

        # Determine UI visibility for background image selection
        # Show the background image button if:
        # 1. Mode is "Background Replacement"
        # 2. OR mode is "Batch Processing" and the selected background sub-mode is "Image"
        selected_mode = self.mode_selector.currentText()
        bg_sub_mode = self.bg_mode_selector.currentText()
        self.bg_image_btn.setVisible(
            (selected_mode == "Background Replacement") or 
            (selected_mode == "Batch Processing" and bg_sub_mode == "Image")
)


    def change_brightness(self, value):
        print("🔍 Entered function: change_brightness")
        # Placeholder for logic to adjust brightness of the current image preview
        # `value` ranges from -100 to 100, set by the brightness slider
        # You would typically apply brightness transformation here and update the scene
        # Example (to be implemented): self.update_preview_with_brightness(value)

    def preview_last_result(self, results):
        print("🔍 Entered function: preview_last_result")

        if results:
            # Store the list of processed image paths
            self.processed_images = results
            
            # Reset the current image index to the first result
            self.current_image_index = 0

            # Show the first image in the results list
            self.show_preview(self.processed_images[self.current_image_index])

            # Enable navigation buttons for batch image review
            self.prev_btn.setVisible(True)
            self.next_btn.setVisible(True)

    def run_model(self):
        if not self.image_files and not hasattr(self, 'current_image'):
            print("❌ No image selected.")
            return
        selected_mode = self.mode_selector.currentText()
        print(f"⚙️ Running mode: {selected_mode}")
        self.loading_movie.start()
        self.loading_label.setVisible(True)

        def process():
            print("🔍 Entered function: process")
            try:
                results = []
                if self.image_files:
                    for image_path in self.image_files:
                        output_path = os.path.join("outputs", f"processed_{os.path.basename(image_path)}")
                        if selected_mode == "Transparent PNG Output":
                            result = run_u2net(image_path, output_path)
                        elif selected_mode == "Solid Color Background":
                            result = run_solid_background(image_path, output_path, bg_color=self.selected_color)
                        elif selected_mode == "Background Replacement" and self.selected_bg_image:
                            result = run_background_replace(image_path, self.selected_bg_image, output_path)
                        elif selected_mode == "Batch Processing":
                            bg_mode = self.bg_mode_selector.currentText()
                            if bg_mode == "Solid Color":
                                result = run_solid_background(image_path, output_path, bg_color=self.selected_color)
                            elif bg_mode == "Image" and self.selected_bg_image:
                                result = run_background_replace(image_path, self.selected_bg_image, output_path)
                            elif bg_mode == "Blur":
                                result = run_blur_background(image_path, output_path, blur_intensity=self.blur_intensity)
                            elif bg_mode == "Transparent PNG Output":
                                output_path = os.path.join("outputs", f"{os.path.splitext(os.path.basename(image_path))[0]}_transparent.png")
                                result = run_u2net(image_path, output_path)
                            else:
                                print("❌ Unsupported background mode in Batch Processing.")
                                result = None
                        else:
                            print(f"❌ Unsupported mode: {selected_mode}")
                            result = None
                        if result:
                            results.append(result)
                else:
                    try:
                        if selected_mode == "Transparent PNG Output":
                            output_path = "outputs/result.png"
                            results.append(run_u2net(self.current_image, output_path))
                        elif selected_mode == "Solid Color Background":
                            output_path = "outputs/solid_result.png"
                            results.append(run_solid_background(self.current_image, output_path, bg_color=self.selected_color))
                        elif selected_mode == "Background Replacement":
                            if not self.selected_bg_image:
                                print("❌ No background image selected.")
                                return []
                            output_path = "outputs/replaced_result.png"
                            results.append(run_background_replace(self.current_image, self.selected_bg_image, output_path))
                        elif selected_mode == "Smart Resize + Aspect Lock":
                            from PIL import Image
                            img = Image.open(self.current_image)
                            width = self.resize_width_input.value()
                            height = self.resize_height_input.value()
                            if self.aspect_lock_checkbox.isChecked():
                                original_ratio = img.width / img.height
                                target_ratio = width / height
                                if original_ratio > target_ratio:
                                    height = int(width / original_ratio)
                                else:
                                    width = int(height * original_ratio)
                            img = img.resize((width, height), Image.LANCZOS)
                            output_path = "outputs/resized_result.png"
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            img.save(output_path)
                            results.append(output_path)
                        else:
                            print(f"❌ Unsupported mode: {selected_mode}")
                    except Exception as e:
                        print(f"❌ Error during single image processing: {e}")
                        return []
                return results
            except Exception as e:
                print(f"❌ Error in process thread: {e}")
                return []

        def handle_results(results):
            print("🔍 Entered function: handle_results")
            try:
                if results:
                    self.processed_images = results
                    self.current_image_index = 0
                    for result_path in results:
                        if result_path and os.path.exists(result_path):
                            self.current_image = result_path
                            self.show_preview(result_path)
                            break
                        else:
                            print("⚠️ Output file not found or invalid:", result_path)
            except Exception as e:
                print(f"❌ Error in handle_results: {e}")

        worker = Worker(process)
        worker.signals.result.connect(handle_results)
        worker.signals.finished.connect(self.loading_movie.stop)
        worker.signals.finished.connect(lambda: self.loading_label.setVisible(False))
        self.threadpool.start(worker)
        print("🔍 Entered function: run_model")
        if not self.image_files and not hasattr(self, 'current_image'):
            print("❌ No image selected.")
            return

        selected_mode = self.mode_selector.currentText()
        print(f"⚙️ Running mode: {selected_mode}")

        self.loading_movie.start()
        self.loading_label.setVisible(True)

        def process():
            results = []
            if self.image_files:
                for image_path in self.image_files:
                    output_path = os.path.join("outputs", f"processed_{os.path.basename(image_path)}")

                    if selected_mode == "Transparent PNG Output":
                        result = run_u2net(image_path, output_path)
                    elif selected_mode == "Solid Color Background":
                        result = run_solid_background(image_path, output_path, bg_color=self.selected_color)
                    elif selected_mode == "Background Replacement" and self.selected_bg_image:
                        result = run_background_replace(image_path, self.selected_bg_image, output_path)
                    elif selected_mode == "Batch Processing":
                        bg_mode = self.bg_mode_selector.currentText()
                        if bg_mode == "Solid Color":
                            result = run_solid_background(image_path, output_path, bg_color=self.selected_color)
                        elif bg_mode == "Image" and self.selected_bg_image:
                            result = run_background_replace(image_path, self.selected_bg_image, output_path)
                        elif bg_mode == "Blur":
                            result = run_blur_background(image_path, output_path, blur_intensity=self.blur_intensity)
                        elif bg_mode == "Transparent PNG Output":
                            result = run_u2net(image_path, output_path)
                        else:
                            print("❌ Unsupported background mode in Batch Processing.")
                            result = None
                    else:
                        print(f"❌ Unsupported mode: {selected_mode}")
                        result = None

                    if result:
                        results.append(result)
            else:
                # Single image processing
                try:
                    if selected_mode == "Transparent PNG Output":
                        output_path = "outputs/result.png"
                        results.append(run_u2net(self.current_image, output_path))
                    elif selected_mode == "Solid Color Background":
                        output_path = "outputs/solid_result.png"
                        results.append(run_solid_background(self.current_image, output_path, bg_color=self.selected_color))
                    elif selected_mode == "Background Replacement":
                        if not self.selected_bg_image:
                            print("❌ No background image selected.")
                            return []
                        output_path = "outputs/replaced_result.png"
                        results.append(run_background_replace(self.current_image, self.selected_bg_image, output_path))
                    elif selected_mode == "Smart Resize + Aspect Lock":
                        from PIL import Image
                        img = Image.open(self.current_image)
                        width = self.resize_width_input.value()
                        height = self.resize_height_input.value()
                        if self.aspect_lock_checkbox.isChecked():
                            original_ratio = img.width / img.height
                            target_ratio = width / height
                            if original_ratio > target_ratio:
                                height = int(width / original_ratio)
                            else:
                                width = int(height * original_ratio)
                        img = img.resize((width, height), Image.LANCZOS)
                        output_path = "outputs/resized_result.png"
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        img.save(output_path)
                        results.append(output_path)
                    else:
                        print(f"❌ Unsupported mode: {selected_mode}")
                except Exception as e:
                    print(f"❌ Error during processing: {e}")
                    return []
            return results

        worker = Worker(process)

        def handle_results(results):
            print("🔍 Entered function: handle_results")
            if results:
                self.processed_images = results
                self.current_image_index = 0
                for result_path in results:
                    if result_path and os.path.exists(result_path):
                        self.current_image = result_path
                        self.show_preview(result_path)
                        break
                    else:
                        print("⚠️ Output file not found or invalid:", result_path)

        worker.signals.result.connect(handle_results)
        worker.signals.finished.connect(self.loading_movie.stop)
        worker.signals.finished.connect(lambda: self.loading_label.setVisible(False))
        self.threadpool.start(worker)

    def toggle_ui_controls(self, mode):
        print("🔍 Entered function: toggle_ui_controls")
        show_webcam_controls = (mode == "Real-time Webcam")
        show_batch_controls = (mode == "Batch Processing")
        bg_sub_mode = self.bg_mode_selector.currentText()
        show_resize_controls = (mode == "Smart Resize + Aspect Lock")


        # General UI visibility logic
        self.live_toggle.setVisible(show_webcam_controls)
        self.mirror_checkbox.setVisible(show_webcam_controls)
        self.bg_mode_selector.setVisible(show_webcam_controls or show_batch_controls)

        # Solid color mode (single or batch)
        self.color_picker_btn.setVisible(
            (mode == "Solid Color Background") or
            (mode == "Batch Processing" and bg_sub_mode == "Solid Color") or
            (mode == "Real-time Webcam" and bg_sub_mode == "Solid Color")
        )


        # Background image button visibility
        self.bg_image_btn.setVisible(
            (mode == "Background Replacement") or
            (mode == "Batch Processing" and bg_sub_mode == "Image") or
            (show_webcam_controls and bg_sub_mode == "Image")
        )

        # Blur intensity slider
        self.blur_slider.setVisible(
            (mode == "Batch Processing" and bg_sub_mode == "Blur") or
            (show_webcam_controls and bg_sub_mode == "Blur")
        )

        # Show navigation buttons only for batch
        self.prev_btn.setVisible(mode == "Batch Processing")
        self.next_btn.setVisible(mode == "Batch Processing")

        # Show recording and webcam-related controls
        self.brightness_slider.setVisible(show_webcam_controls)
        self.start_record_btn.setVisible(show_webcam_controls)
        self.pause_record_btn.setVisible(show_webcam_controls)
        self.stop_record_btn.setVisible(show_webcam_controls)
        self.snapshot_btn.setVisible(show_webcam_controls)
        # Show resize inputs if in Smart Resize + Aspect Lock mode
        self.resize_width_input.setVisible(show_resize_controls)
        self.resize_height_input.setVisible(show_resize_controls)
        self.aspect_lock_checkbox.setVisible(show_resize_controls)




    def update_frame(self):
        print("🔍 Entered function: update_frame")
        ret, frame = self.cap.read()
        if not ret:
            return
        
        if self.paused:
            return  # Skip processing if paused

        # Unmirror by default: flip horizontally only if mirroring is enabled
        if self.mirror_checkbox.isChecked():
            frame = cv2.flip(frame, 1)

        # Resize for consistent processing
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness_value)  # <-- Add here
        self.current_frame = frame.copy()  # Store for snapshot


        # Apply background removal
        mask = self.get_segmentation_mask(frame)

        # Apply selected background
        bg_mode = self.bg_mode_selector.currentText()
        if bg_mode == "Solid Color":
            bgr_color = (self.selected_color[2], self.selected_color[1], self.selected_color[0])
            bg = np.full(frame.shape, bgr_color, dtype=np.uint8)

        elif bg_mode == "Image" and hasattr(self, 'bg_image_path') and self.bg_image_path:
            bg = cv2.imread(self.bg_image_path)
            if bg is None:
                print("❌ Failed to load background image.")
                bg = np.zeros_like(frame)
            else:
                bg = cv2.resize(bg, (640, 480))
        elif bg_mode == "Blur":
            bg = cv2.GaussianBlur(frame, (0, 0), self.blur_intensity)
        else:
            bg = np.zeros_like(frame)

        # Blend
        mask = (mask / 255.0).astype(np.float32)
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)
        combined = frame * mask + bg * (1 - mask)
        combined = combined.astype(np.uint8)

        # Save to video if recording
        if self.recording and not self.paused and self.out:
            self.out.write(combined)


        # Display
        rgb_image = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.scene.clear()
        self.scene.addPixmap(pixmap)

    def get_segmentation_mask(self, frame):
        print("🔍 Entered function: get_segmentation_mask")
        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess image
        image_tensor, _ = preprocess_image(image)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        # Run through model
        with torch.no_grad():
            d1 = self.net(image_tensor)[0]

        # Postprocess mask
        mask = postprocess_mask(d1, image.size)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        return mask

    def update_recording_time(self):
        print("🔍 Entered function: update_recording_time")
        self.recording_time = self.recording_time.addSecs(1)
        self.recording_label.setText(f"● REC {self.recording_time.toString('mm:ss')}")



    def button_style(self):
        print("🔍 Entered function: button_style")
        return """
        QPushButton {
            background-color: #1E88E5;
            color: white;
            padding: 10px 16px;
            border-radius: 6px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #42A5F5;
        }
        """

    def dragEnterEvent(self, event):
        print("🔍 Entered function: dragEnterEvent")
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        print("🔍 Entered function: dropEvent")
        urls = event.mimeData().urls()
        if urls:
            image_path = urls[0].toLocalFile()
            self.show_preview(image_path)
            self.current_image = image_path

    def pick_color(self):
        print("🔍 Entered function: pick_color")
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = tuple(map(int, (color.red(), color.green(), color.blue())))
            print(f"Applying solid color: {self.selected_color}")


    def pick_background_image(self):
        print("🔍 Entered function: pick_background_image")
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Background Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.selected_bg_image = file_path
            self.bg_image_path = file_path  # ✅ Fix: now webcam can read it too
            print(f"📸 Background image selected: {file_path}")

    def toggle_live_mode(self, state):
        print("🔍 Entered function: toggle_live_mode")
        if state == Qt.Checked:
            self.start_live_feed()
        else:
            self.stop_live_feed()

    def change_blur_intensity(self, value):
        print("🔍 Entered function: change_blur_intensity")
        self.blur_intensity = value

    def select_bg_image(self):
        print("🔍 Entered function: select_bg_image")
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Background Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.bg_image_path = file_path  # for webcam usage
            self.selected_bg_image = file_path  # for static usage
            print(f"📸 Background image selected: {file_path}")


    def start_live_feed(self):
        print("🔍 Entered function: start_live_feed")
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Approx. 30 frames per second

    def stop_live_feed(self):
        print("🔍 Entered function: stop_live_feed")
        self.timer.stop()
        self.cap.release()
        self.scene.clear()

    def start_recording(self):
        print("🔍 Entered function: start_recording")
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter('outputs/recorded_video.avi', fourcc, 20.0, (frame_width, frame_height))

        if not self.out.isOpened():
            print("❌ Failed to initialize video writer.")
            self.out = None
            return

        self.recording = True
        self.paused = False
        self.start_record_btn.setEnabled(False)
        self.pause_record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(True)
        self.recording_timer.start(1000)
        self.recording_label.setVisible(True)
        print("🎥 Recording started...")



    def pause_recording(self):
        print("🔍 Entered function: pause_recording")
        if self.recording:
            self.paused = not self.paused
            if self.paused:
                self.pause_record_btn.setText("Resume Recording")
                self.recording_timer.stop()
            else:
                self.pause_record_btn.setText("Pause Recording")
                self.recording_timer.start(1000)


    def stop_recording(self):
        print("🔍 Entered function: stop_recording")
        if self.recording:
            self.recording = False
            self.paused = False
            if self.out:
                self.out.release()
                self.out = None
            self.start_record_btn.setEnabled(True)
            self.pause_record_btn.setEnabled(False)
            self.stop_record_btn.setEnabled(False)
            self.pause_record_btn.setText("Pause Recording")
            self.recording_timer.stop()
            self.recording_label.setVisible(False)
            print("🎥 Recording stopped.")
            self.play_video_with_vlc("outputs/recorded_video.avi")


    def play_video_with_vlc(self, video_path):
        print("🔍 Entered function: play_video_with_vlc")
        if not os.path.exists(video_path):
            print("❌ Video file not found:", video_path)
            return

        try:
            instance = vlc.Instance()
            self.vlc_player = instance.media_player_new()
            media = instance.media_new(video_path)
            self.vlc_player.set_media(media)
            self.vlc_player.play()
        except Exception as e:
            print("❌ Failed to play video:", e)




    def take_snapshot(self):
        print("🔍 Entered function: take_snapshot")
        if hasattr(self, 'current_frame'):
            snapshot_path = 'outputs/snapshot.png'
            cv2.imwrite(snapshot_path, self.current_frame)
            print(f"Snapshot saved to {snapshot_path}")

    def run_batch_processing(self):
        print("🔍 Entered function: run_batch_processing")
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Images")
        if not folder_path:
            return

        self.image_files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not self.image_files:
            print("❌ No images found in folder.")
            return

        self.current_image_index = 0
        self.current_image = self.image_files[0]
        self.show_preview(self.current_image)

        if self.mode_selector.currentText() == "Batch Processing":
            self.prev_btn.setVisible(True)
            self.next_btn.setVisible(True)

        # ✅ Force re-evaluate visibility of buttons
        self.toggle_ui_controls("Batch Processing")




    def process_next_image(self):
        print("🔍 Entered function: process_next_image")
        if self.current_batch_index >= len(self.image_files):
            self.batch_timer.stop()
            print("✅ Batch processing complete.")
            return

        image_path = self.image_files[self.current_batch_index]
        output_path = os.path.join("outputs", f"batch_{os.path.basename(image_path)}")

        selected_mode = self.mode_selector.currentText()
        print(f"📸 Processing: {image_path} -> Mode: {selected_mode}")

        if selected_mode == "Transparent PNG Output":
            print("🛠️ Calling image processing function: result = run_u2net(image_path, output_path)")
            result = run_u2net(image_path, output_path)
        elif selected_mode == "Solid Color Background":
            print("🛠️ Calling image processing function: result = run_solid_background(image_path, output_path, bg_color=self.selected_color)")
            result = run_solid_background(image_path, output_path, bg_color=self.selected_color)
        elif selected_mode == "Background Replacement" and self.selected_bg_image:
            print("🛠️ Calling image processing function: result = run_background_replace(image_path, self.selected_bg_image, output_path)")
            result = run_background_replace(image_path, self.selected_bg_image, output_path)
        elif selected_mode == "Batch Processing":
            bg_mode = self.bg_mode_selector.currentText()
            if bg_mode == "Solid Color":
                print("🛠️ Calling image processing function: result = run_solid_background(image_path, output_path, bg_color=self.selected_color)")
                result = run_solid_background(image_path, output_path, bg_color=self.selected_color)
            elif bg_mode == "Image" and self.selected_bg_image:
                print("🛠️ Calling image processing function: result = run_background_replace(image_path, self.selected_bg_image, output_path)")
                result = run_background_replace(image_path, self.selected_bg_image, output_path)
            elif bg_mode == "Transparent PNG Output":
                # Force PNG output path to keep transparency
                output_path = os.path.join("outputs", f"{os.path.splitext(os.path.basename(image_path))[0]}_transparent.png")
                print("🛠️ Calling image processing function: result = run_u2net(image_path, output_path)")
                result = run_u2net(image_path, output_path)
            else:
                print("❌ Unsupported background mode in Batch Processing.")
                result = None

        else:
            print(f"❌ Unsupported mode: {selected_mode}")
            result = None


    def run_batch_processing(self):
        print("🔍 Entered function: run_batch_processing")
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Images")
        if not folder_path:
            return

        self.image_files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not self.image_files:
            print("❌ No images found in folder.")
            return

        self.current_image_index = 0
        self.current_image = self.image_files[0]
        self.show_preview(self.current_image)

        if self.mode_selector.currentText() == "Batch Processing":
            self.prev_btn.setVisible(True)
            self.next_btn.setVisible(True)

        # ✅ Add this to re-evaluate UI buttons
        self.toggle_ui_controls("Batch Processing")


    def process_next_image(self):
        print("🔍 Entered function: process_next_image")
        if self.current_batch_index >= len(self.image_files):
            self.batch_timer.stop()
            print("✅ Batch processing complete.")
            return

        image_path = self.image_files[self.current_batch_index]
        output_path = os.path.join("outputs", f"batch_{os.path.basename(image_path)}")

        selected_mode = self.mode_selector.currentText()
        print(f"📸 Processing: {image_path} -> Mode: {selected_mode}")

        if selected_mode == "Transparent PNG Output":
            print("🛠️ Calling image processing function: result = run_u2net(image_path, output_path)")
            result = run_u2net(image_path, output_path)
        elif selected_mode == "Solid Color Background":
            print("🛠️ Calling image processing function: result = run_solid_background(image_path, output_path, bg_color=self.selected_color)")
            result = run_solid_background(image_path, output_path, bg_color=self.selected_color)
        else:
            print("❌ Unsupported mode for batch processing.")
            self.current_batch_index += 1
            return

        self.show_preview(result)
        self.current_batch_index += 1

    def show_next_image(self):
        print("🔍 Entered function: show_next_image")
        images = getattr(self, "processed_images", self.image_files)
        if images and self.current_image_index < len(images) - 1:
            self.current_image_index += 1
            self.current_image = images[self.current_image_index]
            self.show_preview(self.current_image)


    def show_previous_image(self):
        print("🔍 Entered function: show_previous_image")
        images = getattr(self, "processed_images", self.image_files)
        if images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.current_image = images[self.current_image_index]
            self.show_preview(self.current_image)



    def futuristic_button_style(self):
        print("🔍 Entered function: futuristic_button_style")
        return """
        QPushButton {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #0f2027, stop:1 #2c5364);
            color: #ffffff;
            border: 2px solid #00f0ff;
            border-radius: 12px;
            padding: 14px 24px;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #1c1c1c;
            border: 2px solid #00ffea;
        }
        QPushButton:pressed {
            background-color: #0f0f0f;
            border: 2px solid #00d4ff;
        }
        """


    def combo_box_style(self):
        print("🔍 Entered function: combo_box_style")
        return """
        QComboBox {
            background-color: #2c3e50;
            color: #ecf0f1;
            border: 1px solid #2980b9;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 16px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid #2980b9;
        }
        QComboBox::down-arrow {
            image: none;
            width: 14px;
            height: 14px;
        }
        QComboBox QAbstractItemView {
            background-color: #34495e;
            color: #ecf0f1;
            selection-background-color: #2980b9;
            font-size: 16px;
            padding: 5px;
        }
        """


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI';
            font-size: 15px;
            color: #ffffff;
            background-color: #121212;
        }
        QComboBox {
            background-color: #2c3e50;
            color: #ecf0f1;
            border: 1px solid #2980b9;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 16px;
        }
        QComboBox QAbstractItemView {
            background-color: #34495e;
            color: #ecf0f1;
            selection-background-color: #2980b9;
            font-size: 16px;
            padding: 5px;
        }
        QPushButton {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                              stop:0 #0f2027, stop:1 #2c5364);
            color: #ffffff;
            border: 2px solid #00f0ff;
            border-radius: 12px;
            padding: 14px 24px;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #1c1c1c;
            border: 2px solid #00ffea;
        }
        QPushButton:pressed {
            background-color: #0f0f0f;
            border: 2px solid #00d4ff;
        }
    """)

win = SmartBackgroundApp()
win.show()
sys.exit(app.exec_())