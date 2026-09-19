import threading
import queue
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from config import IMAGE_DIR, VIDEO_DIR, OUTPUT_DIR
from src.detector import VehicleDetector
from src.traffic import (
    traffic_decision,
    calculate_occupancy
)
from src.roi import (
    select_road_area,
    crop_to_roi
)
from src.history import TrafficHistory
from src.video_srvice import process_video


class SmartTrafficApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Density Based Smart Traffic Control System"
        )

        self.root.geometry(
            "1200x750"
        )

        self.root.configure(
            bg="#20232A"
        )

        # --------------------------------
        # Variables
        # --------------------------------

        self.image_path = None
        self.video_path = None

        self.roi = None

        self.detector = None

        self.history = TrafficHistory()

        self.stop_event = threading.Event()

        self.video_running = False

        self.frame_queue = queue.Queue(
            maxsize=2
        )

        self.current_frame = None

        # --------------------------------
        # Create GUI
        # --------------------------------

        self.create_gui()

        # Check video queue regularly
        self.root.after(
            50,
            self.check_queue
        )

        # Close properly
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

    # ==================================================
    # CREATE GUI
    # ==================================================

    def create_gui(self):

        # --------------------------------
        # TITLE
        # --------------------------------

        title = tk.Label(
            self.root,
            text="DENSITY BASED SMART TRAFFIC CONTROL SYSTEM",
            bg="#8B0000",
            fg="white",
            font=("Arial", 20, "bold"),
            pady=15
        )

        title.pack(
            fill="x"
        )

        subtitle = tk.Label(
            self.root,
            text="YOLO Vehicle Detection + Traffic Analysis",
            bg="#20232A",
            fg="white",
            font=("Arial", 12, "bold")
        )

        subtitle.pack(
            pady=8
        )

        # --------------------------------
        # MAIN FRAME
        # --------------------------------

        main_frame = tk.Frame(
            self.root,
            bg="#30343B",
            bd=2,
            relief="groove"
        )

        main_frame.place(
            x=30,
            y=100,
            width=1140,
            height=580
        )

        # ============================================
        # LEFT SIDE
        # ============================================

        left_frame = tk.Frame(
            main_frame,
            bg="#30343B"
        )

        left_frame.place(
            x=20,
            y=20,
            width=350,
            height=530
        )

        # Upload image
        tk.Button(
            left_frame,
            text="1. Upload Traffic Image",
            command=self.upload_image,
            bg="#1976D2",
            fg="white",
            font=("Arial", 11, "bold"),
            width=30,
            height=2
        ).pack(
            pady=5
        )

        # Detect vehicle
        tk.Button(
            left_frame,
            text="2. Detect Vehicles",
            command=self.detect_vehicles,
            bg="#388E3C",
            fg="white",
            font=("Arial", 11, "bold"),
            width=30,
            height=2
        ).pack(
            pady=5
        )

        # Select ROI
        tk.Button(
            left_frame,
            text="3. Select Road / Lane",
            command=self.select_roi,
            bg="#7B1FA2",
            fg="white",
            font=("Arial", 11, "bold"),
            width=30,
            height=2
        ).pack(
            pady=5
        )

        # Open video
        tk.Button(
            left_frame,
            text="4. Open Traffic Video",
            command=self.open_video,
            bg="#F57C00",
            fg="white",
            font=("Arial", 11, "bold"),
            width=30,
            height=2
        ).pack(
            pady=5
        )

        # Start video
        tk.Button(
            left_frame,
            text="5. Start Traffic Video",
            command=self.start_video,
            bg="#00897B",
            fg="white",
            font=("Arial", 11, "bold"),
            width=30,
            height=2
        ).pack(
            pady=5
        )

        # Start camera
        tk.Button(
            left_frame,
            text="6. Start Live Camera",
            command=self.start_camera,
            bg="#009688",
            fg="white",
            font=("Arial", 11, "bold"),
            width=30,
            height=2
        ).pack(
            pady=5
        )

        # Stop
        tk.Button(
            left_frame,
            text="7. Stop Video / Camera",
            command=self.stop_video,
            bg="#D32F2F",
            fg="white",
            font=("Arial", 11, "bold"),
            width=30,
            height=2
        ).pack(
            pady=5
        )

        # Dashboard
        tk.Button(
            left_frame,
            text="8. Dashboard",
            command=self.show_dashboard,
            bg="#455A64",
            fg="white",
            font=("Arial", 11, "bold"),
            width=30,
            height=2
        ).pack(
            pady=5
        )

        # Report
        tk.Button(
            left_frame,
            text="9. Generate Report",
            command=self.generate_report,
            bg="#5D4037",
            fg="white",
            font=("Arial", 11, "bold"),
            width=30,
            height=2
        ).pack(
            pady=5
        )

        # ============================================
        # RIGHT SIDE
        # ============================================

        right_frame = tk.Frame(
            main_frame,
            bg="#272A30"
        )

        right_frame.place(
            x=390,
            y=20,
            width=720,
            height=530
        )

        # --------------------------------
        # Path label
        # --------------------------------

        self.path_label = tk.Label(
            right_frame,
            text="No file selected.",
            bg="#272A30",
            fg="white",
            font=("Arial", 9),
            anchor="w"
        )

        self.path_label.pack(
            fill="x",
            padx=10,
            pady=8
        )

        # --------------------------------
        # Image preview
        # --------------------------------

        preview_frame = tk.Frame(
            right_frame,
            bg="black",
            width=680,
            height=300
        )

        preview_frame.pack(
            padx=10,
            pady=5
        )

        preview_frame.pack_propagate(
            False
        )

        self.preview_label = tk.Label(
            preview_frame,
            text="Traffic Image / Video Preview",
            bg="black",
            fg="white",
            font=("Arial", 13, "bold")
        )

        self.preview_label.pack(
            expand=True
        )

        # --------------------------------
        # RESULTS
        # --------------------------------

        result_frame = tk.Frame(
            right_frame,
            bg="#272A30"
        )

        result_frame.pack(
            pady=10
        )

        self.total_label = self.create_result(
            result_frame,
            "Total Vehicles",
            0,
            0
        )

        self.car_label = self.create_result(
            result_frame,
            "Cars",
            0,
            1
        )

        self.bike_label = self.create_result(
            result_frame,
            "Motorcycles",
            0,
            2
        )

        self.bus_label = self.create_result(
            result_frame,
            "Buses",
            0,
            3
        )

        self.truck_label = self.create_result(
            result_frame,
            "Trucks",
            1,
            0
        )

        self.density_label = self.create_result(
            result_frame,
            "Density",
            1,
            1
        )

        self.level_label = self.create_result(
            result_frame,
            "Traffic Level",
            1,
            2
        )

        self.green_label = self.create_result(
            result_frame,
            "Green Time",
            1,
            3
        )

        # --------------------------------
        # STATUS
        # --------------------------------

        self.status_label = tk.Label(
            self.root,
            text="Ready.",
            bg="#20232A",
            fg="#FFD54F",
            font=("Arial", 11, "bold")
        )

        self.status_label.pack(
            side="bottom",
            pady=5
        )

    # ==================================================
    # RESULT BOX
    # ==================================================

    def create_result(
        self,
        parent,
        title,
        row,
        column
    ):

        frame = tk.Frame(
            parent,
            bg="#30343B",
            bd=1,
            relief="ridge"
        )

        frame.grid(
            row=row,
            column=column,
            padx=5,
            pady=5
        )

        tk.Label(
            frame,
            text=title,
            bg="#30343B",
            fg="#FFD54F",
            font=("Arial", 9, "bold")
        ).pack(
            padx=10,
            pady=(5, 0)
        )

        value_label = tk.Label(
            frame,
            text="--",
            bg="#30343B",
            fg="white",
            font=("Arial", 12, "bold")
        )

        value_label.pack(
            padx=15,
            pady=5
        )

        return value_label

    # ==================================================
    # LOAD YOLO MODEL
    # ==================================================

    def get_detector(self):

        if self.detector is None:

            self.status_label.config(
                text="Loading YOLO model..."
            )

            self.root.update_idletasks()

            self.detector = VehicleDetector()

            self.status_label.config(
                text="YOLO model loaded."
            )

        return self.detector

    # ==================================================
    # UPLOAD IMAGE
    # ==================================================

    def upload_image(self):

        file_path = filedialog.askopenfilename(
            initialdir=IMAGE_DIR,
            title="Select Traffic Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp"
                )
            ]
        )

        if not file_path:
            return

        image = cv2.imread(
            file_path
        )

        if image is None:

            messagebox.showerror(
                "Error",
                "Unable to read image."
            )

            return

        self.image_path = file_path

        self.video_path = None

        self.roi = None

        self.path_label.config(
            text=file_path
        )

        self.show_image(
            image
        )

        self.status_label.config(
            text="Traffic image uploaded."
        )

    # ==================================================
    # DETECT VEHICLES IN IMAGE
    # ==================================================

    def detect_vehicles(self):

        if self.image_path is None:

            messagebox.showwarning(
                "Image Required",
                "Please upload an image first."
            )

            return

        try:

            image = cv2.imread(
                self.image_path
            )

            (
                road_image,
                offset_x,
                offset_y,
                road_width,
                road_height
            ) = crop_to_roi(
                image,
                self.roi
            )

            detector = self.get_detector()

            detections, counts = (
                detector.detect_image(
                    road_image
                )
            )

            result_image = (
                detector.draw_detections(
                    image,
                    detections,
                    offset_x,
                    offset_y
                )
            )

            # Draw ROI
            if self.roi is not None:

                x, y, width, height = (
                    self.roi
                )

                cv2.rectangle(
                    result_image,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 255),
                    3
                )

            # Total vehicles
            total = sum(
                counts.values()
            )

            # Density
            density = calculate_occupancy(
                detections,
                road_width * road_height
            )

            # Traffic level
            level, green_time = (
                traffic_decision(
                    total
                )
            )

            # Show results
            self.update_results(
                counts,
                density,
                level,
                green_time
            )

            # Show image
            self.show_image(
                result_image
            )

            # Save image
            output_file = (
                OUTPUT_DIR /
                "vehicle_detection.png"
            )

            cv2.imwrite(
                str(output_file),
                result_image
            )

            # Save history
            self.history.save(
                Path(
                    self.image_path
                ).name,
                "image",
                counts,
                total,
                density,
                level,
                green_time
            )

            self.status_label.config(
                text="Vehicle detection completed."
            )

            messagebox.showinfo(
                "Detection Complete",
                f"Cars: {counts['car']}\n"
                f"Motorcycles: {counts['motorcycle']}\n"
                f"Buses: {counts['bus']}\n"
                f"Trucks: {counts['truck']}\n\n"
                f"Total: {total}\n"
                f"Traffic: {level}\n"
                f"Green Time: {green_time} seconds"
            )

        except Exception as error:

            messagebox.showerror(
                "Detection Error",
                str(error)
            )

    # ==================================================
    # SELECT ROAD / LANE
    # ==================================================

    def select_roi(self):

        frame = None

        # From image
        if self.image_path:

            frame = cv2.imread(
                self.image_path
            )

        # From video
        elif self.video_path:

            capture = cv2.VideoCapture(
                self.video_path
            )

            success, frame = (
                capture.read()
            )

            capture.release()

            if not success:
                frame = None

        if frame is None:

            messagebox.showwarning(
                "Source Required",
                "First upload an image or open a video."
            )

            return

        self.roi = select_road_area(
            frame
        )

        if self.roi is None:

            self.status_label.config(
                text="ROI selection cancelled."
            )

            return

        x, y, width, height = (
            self.roi
        )

        preview = frame.copy()

        cv2.rectangle(
            preview,
            (x, y),
            (x + width, y + height),
            (0, 255, 255),
            3
        )

        self.show_image(
            preview
        )

        self.status_label.config(
            text="Road/Lane area selected."
        )

    # ==================================================
    # OPEN VIDEO
    # ==================================================

    def open_video(self):

        file_path = filedialog.askopenfilename(
            initialdir=VIDEO_DIR,
            title="Select Traffic Video",
            filetypes=[
                (
                    "Video Files",
                    "*.mp4 *.avi *.mov *.mkv"
                )
            ]
        )

        if not file_path:
            return

        capture = cv2.VideoCapture(
            file_path
        )

        success, frame = (
            capture.read()
        )

        capture.release()

        if not success:

            messagebox.showerror(
                "Error",
                "Unable to open video."
            )

            return

        self.video_path = file_path

        self.image_path = None

        self.roi = None

        self.path_label.config(
            text=file_path
        )

        self.show_image(
            frame
        )

        self.status_label.config(
            text="Video selected. Click Start Traffic Video."
        )

    # ==================================================
    # START VIDEO
    # ==================================================

    def start_video(self):

        if self.video_path is None:

            messagebox.showwarning(
                "Video Required",
                "Please open a traffic video first."
            )

            return

        self.start_processing(
            self.video_path
        )

    # ==================================================
    # START CAMERA
    # ==================================================

    def start_camera(self):

        self.start_processing(
            0
        )

    # ==================================================
    # PROCESS VIDEO
    # ==================================================

    def start_processing(
        self,
        source
    ):

        if self.video_running:

            messagebox.showinfo(
                "Already Running",
                "Video/camera is already running."
            )

            return

        self.stop_event.clear()

        self.video_running = True

        def video_thread():

            try:

                process_video(
                    source,
                    self.get_detector(),
                    self.roi,
                    self.stop_event,
                    self.receive_video_frame,
                    self.save_video_data
                )

            except Exception as error:

                self.add_to_queue(
                    (
                        "error",
                        str(error)
                    )
                )

            finally:

                self.video_running = False

                self.add_to_queue(
                    (
                        "status",
                        "Video processing stopped."
                    )
                )

        threading.Thread(
            target=video_thread,
            daemon=True
        ).start()

        self.status_label.config(
            text="Video processing started."
        )

    # ==================================================
    # STOP VIDEO
    # ==================================================

    def stop_video(self):

        self.stop_event.set()

        self.status_label.config(
            text="Stopping video..."
        )

    # ==================================================
    # RECEIVE VIDEO FRAME
    # ==================================================

    def receive_video_frame(
        self,
        frame,
        counts,
        total,
        density,
        level,
        green_time
    ):

        self.add_to_queue(
            (
                "frame",
                frame,
                counts,
                total,
                density,
                level,
                green_time
            )
        )

    # ==================================================
    # SAVE VIDEO DATA
    # ==================================================

    def save_video_data(
        self,
        counts,
        total,
        density,
        level,
        green_time,
        elapsed_time
    ):

        if self.video_path:

            source = Path(
                self.video_path
            ).name

            mode = "video"

        else:

            source = "Webcam"

            mode = "webcam"

        self.history.save(
            source,
            mode,
            counts,
            total,
            density,
            level,
            green_time
        )

    # ==================================================
    # QUEUE
    # ==================================================

    def add_to_queue(
        self,
        data
    ):

        try:

            if self.frame_queue.full():

                self.frame_queue.get_nowait()

            self.frame_queue.put_nowait(
                data
            )

        except Exception:
            pass

    # ==================================================
    # CHECK QUEUE
    # ==================================================

    def check_queue(self):

        try:

            while True:

                data = (
                    self.frame_queue.get_nowait()
                )

                # Frame
                if data[0] == "frame":

                    (
                        _,
                        frame,
                        counts,
                        total,
                        density,
                        level,
                        green_time
                    ) = data

                    self.show_image(
                        frame
                    )

                    self.update_results(
                        counts,
                        density,
                        level,
                        green_time
                    )

                # Status
                elif data[0] == "status":

                    self.status_label.config(
                        text=data[1]
                    )

                # Error
                elif data[0] == "error":

                    messagebox.showerror(
                        "Video Error",
                        data[1]
                    )

        except queue.Empty:
            pass

        self.root.after(
            50,
            self.check_queue
        )

    # ==================================================
    # UPDATE RESULTS
    # ==================================================

    def update_results(
        self,
        counts,
        density,
        level,
        green_time
    ):

        total = sum(
            counts.values()
        )

        self.total_label.config(
            text=str(total)
        )

        self.car_label.config(
            text=str(
                counts.get(
                    "car",
                    0
                )
            )
        )

        self.bike_label.config(
            text=str(
                counts.get(
                    "motorcycle",
                    0
                )
            )
        )

        self.bus_label.config(
            text=str(
                counts.get(
                    "bus",
                    0
                )
            )
        )

        self.truck_label.config(
            text=str(
                counts.get(
                    "truck",
                    0
                )
            )
        )

        self.density_label.config(
            text=f"{density:.2f}%"
        )

        self.level_label.config(
            text=level
        )

        self.green_label.config(
            text=f"{green_time} sec"
        )

    # ==================================================
    # SHOW IMAGE
    # ==================================================

    def show_image(
        self,
        frame
    ):

        self.current_frame = (
            frame.copy()
        )

        # Convert BGR → RGB
        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            image
        )

        # Resize
        image.thumbnail(
            (680, 300)
        )

        photo = ImageTk.PhotoImage(
            image
        )

        self.preview_label.config(
            image=photo,
            text=""
        )

        self.preview_label.image = (
            photo
        )

    # ==================================================
    # DASHBOARD
    # ==================================================

    def show_dashboard(self):

        dashboard = tk.Toplevel(
            self.root
        )

        dashboard.title(
            "Traffic Dashboard"
        )

        dashboard.geometry(
            "700x500"
        )

        dashboard.configure(
            bg="#20232A"
        )

        tk.Label(
            dashboard,
            text="TRAFFIC DASHBOARD",
            bg="#20232A",
            fg="white",
            font=("Arial", 20, "bold")
        ).pack(
            pady=20
        )

        rows = self.history.rows

        if not rows:

            tk.Label(
                dashboard,
                text="No traffic history available.",
                bg="#20232A",
                fg="white",
                font=("Arial", 13)
            ).pack(
                pady=30
            )

            return

        total_records = len(
            rows
        )

        average_vehicles = sum(
            int(row["total"])
            for row in rows
        ) / total_records

        average_density = sum(
            float(
                row["density_percent"]
            )
            for row in rows
        ) / total_records

        tk.Label(
            dashboard,
            text=f"Total Records: {total_records}",
            bg="#20232A",
            fg="white",
            font=("Arial", 13)
        ).pack(
            pady=8
        )

        tk.Label(
            dashboard,
            text=f"Average Vehicles: {average_vehicles:.2f}",
            bg="#20232A",
            fg="#FFD54F",
            font=("Arial", 13, "bold")
        ).pack(
            pady=8
        )

        tk.Label(
            dashboard,
            text=f"Average Density: {average_density:.2f}%",
            bg="#20232A",
            fg="#69F0AE",
            font=("Arial", 13, "bold")
        ).pack(
            pady=8
        )

    # ==================================================
    # GENERATE REPORT
    # ==================================================

    def generate_report(self):

        output_file = (
            OUTPUT_DIR /
            "traffic_report.csv"
        )

        self.history.export_report(
            output_file
        )

        messagebox.showinfo(
            "Report Generated",
            f"Report saved at:\n{output_file}"
        )

    # ==================================================
    # CLOSE
    # ==================================================

    def close_application(self):

        self.stop_event.set()

        self.root.destroy()