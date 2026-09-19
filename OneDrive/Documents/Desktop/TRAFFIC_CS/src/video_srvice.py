import time

import cv2

from src.roi import crop_to_roi
from src.traffic import (
    calculate_occupancy,
    traffic_decision
)


def process_video(
    video_source,
    detector,
    roi,
    stop_event,
    frame_callback,
    data_callback
):
    """
    Process video or webcam frames.
    """

    capture = cv2.VideoCapture(
        video_source
    )

    if not capture.isOpened():

        print("Unable to open video source.")

        return

    start_time = time.time()

    last_save_time = 0

    while not stop_event.is_set():

        success, frame = capture.read()

        if not success:
            break

        # -----------------------------
        # Select road/lane
        # -----------------------------

        (
            road_frame,
            offset_x,
            offset_y,
            road_width,
            road_height
        ) = crop_to_roi(
            frame,
            roi
        )

        # -----------------------------
        # YOLO tracking
        # -----------------------------

        detections, counts = (
            detector.track_video_frame(
                road_frame
            )
        )

        # -----------------------------
        # Draw detections
        # -----------------------------

        output_frame = (
            detector.draw_detections(
                frame,
                detections,
                offset_x,
                offset_y
            )
        )

        # -----------------------------
        # Traffic calculation
        # -----------------------------

        total_vehicles = sum(
            counts.values()
        )

        density = calculate_occupancy(
            detections,
            road_width * road_height
        )

        traffic_level, green_time = (
            traffic_decision(
                total_vehicles
            )
        )

        # -----------------------------
        # Draw ROI
        # -----------------------------

        if roi is not None:

            x, y, width, height = roi

            cv2.rectangle(
                output_frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 255),
                3
            )

        # -----------------------------
        # Send frame to GUI
        # -----------------------------

        frame_callback(
            output_frame,
            counts,
            total_vehicles,
            density,
            traffic_level,
            green_time
        )

        # -----------------------------
        # Save history every second
        # -----------------------------

        current_time = time.time()

        if current_time - last_save_time >= 1:

            elapsed_time = (
                current_time - start_time
            )

            data_callback(
                counts,
                total_vehicles,
                density,
                traffic_level,
                green_time,
                elapsed_time
            )

            last_save_time = current_time

    capture.release()