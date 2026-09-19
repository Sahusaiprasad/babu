from ultralytics import YOLO

from config import MODEL_NAME, VEHICLE_CLASSES


class VehicleDetector:
    """
    Handles vehicle detection and tracking using YOLO.
    """

    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO(MODEL_NAME)
        print("YOLO model loaded.")

    def detect_image(self, image):
        """
        Detect vehicles in one image.
        """

        results = self.model.predict(
            source=image,
            conf=0.25,
            verbose=False
        )

        return self.process_results(results)

    def track_video_frame(self, frame):
        """
        Detect and track vehicles in a video frame.
        """

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False
        )

        return self.process_results(
            results,
            tracking=True
        )

    def process_results(self, results, tracking=False):
        """
        Convert YOLO results into simple Python dictionaries.
        """

        detections = []

        counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            coordinates = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            track_ids = None

            if tracking and boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)

            for i, (box, class_id, confidence) in enumerate(
                zip(
                    coordinates,
                    class_ids,
                    confidences
                )
            ):

                class_name = str(
                    self.model.names[int(class_id)]
                ).lower()

                # Ignore objects that aren't vehicles
                if class_name not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = [
                    int(value) for value in box
                ]

                track_id = None

                if (
                    track_ids is not None
                    and i < len(track_ids)
                ):
                    track_id = int(track_ids[i])

                counts[class_name] += 1

                detections.append(
                    {
                        "class": class_name,
                        "confidence": float(confidence),
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "track_id": track_id
                    }
                )

        return detections, counts

    def draw_detections(
        self,
        frame,
        detections,
        offset_x=0,
        offset_y=0
    ):
        """
        Draw bounding boxes and labels on an image.
        """

        output = frame.copy()

        import cv2

        for detection in detections:

            x1, y1, x2, y2 = detection["box"]

            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y

            label = (
                f'{detection["class"]} '
                f'{detection["confidence"]:.2f}'
            )

            if detection["track_id"] is not None:
                label += (
                    f' ID:{detection["track_id"]}'
                )

            # Bounding box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Text
            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        return output