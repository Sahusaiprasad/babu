from config import SIGNAL_RULES


def traffic_decision(vehicle_count):
    """
    Decide traffic level and green signal time
    based on number of detected vehicles.
    """

    for low, high, level, green_time in SIGNAL_RULES:

        if low <= vehicle_count <= high:
            return level, green_time

    return "VERY HIGH", 60


def calculate_occupancy(detections, roi_area):
    """
    Estimate how much of the selected road area
    is occupied by detected vehicles.
    """

    if roi_area <= 0:
        return 0.0

    occupied_area = 0

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        box_area = width * height

        occupied_area += box_area

    occupancy = (
        occupied_area / roi_area
    ) * 100

    return min(occupancy, 100.0)