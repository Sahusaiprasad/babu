import cv2


def select_road_area(frame):
    """
    Allow the user to select a road/lane area.
    """

    roi = cv2.selectROI(
        "Select Road / Lane",
        frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow("Select Road / Lane")

    x, y, width, height = [
        int(value) for value in roi
    ]

    if width == 0 or height == 0:
        return None

    return x, y, width, height


def crop_to_roi(frame, roi):
    """
    Crop the frame using selected road/lane ROI.
    """

    if roi is None:

        height, width = frame.shape[:2]

        return (
            frame,
            0,
            0,
            width,
            height
        )

    x, y, width, height = roi

    frame_height, frame_width = frame.shape[:2]

    x = max(
        0,
        min(x, frame_width - 1)
    )

    y = max(
        0,
        min(y, frame_height - 1)
    )

    width = max(
        1,
        min(width, frame_width - x)
    )

    height = max(
        1,
        min(height, frame_height - y)
    )

    cropped = frame[
        y:y + height,
        x:x + width
    ]

    return (
        cropped,
        x,
        y,
        width,
        height
    )
