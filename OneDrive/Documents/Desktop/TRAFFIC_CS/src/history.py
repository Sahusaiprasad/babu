import csv
from datetime import datetime
from pathlib import Path

from config import HISTORY_FILE


FIELDS = [
    "timestamp",
    "source",
    "mode",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "total",
    "density_percent",
    "traffic_level",
    "green_time_seconds"
]


class TrafficHistory:

    def __init__(self):

        self.create_file_if_needed()

    @property
    def rows(self):
        with open(
            HISTORY_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:
            return list(csv.DictReader(file))

    def create_file_if_needed(self):

        if not HISTORY_FILE.exists() or HISTORY_FILE.stat().st_size == 0:

            with open(
                HISTORY_FILE,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=FIELDS
                )

                writer.writeheader()

    def save(
        self,
        source,
        mode,
        counts,
        total,
        density,
        traffic_level,
        green_time
    ):

        row = {
            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "source":
                source,

            "mode":
                mode,

            "car":
                counts.get("car", 0),

            "motorcycle":
                counts.get("motorcycle", 0),

            "bus":
                counts.get("bus", 0),

            "truck":
                counts.get("truck", 0),

            "total":
                total,

            "density_percent":
                round(density, 2),

            "traffic_level":
                traffic_level,

            "green_time_seconds":
                green_time
        }

        with open(
            HISTORY_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDS
            )

            writer.writerow(row)

    def export_report(self, output_file):
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(
            HISTORY_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as source:
            rows = list(csv.DictReader(source))

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as report:
            writer = csv.DictWriter(
                report,
                fieldnames=FIELDS
            )
            writer.writeheader()
            writer.writerows(rows)