import os
import io
import urllib.request
from PIL import Image
from db import RebrickableOfflineDB

DEFAULT_TRAINING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training_data"))

COMMON_BULK_PARTS = [
    "3001",
    "3002",
    "3003",
    "3004",
    "3005",
    "3010",
    "3020",
    "3021",
    "3022",
    "3023",
    "3024",
    "3710",
    "3039",
    "3040"
]


def sanitize_folder_name(name: str) -> str:
    """
    Strips invalid filesystem characters and whitespace from color names.

    @parameters:
        @param name: str - Raw color name string.
    @returns:
        str - Cleaned alphanumeric string.
    """
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()


def augment_and_save(image_bytes: bytes, destination_dir: str, base_filename: str, rotations: int = 16) -> None:
    """
    Decodes an image, generates rotated variants against a white canvas, and saves to disk.

    @parameters:
        @param image_bytes: bytes - Raw image binary payload.
        @param destination_dir: str - Target directory path.
        @param base_filename: str - Prefix name for saved image files.
        @param rotations: int - Number of synthetic rotational steps across 360 degrees.
    @returns:
        None
    """
    os.makedirs(destination_dir, exist_ok=True)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    step = 360 // rotations
    for idx, angle in enumerate(range(0, 360, step)):
        rotated = pil_img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(255, 255, 255))
        target_path = os.path.join(destination_dir, f"{base_filename}_rot_{idx}.jpg")
        rotated.save(target_path, "JPEG", quality=95)


def harvest_training_data(output_root: str = DEFAULT_TRAINING_DIR, parts_list: list[str] = COMMON_BULK_PARTS) -> None:
    """
    Queries local database for image URLs and writes augmented datasets into part_color folders.

    @parameters:
        @param output_root: str - Destination root folder for the generated dataset.
        @param parts_list: list[str] - List of specific part numbers to fetch.
    @returns:
        None
    """
    reb_db = RebrickableOfflineDB()
    conn = reb_db.get_connection()

    placeholders = ",".join(["?"] * len(parts_list))
    query = f"""
        SELECT ip.part_num, c.name AS color_name, ip.img_url
        FROM inventory_parts ip
        JOIN colors c ON ip.color_id = c.id
        WHERE ip.img_url IS NOT NULL 
          AND ip.img_url != ''
          AND ip.part_num IN ({placeholders})
        GROUP BY ip.part_num, c.name;
    """

    rows = conn.execute(query, parts_list).fetchall()
    conn.close()

    print(f"Located {len(rows)} distinct part-color variants with official image URLs in local DB.")

    headers = {"User-Agent": "Mozilla/5.0"}

    for idx, (part_num, color_name, img_url) in enumerate(rows):
        safe_color = sanitize_folder_name(color_name)
        folder_name = f"{part_num}_{safe_color}"
        target_folder = os.path.join(output_root, folder_name)

        if os.path.exists(target_folder) and len(os.listdir(target_folder)) > 0:
            continue

        print(f"[{idx + 1}/{len(rows)}] Fetching: {folder_name}...")

        try:
            req = urllib.request.Request(img_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read()

            augment_and_save(
                image_bytes=content,
                destination_dir=target_folder,
                base_filename=f"{part_num}_{safe_color}",
                rotations=16
            )
        except Exception as err:
            print(f"Failed to download image for {folder_name}: {err}")

    print(f"\nDataset generation finished. Training data populated in '{output_root}'.")


if __name__ == "__main__":
    harvest_training_data()
