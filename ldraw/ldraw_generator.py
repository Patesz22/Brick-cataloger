import os
import sys
import math
import shutil
import subprocess
import concurrent.futures
import numpy as np
from PIL import Image

LDRAW_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(LDRAW_SCRIPT_DIR, ".."))
LDRAW_LIBRARY_DIR = os.path.join(LDRAW_SCRIPT_DIR, "ldraw_library")
DEFAULT_TRAINING_DIR = os.path.join(ROOT_DIR, "training_data")
SRC_DIR = os.path.join(ROOT_DIR, "src")

for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from db import RebrickableOfflineDB
except ModuleNotFoundError:
    from src.db import RebrickableOfflineDB


def locate_ldview_binary() -> str:
    """
    Locates the hardware-accelerated LDView binary across standard paths.

    @parameters:
        None
    @returns:
        str - Path to the executable.
    """
    bin_path = shutil.which("ldview") or shutil.which("LDView")
    if bin_path:
        return bin_path

    common_paths = [
        r"C:\Program Files\LDView\LDView64.exe",
        r"C:\Program Files (x86)\LDView\LDView.exe",
        r"C:\Program Files\LDView\LDView.exe"
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "LDView executable not found. Install LDView from https://tcobbs.github.io/ldview/ "
        "and add it to your system PATH or standard Program Files folder."
    )


def compute_ldraw_matrix(yaw_deg: float, tilt_x_deg: float = 22.5) -> str:
    """
    Computes a combined 3D rotation matrix for LDraw Line Type 1 definitions.

    @parameters:
        @param yaw_deg: float - Turntable rotation around the vertical Y-axis in degrees.
        @param tilt_x_deg: float - Downward elevation pitch tilt around the X-axis in degrees.
    @returns:
        str - Space-delimited string of 9 matrix float values (a b c d e f g h i).
    """
    theta = math.radians(yaw_deg)
    phi = math.radians(tilt_x_deg)

    r_y = np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0, 1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)]
    ], dtype=np.float32)

    r_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(phi), -math.sin(phi)],
        [0.0, math.sin(phi), math.cos(phi)]
    ], dtype=np.float32)

    mat = np.dot(r_x, r_y)
    flat = mat.flatten()
    return " ".join(f"{val:.6f}" for val in flat)


def render_single_part_job(
        part_num: str,
        color_name: str,
        rgb_hex: str,
        output_root: str,
        ldview_bin: str,
        library_dir: str,
        rotations: int = 16,
        tilt_x: float = 22.5,
        render_res: int = 256,
        cnn_size: int = 224
) -> int:
    """
    Renders rotating 3D turntable samples scaled to 224x224 with studio lighting.

    @parameters:
        @param part_num: str - Target Brick design number.
        @param color_name: str - Name of the color class.
        @param rgb_hex: str - Hexadecimal RGB string.
        @param output_root: str - Destination root folder.
        @param ldview_bin: str - Executable path for LDView.
        @param library_dir: str - LDraw library root directory.
        @param rotations: int - Number of rotation steps across 360 degrees.
        @param tilt_x: float - X-axis tilt in degrees to reveal studs and thickness.
        @param render_res: int - Render snapshot resolution.
        @param cnn_size: int - Final output square dimension in pixels (224).
    @returns:
        int - Count of generated images.
    """
    clean_color = "".join(c for c in color_name if c.isalnum() or c in ("-", "_")).strip()
    target_folder = os.path.join(output_root, f"{part_num}_{clean_color}")

    os.makedirs(target_folder, exist_ok=True)
    hex_clean = (rgb_hex or "808080").lstrip("#").upper()

    # Contrast compensation: prevent pure black plastic from collapsing into silhouettes
    r = int(hex_clean[0:2], 16) if len(hex_clean) >= 6 else 128
    g = int(hex_clean[2:4], 16) if len(hex_clean) >= 6 else 128
    b = int(hex_clean[4:6], 16) if len(hex_clean) >= 6 else 128

    if max(r, g, b) < 50:
        hex_clean = "2E3137"

    direct_color = f"0x2{hex_clean}"
    step_angle = 360.0 / rotations
    generated_count = 0

    for r_idx in range(rotations):
        yaw = r_idx * step_angle
        matrix_str = compute_ldraw_matrix(yaw_deg=yaw, tilt_x_deg=tilt_x)

        temp_ldr = os.path.join(target_folder, f"_temp_{part_num}_{r_idx}.ldr")
        temp_png = os.path.join(target_folder, f"_gpu_render_{r_idx}.png")
        final_jpg = os.path.join(target_folder, f"{part_num}_{clean_color}_rot_{r_idx}.jpg")

        ldr_content = (
            f"0 FILE {part_num}.ldr\n"
            f"0 {part_num} in {color_name} rot {yaw:.1f}\n"
            f"1 {direct_color} 0 0 0 {matrix_str} {part_num}.dat\n"
            f"0 NOFILE\n"
        )

        with open(temp_ldr, "w", encoding="utf-8") as f:
            f.write(ldr_content)

        cmd = [
            ldview_bin,
            temp_ldr,
            f"-LDrawDir={library_dir}",
            f"-SaveSnapshot={temp_png}",
            f"-SaveWidth={render_res}",
            f"-SaveHeight={render_res}",
            "-SaveAlpha=1",
            "-AutoCrop=1",
            "-DefaultAngles=1",
            "-FOV=30",
            "-OpenGL=1",
            "-FSAA=0",
            "-Quality=1",
            "-Light1=1.0,1.0,1.0,0.6,-0.8,0.6",
            "-Light2=0.8,0.8,0.8,-0.6,-0.5,-0.4",
            "-Light3=0.6,0.6,0.6,0.0,0.8,-0.5",
            "-Ambient=0.25",
            "-Specular=1",
            "-EdgeLines=1",
            "-ConditionalLines=1",
            "-LineThickness=1",
            "-ProcessEvents=0"
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15
            )
            if os.path.isfile(temp_png):
                with Image.open(temp_png) as img:
                    rgba = img.convert("RGBA")
                    canvas = Image.new("RGB", rgba.size, (255, 255, 255))
                    canvas.paste(rgba, mask=rgba.split()[3])

                    canvas.thumbnail((cnn_size - 16, cnn_size - 16), Image.Resampling.BILINEAR)
                    final_canvas = Image.new("RGB", (cnn_size, cnn_size), (255, 255, 255))
                    offset_x = (cnn_size - canvas.width) // 2
                    offset_y = (cnn_size - canvas.height) // 2
                    final_canvas.paste(canvas, (offset_x, offset_y))
                    final_canvas.save(final_jpg, "JPEG", quality=92)

                os.remove(temp_png)
                generated_count += 1
        except Exception:
            pass

        if os.path.exists(temp_ldr):
            os.remove(temp_ldr)

    return generated_count


class GPULDrawBatchPipeline:
    """
    Manages throttled GPU batch rendering with embedded matrix rotation and tilt.
    Excludes already-generated parts and folders from the queue upfront.
    """

    def __init__(
            self,
            library_dir: str = LDRAW_LIBRARY_DIR,
            output_dir: str = DEFAULT_TRAINING_DIR,
            max_workers: int = 2
    ):
        """
        Initializes the rendering pipeline with safe worker limits.

        @parameters:
            @param library_dir: str - Local directory of the unpacked LDraw library.
            @param output_dir: str - Target root destination directory.
            @param max_workers: int - Worker limit preventing GPU driver TDR timeouts.
        @returns:
            None
        """
        self.library_dir = library_dir
        self.output_dir = output_dir
        self.ldview_bin = locate_ldview_binary()
        self.reb_db = RebrickableOfflineDB()
        self.max_workers = max_workers

    def run_gpu_batch(
            self,
            rotations_per_part: int = 16,
            tilt_x: float = 22.5,
            render_res: int = 256,
            cnn_size: int = 224,
            max_elements: int | None = None,
            exclude_existing_part_ids: bool = False
    ) -> None:
        """
        Executes batch rendering sorted by quantity, skipping elements that already have folders.

        @parameters:
            @param rotations_per_part: int - Number of rotation views per element.
            @param tilt_x: float - X-axis elevation tilt in degrees.
            @param render_res: int - Render snapshot resolution.
            @param cnn_size: int - Final output image square size (224).
            @param max_elements: int | None - Cap on total parts processed.
            @param exclude_existing_part_ids: bool - If True, skips a part_num if ANY folder
                                                     for that part_num already exists.
                                                     If False, only skips if the specific
                                                     part_num + color folder already exists.
        @returns:
            None
        """
        print(f"[INIT] Scanning destination folder '{self.output_dir}' for existing data...")

        existing_folders = set()
        existing_part_ids = set()

        if os.path.exists(self.output_dir):
            for d in os.listdir(self.output_dir):
                full_d = os.path.join(self.output_dir, d)
                if os.path.isdir(full_d):
                    # Check that the folder actually contains rendered images
                    images = [f for f in os.listdir(full_d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(images) >= rotations_per_part:
                        existing_folders.add(d)
                        if "_" in d:
                            existing_part_ids.add(d.split("_")[0])

        print(f"[CACHE] Found {len(existing_folders)} completed folders ({len(existing_part_ids)} unique part IDs).")

        query = """
                SELECT ip.part_num, c.name, c.rgb, SUM(ip.quantity) AS total_count
                FROM inventory_parts ip
                         JOIN colors c ON ip.color_id = c.id
                WHERE c.rgb IS NOT NULL AND c.rgb != ''
                GROUP BY ip.part_num, c.name
                ORDER BY total_count DESC, ip.part_num ASC; \
                """

        with self.reb_db.get_connection() as conn:
            rows = conn.execute(query).fetchall()

        parts_dir = os.path.join(self.library_dir, "parts")
        tasks = []
        skipped_count = 0

        for row in rows:
            part_num = str(row[0])
            color_name = str(row[1])
            rgb_hex = row[2]
            total_count = row[3]

            clean_color = "".join(c for c in color_name if c.isalnum() or c in ("-", "_")).strip()
            folder_name = f"{part_num}_{clean_color}"

            if exclude_existing_part_ids and part_num in existing_part_ids:
                skipped_count += 1
                continue

            if folder_name in existing_folders:
                skipped_count += 1
                continue

            part_path = os.path.join(parts_dir, f"{part_num}.dat")
            if os.path.isfile(part_path):
                tasks.append((part_num, color_name, rgb_hex, total_count))
                if exclude_existing_part_ids:
                    existing_part_ids.add(part_num)

            if max_elements is not None and len(tasks) >= max_elements:
                break

        print(f"[FILTER] Excluded {skipped_count} elements with existing folders.")
        print(f"[QUEUE] Queued {len(tasks)} new elements for generation.")

        if not tasks:
            print("[DONE] All elements in catalog already have rendered folders. Nothing to do.")
            return

        total_completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    render_single_part_job,
                    task[0],
                    task[1],
                    task[2],
                    self.output_dir,
                    self.ldview_bin,
                    self.library_dir,
                    rotations_per_part,
                    tilt_x,
                    render_res,
                    cnn_size
                )
                for task in tasks
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    count = future.result()
                    if count > 0:
                        total_completed += 1
                        if total_completed % 5 == 0 or total_completed == len(tasks):
                            print(f"[PROGRESS] Rendered {total_completed}/{len(tasks)} elements...")
                except Exception as err:
                    print(f"[WORKER ERROR] {err}")

        print(f"\n[DONE] Generation complete. New images saved in '{self.output_dir}'.")


if __name__ == "__main__":
    pipeline = GPULDrawBatchPipeline(max_workers=2)

    # Set exclude_existing_part_ids=True if you want only 1 folder per part number
    # Set exclude_existing_part_ids=False to allow multiple colors of the same part
    pipeline.run_gpu_batch(
        rotations_per_part=16,
        tilt_x=22.5,
        render_res=256,
        cnn_size=224,
        exclude_existing_part_ids=True
    )