import os
import sys
import math
import random
import shutil
import tempfile
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


def configure_process_background_priority() -> None:
    """
    Sets the host process and all child workers to below-normal OS scheduling priority.

    @parameters:
        None
    @returns:
        None
    """
    if sys.platform == "win32":
        try:
            import ctypes
            below_normal_priority_class = 0x00004000
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(process_handle, below_normal_priority_class)
        except Exception:
            pass


def get_scratch_directory() -> str:
    """
    Identifies a high-speed RAM-disk or configures a persistent OS memory cache directory.

    @parameters:
        None
    @returns:
        str - Path to the ephemeral storage buffer.
    """
    env_ramdisk = os.environ.get("BRICK_RAMDISK")
    if env_ramdisk and os.path.isdir(env_ramdisk):
        return env_ramdisk

    for candidate in [r"R:\render_scratch", r"B:\render_scratch", r"Z:\render_scratch"]:
        if os.path.isdir(candidate):
            return candidate

    system_temp = os.path.join(tempfile.gettempdir(), "brick_render_scratch")
    os.makedirs(system_temp, exist_ok=True)
    return system_temp


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


def compute_orthonormal_euler_matrix(pitch_x_deg: float, yaw_y_deg: float, roll_z_deg: float = 0.0) -> str:
    """
    Derives an orthonormal 3x3 transformation matrix preserving triangle winding parity.

    @parameters:
        @param pitch_x_deg: float - Elevation tilt angle around the X axis in degrees.
        @param yaw_y_deg: float - Turntable rotation angle around the Y axis in degrees.
        @param roll_z_deg: float - In-plane rotation angle around the Z axis in degrees.
    @returns:
        str - Space-delimited string of 9 matrix float values.
    """
    rad_x = math.radians(pitch_x_deg)
    rad_y = math.radians(yaw_y_deg)
    rad_z = math.radians(roll_z_deg)

    r_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(rad_x), -math.sin(rad_x)],
        [0.0, math.sin(rad_x), math.cos(rad_x)]
    ], dtype=np.float32)

    r_y = np.array([
        [math.cos(rad_y), 0.0, math.sin(rad_y)],
        [0.0, 1.0, 0.0],
        [-math.sin(rad_y), 0.0, math.cos(rad_y)]
    ], dtype=np.float32)

    r_z = np.array([
        [math.cos(rad_z), -math.sin(rad_z), 0.0],
        [math.sin(rad_z), math.cos(rad_z), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    mat = np.dot(r_z, np.dot(r_y, r_x))
    flat = mat.flatten()
    return " ".join(f"{val:.6f}" for val in flat)


def sample_physical_pose(index: int, total_rotations: int) -> tuple[float, float, float]:
    """
    Distributes viewpoints across four distinct physical stable resting states:
    lying flat on side (90 deg), upright (25 deg), inverted (160 deg), and tumble (55 deg).

    @parameters:
        @param index: int - Current orientation index.
        @param total_rotations: int - Total target viewpoints.
    @returns:
        tuple[float, float, float] - (pitch_x, yaw_y, roll_z) in degrees.
    """
    slot = index % 4
    step = (360.0 / max(1, total_rotations // 4)) * (index // 4)

    if slot == 0:
        return (90.0, step, 0.0)
    elif slot == 1:
        return (25.0, step, 0.0)
    elif slot == 2:
        return (160.0, step, 0.0)
    else:
        return (55.0, step, 30.0)


def composite_clean_piece(png_path: str, cnn_size: int = 224) -> Image.Image | None:
    """
    Extracts the non-zero alpha bounding box and letterboxes the rendered piece.
    Returns None if the render buffer is blank or fully transparent.

    @parameters:
        @param png_path: str - Path to rendered snapshot PNG.
        @param cnn_size: int - Final output dimension in pixels.
    @returns:
        Image.Image | None - Composited image, or None if the buffer is empty.
    """
    try:
        with Image.open(png_path) as src_img:
            rgba = src_img.convert("RGBA")
    except Exception:
        return None

    alpha_channel = rgba.split()[3]
    extrema = alpha_channel.getextrema()
    if extrema[1] == 0:
        return None

    bbox = rgba.getbbox()
    if bbox is None:
        return None

    cropped = rgba.crop(bbox)

    max_dim = cnn_size - 24
    cropped.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)

    piece_w, piece_h = cropped.size
    offset_x = (cnn_size - piece_w) // 2
    offset_y = (cnn_size - piece_h) // 2

    canvas = Image.new("RGB", (cnn_size, cnn_size), (255, 255, 255))
    canvas.paste(cropped, (offset_x, offset_y), mask=cropped.split()[3])

    return canvas


def render_single_part_job(
    part_num: str,
    color_name: str,
    rgb_hex: str,
    output_root: str,
    ldview_bin: str,
    library_dir: str,
    rotations: int = 16,
    render_res: int = 256,
    cnn_size: int = 224
) -> int:
    """
    Renders multi-axis orientations using headless execution with alpha verification.

    @parameters:
        @param part_num: str - Target design identifier.
        @param color_name: str - Color name.
        @param rgb_hex: str - Hexadecimal color code.
        @param output_root: str - Target directory.
        @param ldview_bin: str - Path to LDView binary.
        @param library_dir: str - LDraw library root directory.
        @param rotations: int - Number of multi-axis orientations to generate.
        @param render_res: int - LDView render buffer dimension.
        @param cnn_size: int - Output dimension in pixels.
    @returns:
        int - Successfully written image count.
    """
    configure_process_background_priority()

    clean_color = "".join(c for c in color_name if c.isalnum() or c in ("-", "_")).strip()
    target_folder = os.path.join(output_root, f"{part_num}_{clean_color}")
    os.makedirs(target_folder, exist_ok=True)

    hex_clean = (rgb_hex or "808080").lstrip("#").upper()
    r = int(hex_clean[0:2], 16) if len(hex_clean) >= 6 else 128
    g = int(hex_clean[2:4], 16) if len(hex_clean) >= 6 else 128
    b = int(hex_clean[4:6], 16) if len(hex_clean) >= 6 else 128

    if max(r, g, b) < 50:
        hex_clean = "2E3137"

    direct_color = f"0x2{hex_clean}"
    generated_count = 0
    scratch_dir = get_scratch_directory()
    proc_id = os.getpid()

    subprocess_flags = 0
    if sys.platform == "win32":
        subprocess_flags = 0x08000000 | 0x00004000

    for r_idx in range(rotations):
        pitch, yaw, roll = sample_physical_pose(r_idx, rotations)
        matrix_str = compute_orthonormal_euler_matrix(pitch, yaw, roll)

        temp_ldr = os.path.join(scratch_dir, f"_buf_{proc_id}_{part_num}_{r_idx}.ldr")
        temp_png = os.path.join(scratch_dir, f"_buf_{proc_id}_{part_num}_{r_idx}.png")
        final_jpg = os.path.join(target_folder, f"{part_num}_{clean_color}_rot_{r_idx}.jpg")

        ldr_content = (
            f"0 FILE {part_num}.ldr\n"
            f"0 {part_num} {color_name} pose={r_idx}\n"
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
            "-DefaultAngles=1",
            "-FOV=30",
            "-OpenGL=1",
            "-Quality=2",
            "-HiResPrimitives=1",
            "-FSAA=0",
            "-EdgeLines=1",
            "-ConditionalLines=1",
            "-Specular=1",
            "-SpecPower=32",
            "-Ambient=0.30",
            "-Light1=1.2,1.2,1.2,0.4,-0.9,0.5",
            "-Light2=0.5,0.5,0.5,-0.6,-0.4,-0.4",
            "-Light3=0.3,0.3,0.3,0.0,0.9,-0.6",
            "-ProcessEvents=0"
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess_flags,
                timeout=12
            )
            if os.path.isfile(temp_png):
                clean_img = composite_clean_piece(temp_png, cnn_size=cnn_size)
                if clean_img is not None:
                    clean_img.save(final_jpg, "JPEG", quality=95)
                    generated_count += 1
                os.remove(temp_png)
        except Exception:
            pass

        if os.path.exists(temp_ldr):
            os.remove(temp_ldr)

    return generated_count


class GPULDrawBatchPipeline:
    """
    Manages throttled GPU batch rendering with background scheduling and resource management.
    """

    def __init__(
        self,
        library_dir: str = LDRAW_LIBRARY_DIR,
        output_dir: str = DEFAULT_TRAINING_DIR,
        max_workers: int = 4
    ):
        """
        Initializes the rendering pipeline with non-disruptive worker limits.

        @parameters:
            @param library_dir: str - Local directory of the unpacked library.
            @param output_dir: str - Target root destination directory.
            @param max_workers: int - Background worker pool size.
        @returns:
            None
        """
        self.library_dir = library_dir
        self.output_dir = output_dir
        self.ldview_bin = locate_ldview_binary()
        self.reb_db = RebrickableOfflineDB()
        self.max_workers = max_workers
        configure_process_background_priority()

    def run_gpu_batch(
        self,
        rotations_per_part: int = 16,
        render_res: int = 256,
        cnn_size: int = 224,
        max_elements: int | None = None,
        exclude_existing_part_ids: bool = False
    ) -> None:
        """
        Executes batch rendering skipping elements that already have folders.

        @parameters:
            @param rotations_per_part: int - Number of multi-axis views per element.
            @param render_res: int - Render snapshot resolution.
            @param cnn_size: int - Final output image square size.
            @param max_elements: int | None - Cap on total parts processed.
            @param exclude_existing_part_ids: bool - If True, keeps only one mold across colors.
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
            ORDER BY total_count DESC, ip.part_num ASC;
        """

        with self.reb_db.get_connection() as conn:
            rows = conn.execute(query).fetchall()

        parts_dir = os.path.join(self.library_dir, "parts")
        tasks = []
        queued_part_ids = set(existing_part_ids)
        disk_skipped = 0
        color_dedup_skipped = 0
        missing_dat_skipped = 0

        for row in rows:
            part_num = str(row[0])
            color_name = str(row[1])
            rgb_hex = row[2]
            total_count = row[3]

            clean_color = "".join(c for c in color_name if c.isalnum() or c in ("-", "_")).strip()
            folder_name = f"{part_num}_{clean_color}"

            if folder_name in existing_folders:
                disk_skipped += 1
                continue

            if exclude_existing_part_ids and part_num in queued_part_ids:
                color_dedup_skipped += 1
                continue

            part_path = os.path.join(parts_dir, f"{part_num}.dat")
            if not os.path.isfile(part_path):
                missing_dat_skipped += 1
                continue

            tasks.append((part_num, color_name, rgb_hex, total_count))
            if exclude_existing_part_ids:
                queued_part_ids.add(part_num)

            if max_elements is not None and len(tasks) >= max_elements:
                break

        print(f"[FILTER] Skipped on disk: {disk_skipped}")
        print(f"[FILTER] Deduplicated colors: {color_dedup_skipped}")
        print(f"[FILTER] Missing geometry .dat: {missing_dat_skipped}")
        print(f"[QUEUE] Queued {len(tasks)} unique pieces for generation.")

        if not tasks:
            print("[DONE] All elements in catalog already have rendered folders.")
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

        print(f"\n[DONE] Generation complete. Clean renders saved in '{self.output_dir}'.")


if __name__ == "__main__":
    pipeline = GPULDrawBatchPipeline(max_workers=4)
    pipeline.run_gpu_batch(
        rotations_per_part=16,
        render_res=256,
        cnn_size=224,
        exclude_existing_part_ids=True
    )
