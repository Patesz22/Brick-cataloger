import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from db import RebrickableOfflineDB
from bulk_scanner import BulkBrickScanner, DEFAULT_MODEL_PATH, DEFAULT_TRAINING_DIR
from train import execute_training


class BrickCatalogerGUI(tk.Tk):
    """
    Desktop Graphical User Interface for the Brick AI Scanner & Inventory System.
    """

    def __init__(self):
        super().__init__()

        self.title("Brick AI Scanner & Inventory System")
        self.geometry("1280x820")
        self.minsize(1050, 680)

        # State attributes
        self.current_image_path = None
        self.display_pil_image = None
        self.tk_image_reference = None
        self.scanner = None
        self.reb_db = RebrickableOfflineDB()

        self._configure_styles()
        self._build_layout()
        self._check_hardware()

    def _configure_styles(self) -> None:
        """Configures ttk styles for clean visual appearance."""
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TFrame", background="#21252B")
        style.configure("Header.TFrame", background="#1E1E24")
        style.configure("TLabel", background="#21252B", foreground="#E6E6E6", font=("Segoe UI", 10))
        style.configure("HeaderTitle.TLabel", background="#1E1E24", foreground="#F8F9FA", font=("Segoe UI", 13, "bold"))
        style.configure("HeaderSub.TLabel", background="#1E1E24", foreground="#00E676", font=("Segoe UI", 9, "bold"))
        style.configure("Status.TLabel", background="#181A1F", foreground="#ABB2BF", font=("Segoe UI", 9))

        style.configure("Action.TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.configure(
            "Treeview",
            background="#282C34",
            foreground="#ABB2BF",
            fieldbackground="#282C34",
            font=("Segoe UI", 9),
            rowheight=26
        )
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#21252B", foreground="#FFFFFF")
        style.map("Treeview", background=[("selected", "#3E4451")])

    def _build_layout(self) -> None:
        """Constructs the responsive split-panel window hierarchy."""
        # Top Header Bar
        header = ttk.Frame(self, style="Header.TFrame", padding=(16, 12))
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = ttk.Label(header, text="Brick AI Cataloger", style="HeaderTitle.TLabel")
        title_lbl.pack(side=tk.LEFT)

        self.device_lbl = ttk.Label(header, text="Device: Detecting...", style="HeaderSub.TLabel")
        self.device_lbl.pack(side=tk.RIGHT)

        # Toolbar Frame
        toolbar = ttk.Frame(self, padding=(12, 8))
        toolbar.pack(fill=tk.X, side=tk.TOP)

        self.btn_open = ttk.Button(toolbar, text="📁 Select Tray Image", style="Action.TButton",
                                   command=self.on_select_image)
        self.btn_open.pack(side=tk.LEFT, padx=4)

        self.btn_scan = ttk.Button(toolbar, text="🔍 Scan & Detect", style="Action.TButton", command=self.on_start_scan,
                                   state=tk.DISABLED)
        self.btn_scan.pack(side=tk.LEFT, padx=4)

        self.btn_harvest = ttk.Button(toolbar, text="✂️ Harvest Training Crops", style="Action.TButton",
                                      command=self.on_harvest_crops, state=tk.DISABLED)
        self.btn_harvest.pack(side=tk.LEFT, padx=4)

        self.btn_train = ttk.Button(toolbar, text="⚡ Train Model", style="Action.TButton", command=self.on_train_model)
        self.btn_train.pack(side=tk.LEFT, padx=4)

        self.btn_sync = ttk.Button(toolbar, text="🔄 Sync Database", style="Action.TButton",
                                   command=self.on_sync_database)
        self.btn_sync.pack(side=tk.LEFT, padx=4)

        # Main Split Content Area
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#181A1F", bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: Image Visualizer with Canvas
        left_frame = ttk.Frame(paned, padding=6)
        paned.add(left_frame, minsize=480)

        canvas_header = ttk.Label(left_frame, text="TRAY VIEW & DETECTION BOXES", font=("Segoe UI", 9, "bold"),
                                  foreground="#98C379")
        canvas_header.pack(anchor=tk.W, pady=(0, 4))

        self.canvas_container = tk.Frame(left_frame, bg="#181A1F", highlightthickness=1, highlightbackground="#3E4451")
        self.canvas_container.pack(fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(self.canvas_container, bg="#181A1F", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Configure>", self._on_canvas_resize)

        # Right: Detection Table
        right_frame = ttk.Frame(paned, padding=6)
        paned.add(right_frame, minsize=380)

        table_header = ttk.Label(right_frame, text="DETECTED INVENTORY BREAKDOWN", font=("Segoe UI", 9, "bold"),
                                 foreground="#61AFEF")
        table_header.pack(anchor=tk.W, pady=(0, 4))

        tree_scroll = ttk.Scrollbar(right_frame, orient=tk.VERTICAL)
        columns = ("part_num", "part_name", "color", "confidence", "element_id", "source")
        self.tree = ttk.Treeview(right_frame, columns=columns, show="headings", yscrollcommand=tree_scroll.set)
        tree_scroll.config(command=self.tree.yview)

        self.tree.heading("part_num", text="Part #")
        self.tree.heading("part_name", text="Part Name")
        self.tree.heading("color", text="Color")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("element_id", text="Element ID")
        self.tree.heading("source", text="Source")

        self.tree.column("part_num", width=65, anchor=tk.CENTER)
        self.tree.column("part_name", width=140, anchor=tk.W)
        self.tree.column("color", width=85, anchor=tk.CENTER)
        self.tree.column("confidence", width=75, anchor=tk.CENTER)
        self.tree.column("element_id", width=75, anchor=tk.CENTER)
        self.tree.column("source", width=95, anchor=tk.CENTER)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom Status Bar
        self.status_bar = ttk.Label(self, text="Ready. Select an image to begin.", style="Status.TLabel",
                                    padding=(12, 4))
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _check_hardware(self) -> None:
        """Determines active PyTorch computation device."""
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            self.device_lbl.config(text=f"CUDA: {dev_name} (Ready)", foreground="#00E676")
        else:
            self.device_lbl.config(text="Hardware: CPU Mode", foreground="#E5C07B")

    def _set_status(self, text: str) -> None:
        """Thread-safe status bar updater."""
        self.status_bar.config(text=text)

    def _on_canvas_resize(self, event=None) -> None:
        """Re-renders canvas when the window or panel is resized."""
        if self.display_pil_image:
            self._render_image_to_canvas(self.display_pil_image)

    def _render_image_to_canvas(self, pil_img: Image.Image) -> None:
        """Letterboxes and fits a PIL Image into the active Canvas view."""
        cw = max(10, self.image_canvas.winfo_width())
        ch = max(10, self.image_canvas.winfo_height())

        img_w, img_h = pil_img.size
        ratio = min(cw / img_w, ch / img_h)
        target_w = max(1, int(img_w * ratio))
        target_h = max(1, int(img_h * ratio))

        resized = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        self.tk_image_reference = ImageTk.PhotoImage(resized)

        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            cw // 2,
            ch // 2,
            image=self.tk_image_reference,
            anchor=tk.CENTER
        )

    def on_select_image(self) -> None:
        """Prompts for an image using the native file picker dialog."""
        path = filedialog.askopenfilename(
            title="Select Bulk Brick Tray Photo",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        if not path or not os.path.isfile(path):
            return

        self.current_image_path = os.path.normpath(path)
        self.display_pil_image = Image.open(self.current_image_path).convert("RGB")
        self._render_image_to_canvas(self.display_pil_image)

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.btn_scan.config(state=tk.NORMAL)
        self.btn_harvest.config(state=tk.NORMAL)
        self._set_status(
            f"Loaded: {os.path.basename(self.current_image_path)} ({self.display_pil_image.width}x{self.display_pil_image.height})")

    def on_start_scan(self) -> None:
        """Triggers the inference pipeline asynchronously."""
        if not self.current_image_path:
            return

        self.btn_scan.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self._set_status("Scanning tray, isolating pieces, and projecting features...")

        def run_thread():
            try:
                if self.scanner is None:
                    self.scanner = BulkBrickScanner(model_path=DEFAULT_MODEL_PATH)

                annotated_path, detections = self.scanner.process_tray_image(self.current_image_path)
                self.after(0, lambda p=annotated_path, d=detections: self._on_scan_completed(p, d))
            except Exception as err:
                err_str = str(err)
                self.after(0, lambda msg=err_str: self._on_scan_failed(msg))

        threading.Thread(target=run_thread, daemon=True).start()

    def _on_scan_completed(self, annotated_path: str, detections: list[dict]) -> None:
        """Updates canvas with bounding boxes and fills the data table."""
        self.btn_scan.config(state=tk.NORMAL)
        self.btn_open.config(state=tk.NORMAL)

        if os.path.isfile(annotated_path):
            self.display_pil_image = Image.open(annotated_path).convert("RGB")
            self._render_image_to_canvas(self.display_pil_image)

        for item in self.tree.get_children():
            self.tree.delete(item)

        for d in detections:
            conf_str = f"{d['confidence'] * 100:.1f}%"
            self.tree.insert("", tk.END, values=(
                d["part_num"],
                d["part_name"],
                d["color_name"],
                conf_str,
                d["element_id"],
                d["source"]
            ))

        self._set_status(f"Completed scan. Detected and cataloged {len(detections)} parts.")

    def _on_scan_failed(self, error_msg: str) -> None:
        """Handles scan runtime exceptions."""
        self.btn_scan.config(state=tk.NORMAL)
        self.btn_open.config(state=tk.NORMAL)
        self._set_status(f"Scan failed: {error_msg}")
        messagebox.showerror("Scan Error", f"Failed to complete scan:\n\n{error_msg}")

    def on_harvest_crops(self) -> None:
        """Extracts piece crops and adds them to the training dataset."""
        if not self.current_image_path:
            return

        self._set_status("Harvesting training crops from tray photo...")

        def run_thread():
            try:
                if self.scanner is None:
                    self.scanner = BulkBrickScanner(model_path=DEFAULT_MODEL_PATH)

                count = self.scanner.harvest_tray_crops_for_training(
                    image_path=self.current_image_path,
                    output_dir=DEFAULT_TRAINING_DIR
                )
                self.after(0, lambda c=count: messagebox.showinfo("Harvest Complete",
                                                                  f"Successfully extracted {c} crops into '{DEFAULT_TRAINING_DIR}'."))
                self.after(0, lambda c=count: self._set_status(f"Harvested {c} crops into dataset."))
            except Exception as err:
                err_str = str(err)
                self.after(0, lambda msg=err_str: messagebox.showerror("Harvest Error", msg))

        threading.Thread(target=run_thread, daemon=True).start()

    def on_train_model(self) -> None:
        """Launches background training loop with focal loss & ArcFace."""
        if not os.path.isdir(DEFAULT_TRAINING_DIR) or len(os.listdir(DEFAULT_TRAINING_DIR)) == 0:
            messagebox.showwarning("No Data",
                                   f"Training folder '{DEFAULT_TRAINING_DIR}' is empty. Generate data first.")
            return

        confirm = messagebox.askyesno("Train Model", "Start training EfficientNet-B0 with ArcFace & Focal Loss?")
        if not confirm:
            return

        self.btn_train.config(state=tk.DISABLED)
        self._set_status("Training in progress (Linear probe + fine-tune)... check console for epoch loss.")

        def run_thread():
            try:
                execute_training(data_dir=DEFAULT_TRAINING_DIR, epochs=15, batch_size=32)
                self.after(0, lambda: messagebox.showinfo("Training Complete",
                                                          "Model training finished and weights saved."))
                self.after(0, lambda: self._set_status("Training completed successfully."))
            except Exception as err:
                err_str = str(err)
                self.after(0, lambda msg=err_str: messagebox.showerror("Training Error", msg))
            finally:
                self.after(0, lambda: self.btn_train.config(state=tk.NORMAL))

        threading.Thread(target=run_thread, daemon=True).start()

    def on_sync_database(self) -> None:
        """Runs the Rebrickable offline database sync."""
        self._set_status("Verifying Rebrickable catalog database...")

        def run_thread():
            try:
                self.reb_db.sync_database()
                self.after(0, lambda: messagebox.showinfo("Database", "Rebrickable database is synced and up to date."))
                self.after(0, lambda: self._set_status("Database is up to date."))
            except Exception as err:
                err_str = str(err)
                self.after(0, lambda msg=err_str: messagebox.showerror("Database Error", msg))

        threading.Thread(target=run_thread, daemon=True).start()


if __name__ == "__main__":
    app = BrickCatalogerGUI()
    app.mainloop()
