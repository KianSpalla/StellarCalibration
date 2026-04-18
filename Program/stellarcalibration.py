import math
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from pipeline import run_calibration

BG       = "#0d1117"
SURFACE  = "#161b22"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
ACCENT_H = "#79b8ff"
FG       = "#e6edf3"
FG_DIM   = "#8b949e"
SUCCESS  = "#3fb950"
ERROR    = "#f85149"
WARN     = "#d29922"

FONT       = ("Segoe UI", 10)
FONT_SB    = ("Segoe UI", 10, "bold")
FONT_LG    = ("Segoe UI", 12, "bold")
FONT_TITLE = ("Segoe UI", 17, "bold")
FONT_MONO  = ("Consolas", 9)

THUMB_SIZE = (280, 196)
WIN_WIDTH  = 640
WIN_HEIGHT = 760


def _to_displayable(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode in ("RGB", "RGBA", "L", "P"):
        return pil_img

    import numpy as np
    arr = np.array(pil_img, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr[:] = 0
    return Image.fromarray(arr.astype(np.uint8), mode="L")


class HoverButton(tk.Button):
    def __init__(self, master, bg_normal: str, bg_hover: str,
                 fg_normal: str = FG, **kw):
        super().__init__(
            master,
            bg=bg_normal,
            fg=fg_normal,
            activebackground=bg_hover,
            activeforeground=fg_normal,
            relief="flat",
            cursor="hand2",
            **kw,
        )
        self._bg_n = bg_normal
        self._bg_h = bg_hover

        self.bind("<Enter>", lambda _: self.config(bg=bg_hover))
        self.bind("<Leave>", lambda _: self.config(bg=bg_normal))


class StarCalibrationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._result: dict | None = None
        self._thumb_ref = None
        self._running = False

        self._build_window()
        self._build_ui()

    def _build_window(self):
        r = self.root
        r.title("Star Calibration")
        r.configure(bg=BG)
        r.resizable(True, True)
        r.minsize(420, 580)

        r.update_idletasks()
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        x = (sw - WIN_WIDTH)  // 2
        y = (sh - WIN_HEIGHT) // 2
        r.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{x}+{y}")

        try:
            _base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
            _ico = os.path.join(_base, "stars.ico")
            r.iconbitmap(_ico)
        except Exception:
            pass

        style = ttk.Style(r)
        style.theme_use("default")
        style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor=SURFACE,
            background=ACCENT,
            borderwidth=0,
            thickness=6,
        )
        style.configure("TSeparator", background=BORDER)

        # Colour the native Windows title bar to match BG
        try:
            import ctypes
            DWMWA_CAPTION_COLOR = 35
            # COLORREF is 0x00BBGGRR
            r_val = int(BG[1:3], 16)
            g_val = int(BG[3:5], 16)
            b_val = int(BG[5:7], 16)
            colorref = ctypes.c_uint32(r_val | (g_val << 8) | (b_val << 16))
            hwnd = ctypes.windll.user32.GetParent(r.winfo_id())
            if hwnd == 0:
                hwnd = r.winfo_id()
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_CAPTION_COLOR,
                ctypes.byref(colorref), ctypes.sizeof(colorref),
            )
        except Exception:
            pass

    def _build_ui(self):
        root = self.root

        # Thin accent bar at very top
        tk.Frame(root, bg=ACCENT, height=3).pack(fill="x")

        hdr = tk.Frame(root, bg=SURFACE, pady=22)
        hdr.pack(fill="x")

        # Logo + title row
        title_row = tk.Frame(hdr, bg=SURFACE)
        title_row.pack()
        tk.Label(
            title_row, text="★",
            font=("Segoe UI", 20), bg=SURFACE, fg=ACCENT,
        ).pack(side="left", padx=(0, 10))
        title_col = tk.Frame(title_row, bg=SURFACE)
        title_col.pack(side="left")
        tk.Label(
            title_col, text="Star Calibration",
            font=FONT_TITLE, bg=SURFACE, fg=FG, anchor="w",
        ).pack(anchor="w")
        tk.Label(
            title_col,
            text="GONet all-sky image · SIMBAD star catalogue",
            font=("Segoe UI", 9), bg=SURFACE, fg=FG_DIM, anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        ttk.Separator(root).pack(fill="x")

        # ── Scrollable body ───────────────────────────────────────────────
        scroll_outer = tk.Frame(root, bg=BG)
        scroll_outer.pack(fill="both", expand=True)

        style = ttk.Style(root)
        style.configure("Dark.Vertical.TScrollbar",
                        troughcolor=BG, background=BG,
                        arrowcolor=BG, borderwidth=0, width=6)
        style.map("Dark.Vertical.TScrollbar",
                  background=[("active", BORDER), ("pressed", FG_DIM)])

        vscroll = ttk.Scrollbar(scroll_outer, orient="vertical",
                                style="Dark.Vertical.TScrollbar")
        vscroll.pack(side="right", fill="y")

        canvas = tk.Canvas(scroll_outer, bg=BG, highlightthickness=0,
                           yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.config(command=canvas.yview)

        body = tk.Frame(canvas, bg=BG, padx=28, pady=20)
        _body_id = canvas.create_window((0, 0), window=body, anchor="nw")

        def _on_body_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(_body_id, width=event.width)

        body.bind("<Configure>", _on_body_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        root.bind_all("<MouseWheel>", _on_mousewheel)

        tk.Label(
            body, text="Image File", font=FONT_SB,
            bg=BG, fg=FG_DIM, anchor="w",
        ).pack(fill="x")

        file_row = tk.Frame(body, bg=BG)
        file_row.pack(fill="x", pady=(4, 14))

        self.file_var = tk.StringVar()

        file_entry = tk.Entry(
            file_row,
            textvariable=self.file_var,
            font=FONT_MONO, bg=SURFACE, fg=FG,
            insertbackground=FG,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        file_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        HoverButton(
            file_row, bg_normal=SURFACE, bg_hover=BORDER,
            text="Browse…", font=FONT_SB,
            command=self._select_file, padx=12, pady=5,
        ).pack(side="left")

        self.thumb_frame = tk.Frame(
            body, bg=SURFACE,
            width=THUMB_SIZE[0], height=THUMB_SIZE[1],
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.thumb_frame.pack_propagate(False)
        self.thumb_frame.pack(pady=(0, 18))

        self.thumb_label = tk.Label(
            self.thumb_frame, bg=SURFACE, fg=FG_DIM,
            font=FONT, text="No image selected",
        )
        self.thumb_label.place(relx=0.5, rely=0.5, anchor="center")

        # ── Detection / catalog parameters ──────────────────────────────────
        tk.Label(body, text="PARAMETERS", font=("Segoe UI", 8, "bold"),
                 bg=BG, fg=FG_DIM, anchor="w").pack(fill="x", pady=(0, 6))

        params_card = tk.Frame(body, bg=SURFACE,
                               highlightthickness=1, highlightbackground=BORDER,
                               padx=16, pady=14)
        params_card.pack(fill="x", pady=(0, 14))

        params_row_1 = tk.Frame(params_card, bg=SURFACE)
        params_row_1.pack(fill="x", pady=(0, 10))

        tk.Label(params_row_1, text="Min pixels:",
                 font=FONT, bg=SURFACE, fg=FG_DIM).pack(side="left")
        self.pixel_min_var = tk.IntVar(value=5)
        tk.Spinbox(
            params_row_1, from_=1, to=200, increment=1,
            textvariable=self.pixel_min_var, width=5,
            font=FONT, bg=BG, fg=FG,
            buttonbackground=BORDER, relief="flat",
            insertbackground=FG,
        ).pack(side="left", padx=(6, 24))

        tk.Label(params_row_1, text="Max pixels:",
                 font=FONT, bg=SURFACE, fg=FG_DIM).pack(side="left")
        self.pixel_max_var = tk.IntVar(value=50)
        tk.Spinbox(
            params_row_1, from_=2, to=1000, increment=1,
            textvariable=self.pixel_max_var, width=6,
            font=FONT, bg=BG, fg=FG,
            buttonbackground=BORDER, relief="flat",
            insertbackground=FG,
        ).pack(side="left", padx=(6, 0))

        params_row_2 = tk.Frame(params_card, bg=SURFACE)
        params_row_2.pack(fill="x")

        tk.Label(params_row_2, text="Radius (deg):",
                 font=FONT, bg=SURFACE, fg=FG_DIM).pack(side="left")
        self.radius_var = tk.DoubleVar(value=60.0)
        tk.Spinbox(
            params_row_2, from_=10.0, to=90.0, increment=1.0,
            textvariable=self.radius_var, width=6,
            font=FONT, bg=BG, fg=FG,
            buttonbackground=BORDER, relief="flat",
            insertbackground=FG,
        ).pack(side="left", padx=(6, 24))

        tk.Label(params_row_2, text="Section size:",
                 font=FONT, bg=SURFACE, fg=FG_DIM).pack(side="left")
        self.section_size_var = tk.IntVar(value=200)
        tk.Spinbox(
            params_row_2, from_=50, to=1000, increment=10,
            textvariable=self.section_size_var, width=6,
            font=FONT, bg=BG, fg=FG,
            buttonbackground=BORDER, relief="flat",
            insertbackground=FG,
        ).pack(side="left", padx=(6, 24))

        tk.Label(params_row_2, text="Vmag limit:",
                 font=FONT, bg=SURFACE, fg=FG_DIM).pack(side="left")
        self.vmax_var = tk.DoubleVar(value=2.5)
        tk.Spinbox(
            params_row_2, from_=1.0, to=8.0, increment=0.5,
            textvariable=self.vmax_var, width=5,
            font=FONT, bg=BG, fg=FG,
            buttonbackground=BORDER, relief="flat",
            insertbackground=FG,
        ).pack(side="left", padx=(6, 0))

        # ── Options row ───────────────────────────────────────────────────
        opts_row = tk.Frame(body, bg=BG)
        opts_row.pack(fill="x", pady=(0, 14))

        self.show_plots_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            opts_row, text="Show diagnostic plots after run",
            variable=self.show_plots_var,
            font=FONT, bg=BG, fg=FG_DIM,
            activebackground=BG, activeforeground=FG,
            selectcolor=SURFACE,
        ).pack(side="left")

        self.run_btn = HoverButton(
            body, bg_normal=ACCENT, bg_hover=ACCENT_H,
            fg_normal=BG,
            text="▶  Run Calibration",
            font=FONT_LG,
            command=self._start_calibration,
            padx=24, pady=10,
        )
        self.run_btn.pack(fill="x", pady=(0, 16))

        self.progress = ttk.Progressbar(
            body, mode="indeterminate",
            style="Dark.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(0, 6))

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            body, textvariable=self.status_var,
            font=("Segoe UI", 9), bg=BG, fg=FG_DIM,
        ).pack()

        ttk.Separator(body).pack(fill="x", pady=16)

        self.results_outer = tk.Frame(body, bg=BG)

        tk.Label(
            self.results_outer, text="RESULTS",
            font=("Segoe UI", 8, "bold"), bg=BG, fg=FG_DIM, anchor="w",
        ).pack(fill="x", pady=(0, 8))

        self.results_box = tk.Frame(
            self.results_outer, bg=SURFACE,
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.results_box.pack(fill="x")

        self.result_labels: dict = {}
        metrics = [
            ("score",  "Match Score"),
            ("rms",    "RMS Error"),
            ("zenith", "Zenith Pixel"),
            ("shift",  "Applied Shift"),
        ]
        for col in range(2):
            self.results_box.columnconfigure(col, weight=1)

        for i, (key, display_name) in enumerate(metrics):
            r, c = divmod(i, 2)
            cell = tk.Frame(self.results_box, bg=SURFACE, padx=18, pady=14)
            cell.grid(row=r, column=c, sticky="nsew",
                      padx=(0, 1 if c == 0 else 0),
                      pady=(0, 1 if r == 0 else 0))
            # Thin divider lines between cells via 1px border color
            cell.config(highlightthickness=1 if c == 0 else 0,
                        highlightbackground=BORDER)

            tk.Label(cell, text=display_name.upper(),
                     font=("Segoe UI", 7, "bold"),
                     bg=SURFACE, fg=FG_DIM, anchor="w").pack(anchor="w")

            lbl = tk.Label(cell, text="—",
                           font=("Segoe UI", 12, "bold"),
                           bg=SURFACE, fg=FG, anchor="w")
            lbl.pack(anchor="w", pady=(4, 0))
            self.result_labels[key] = lbl

        btn_row = tk.Frame(self.results_outer, bg=BG)
        btn_row.pack(fill="x", pady=(10, 0))

        self.save_btn = HoverButton(
            btn_row,
            bg_normal=SURFACE, bg_hover=BORDER,
            text="⬇  Save Shifted Image",
            font=FONT_SB, padx=16, pady=7,
            command=self._save_shiftedImage,
        )
        self.save_btn.pack(side="right")

    def _select_file(self):
        path = filedialog.askopenfilename(
            title="Select GONet Image",
            filetypes=[
                ("All Files",  "*.*"),
                ("JPG Files",  "*.jpg"),
                ("JPEG Files", "*.jpeg"),
            ],
        )
        if not path:
            return

        self.file_var.set(path)
        self._load_thumbnail(path)
        self._hide_results()
        self.status_var.set("Ready")

    def _load_thumbnail(self, path: str):
        try:
            img   = _to_displayable(Image.open(path))
            img.thumbnail(THUMB_SIZE, Image.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self._thumb_ref = photo
            self.thumb_label.config(image=photo, text="")
        except Exception:
            self.thumb_label.config(image="", text="Preview unavailable")

    def _start_calibration(self):
        path = self.file_var.get().strip()
        if not path:
            messagebox.showwarning("No file selected",
                                   "Please select a GONet image file first.")
            return

        if self._running:
            return

        self._running = True
        self._result  = None

        self.run_btn.config(state="disabled", text="  Processing…")
        self.progress.start(12)
        self.status_var.set("Running calibration — this may take a minute…")
        self._hide_results()

        try:
            pixel_min = int(self.pixel_min_var.get())
            pixel_max = int(self.pixel_max_var.get())
            radius_deg = float(self.radius_var.get())
            section_size = int(self.section_size_var.get())
            vmax = float(self.vmax_var.get())
        except (tk.TclError, ValueError):
            pixel_min, pixel_max, radius_deg, section_size, vmax = 5, 50, 60.0, 200, 2.5

        if pixel_max < pixel_min:
            pixel_min, pixel_max = pixel_max, pixel_min

        radius_deg = max(1.0, min(90.0, radius_deg))
        section_size = max(10, section_size)

        show_plots = bool(self.show_plots_var.get())
        threading.Thread(
            target=self._worker,
            args=(path, vmax, pixel_min, pixel_max, radius_deg, section_size, show_plots),
            daemon=True,
        ).start()

    def _worker(
        self,
        path: str,
        vmax: float,
        pixel_min: int,
        pixel_max: int,
        radius_deg: float,
        section_size: int,
        show_plots: bool,
    ):
        try:
            result = run_calibration(
                path,
                show_plots=False,
                vmag=vmax,
                pixelMin=pixel_min,
                pixelMax=pixel_max,
                catalogRadiusDeg=radius_deg,
                sectionSize=section_size,
            )
            result["_show_plots"] = show_plots
            self.root.after(0, lambda: self._on_success(result))
        except Exception as exc:
            msg = str(exc)
            self.root.after(0, lambda m=msg: self._on_error(m))

    def _on_success(self, result: dict):
        self._result  = result
        self._running = False

        self.progress.stop()
        self.run_btn.config(state="normal", text="▶  Run Calibration")
        self.status_var.set("Calibration complete.")

        best          = result["best"]
        centerResult = result["centerResult"]

        rms_text = (
            f"{best['rms_pix']:.2f} px"
            if not math.isnan(best["rms_pix"])
            else "n/a"
        )

        zenith_text = (
            f"x={centerResult['zenithX']:.1f}  "
            f"y={centerResult['zenithY']:.1f} px"
        )
        shift_text = (
            f"dx={centerResult['shiftX']:+.1f}  "
            f"dy={centerResult['shiftY']:+.1f} px"
        )

        self.result_labels["score"].config(
            text=f"{best['score']} matches",
            fg=SUCCESS if best["score"] > 5 else WARN,
        )
        self.result_labels["rms"].config(text=rms_text)
        self.result_labels["zenith"].config(text=zenith_text)
        self.result_labels["shift"].config(text=shift_text)

        self._show_results()
        self._open_preview(result)
        if result.get("_show_plots"):
            self._show_diagnostic_plots(result)

    def _show_diagnostic_plots(self, result: dict):
        import numpy as np
        import matplotlib.pyplot as plt
        best          = result["best"]
        centerResult = result["centerResult"]
        centered_img = centerResult.get("centered_sub", centerResult.get("centeredSub"))
        img           = result["img"]
        imgXY        = result["imgXY"]
        imgMean      = float(np.mean(img))
        imgStd       = float(np.std(img))

        plt.figure()
        plt.imshow(img, origin="upper", cmap="gray",
                   vmin=imgMean - 2 * imgStd, vmax=imgMean + 5 * imgStd)
        plt.scatter(imgXY[:, 0], imgXY[:, 1],
                    s=50, edgecolor="red", facecolor="none", label="Image sources")
        plt.scatter(best["predictedXY"][:, 0], best["predictedXY"][:, 1],
                    s=50, edgecolor="blue", facecolor="none", label="Predicted sources")

        # Draw lines between matched catalog predictions and their detected sources.
        matchedCatalogIdx = best.get("matched_catalog_indices", best.get("matched_catallog_indicies", np.array([], dtype=int)))
        matchedImageIdx = best.get("matched_image_indices", best.get("matched_image_indicies", np.array([], dtype=int)))
        matchedCatalogSet = set(np.asarray(matchedCatalogIdx, dtype=int))

        catalogNames = result.get("catalogNames", [])
        for catIdx, srcIdx in zip(np.asarray(matchedCatalogIdx, dtype=int), np.asarray(matchedImageIdx, dtype=int)):
            if 0 <= catIdx < len(best["predictedXY"]) and 0 <= srcIdx < len(imgXY):
                catX, catY = best["predictedXY"][catIdx]
                srcX, srcY = imgXY[srcIdx]
                plt.plot([catX, srcX], [catY, srcY], color="lime", linewidth=0.8, alpha=0.7)
        plt.plot([], [], color="lime", linewidth=0.8, label="Matched pairs")

        #Label matched catalog stars that have a common name
        for catIdx in range(len(best["predictedXY"])):
            if catIdx in matchedCatalogSet and catIdx < len(catalogNames) and catalogNames[catIdx]:
                px, py = best["predictedXY"][catIdx]
                plt.annotate(catalogNames[catIdx], (px, py), color="white",
                             fontsize=7, ha="left", va="bottom",
                             xytext=(4, 4), textcoords="offset points")

        plt.scatter([centerResult["targetCenterX"]], [centerResult["targetCenterY"]],
                    s=100, marker="+", c="yellow", label="Target center")
        plt.scatter([centerResult["zenithX"]], [centerResult["zenithY"]],
                    s=120, marker="x", c="cyan", label="Zenith")
        plt.plot(
            [centerResult["zenithX"], centerResult["targetCenterX"]],
            [centerResult["zenithY"], centerResult["targetCenterY"]],
            color="cyan", linestyle="--", linewidth=1.5, label="Applied shift",
        )
        plt.legend()
        plt.title(f"Best score: {best['score']} matches")
        plt.show(block=False)

        plt.figure()
        plt.imshow(centered_img, origin="upper", cmap="gray",
                   vmin=imgMean - 2 * imgStd, vmax=imgMean + 5 * imgStd)
        plt.scatter([centerResult["targetCenterX"]], [centerResult["targetCenterY"]],
                    s=120, marker="x", c="cyan", label="Centered zenith target")
        plt.legend()
        plt.title("Image shifted so zenith is centered")
        plt.show(block=False)

    def _on_error(self, msg: str):
        self._running = False
        self.progress.stop()
        self.run_btn.config(state="normal", text="▶  Run Calibration")
        self.status_var.set("Calibration failed.")
        messagebox.showerror("Calibration Failed", msg)

    def _show_results(self):
        self.results_outer.pack(fill="x")

    def _hide_results(self):
        self.results_outer.pack_forget()

    def _open_preview(self, result: dict):
        shiftedImage = result.get("shiftedImage") or result.get("shifted_image")
        shiftedFormat = result.get("shiftedFormat") or result.get("shifted_format")
        suggested_ext = result.get("suggested_suffix", ".png")

        if shiftedImage is None:
            messagebox.showinfo(
                "No preview",
                "Calibration finished but no shifted image was returned.",
            )
            return

        win = tk.Toplevel(self.root)
        win.title("Shifted Image Preview")
        win.configure(bg=BG)
        win.resizable(True, True)

        hdr = tk.Frame(win, bg=SURFACE, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Calibrated Image Preview",
                 font=FONT_LG, bg=SURFACE, fg=FG).pack()

        preview = _to_displayable(shiftedImage.copy())
        preview.thumbnail((1000, 700), Image.LANCZOS)

        photo     = ImageTk.PhotoImage(preview)
        imgLabel = tk.Label(win, image=photo, bg=BG)
        imgLabel.image = photo
        imgLabel.pack(padx=16, pady=12)

        btn_bar = tk.Frame(win, bg=BG)
        btn_bar.pack(pady=(0, 14))

        def _save():
            save_path = filedialog.asksaveasfilename(
                title="Save Shifted Image",
                defaultextension=suggested_ext,
                filetypes=[
                    ("JPEG Files", "*.jpg;*.jpeg"),
                    ("PNG Files",  "*.png"),
                    ("TIFF Files", "*.tif;*.tiff"),
                    ("All Files",  "*.*"),
                ],
            )
            if not save_path:
                return
            try:
                # Let PIL infer format from extension; fallback to provided format only if needed.
                try:
                    shiftedImage.save(save_path)
                except Exception:
                    if shiftedFormat:
                        shiftedImage.save(save_path, format=shiftedFormat)
                    else:
                        raise
                messagebox.showinfo("Saved", f"Shifted image saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Save Failed", str(e))

        HoverButton(
            btn_bar, bg_normal=ACCENT, bg_hover=ACCENT_H,
            fg_normal=BG, text="⬇  Save Image",
            font=FONT_SB, padx=16, pady=6,
            command=_save,
        ).pack(side="left", padx=6)

        HoverButton(
            btn_bar, bg_normal=SURFACE, bg_hover=BORDER,
            text="Close", font=FONT_SB, padx=16, pady=6,
            command=win.destroy,
        ).pack(side="left", padx=6)

    def _save_shiftedImage(self):
        if self._result is None:
            return
        self._open_preview(self._result)


root = tk.Tk()

app  = StarCalibrationApp(root)
root.mainloop()
