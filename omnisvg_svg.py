# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
OmniSVG detokenizer + SVG renderer, self-contained.

The original plugin depended on the vendored DeepSVG package just to turn the
14-wide command tensors back into an SVG path string, and on CairoSVG to
rasterise it.  DeepSVG's `svglib/svg.py` imports cairosvg, IPython, networkx and
moviepy at module level, so the whole plugin failed to import unless a full
Cairo stack was installed (which is painful on Windows).

Both are replaced here:

  * token stream -> xy commands -> 14-wide command rows: straight port of
    `SVGTokenizer.process_generated_tokens` / `raster_svg` (identical maths,
    identical token ranges, no third-party dependency);
  * command rows -> SVG `d` attribute: ~40 lines instead of DeepSVG's object
    graph (Move / Line / Cubic / Arc / Close are emitted exactly like
    `SVGPath.to_str` + `SVGPathGroup.to_str` did);
  * SVG -> raster: `pillow-lunasvg`, a self-contained wheel that plugs into
    Pillow, no system libraries.
"""

import io
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image

try:
    import pillow_lunasvg  # noqa: F401  (registers the Pillow SVG plugin)
    _LUNASVG_OK = True
except Exception:  # pragma: no cover - only if the optional wheel is absent
    _LUNASVG_OK = False


# SVGTensor.COMMANDS_SIMPLIFIED -> SVG path command letter
# 0=m 1=l 2=c 3=a 4=EOS 5=SOS 6=z
_CMD_LETTERS = {0: "M", 1: "L", 2: "C", 3: "A", 6: "Z"}


def _num(v) -> str:
    f = float(v)
    if f == int(f):
        return str(int(f))
    return f"{f:.4f}".rstrip("0").rstrip(".")


class SvgCodec:
    """
    OmniSVG token codec, driven by the vendored config.yaml (model specific
    token offsets for the 4B / 8B checkpoints).
    """

    def __init__(self, config_path: str, model_size: str = "4B"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.model_size = model_size
        if model_size not in self.config.get("models", {}):
            raise ValueError(f"Invalid model_size: {model_size}. "
                             f"Must be one of {list(self.config.get('models', {}).keys())}")
        self._load_config()
        self.coord_ids = self._create_coord_id_table()

    # ------------------------------------------------------------------ #
    # config
    # ------------------------------------------------------------------ #
    def _model_value(self, *keys):
        value = self.config.get("models", {}).get(self.model_size, {})
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break
        if value is None:
            value = self.config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
        return value

    def _load_config(self):
        self.NUM_MASK_AND_EOM = self._model_value("tokens", "num_mask_and_eom")
        self.BASE_OFFSET = self._model_value("tokens", "base_offset")

        tokens_cfg = self.config["tokens"]
        self.NUM_SVG_END = tokens_cfg["svg_end"]

        self.PIX_PAD = self._model_value("coordinates", "pix_pad_offset")
        self.COORD_PAD = self._model_value("coordinates", "coord_pad_offset")
        self.BBOX = self.config["coordinates"]["bbox"]

        colors_cfg = self.config["colors"]
        self.COLOR_TOKEN_START_RAW = colors_cfg["color_token_start"]
        self.MAX_COLOR_TOKENS = colors_cfg["max_color_tokens"]

        commands_cfg = self.config["svg_commands"]
        self.CMD_MOVE = commands_cfg["move"]
        self.CMD_LINE = commands_cfg["line"]
        self.CMD_CURVE = commands_cfg["curve"]
        self.CMD_ARC = commands_cfg["arc"]
        self.CMD_CLOSE = commands_cfg["close"]

        model_cfg = self.config["model"]
        self.BOS_TOKEN_ID = model_cfg["bos_token_id"]
        self.EOS_TOKEN_ID = model_cfg["eos_token_id"]
        self.PAD_TOKEN_ID = model_cfg["pad_token_id"]

        arc_cfg = self.config.get("arc", {})
        self.ARC_PARAM_OFFSET = arc_cfg.get("param_offset", 44500)
        self.ARC_PARAM_RANGE = arc_cfg.get("param_range", 100)
        self.ARC_PARAM_START = self.ARC_PARAM_OFFSET + self.BASE_OFFSET

        self.PIXEL_OFFSET = (self.NUM_MASK_AND_EOM - self.BASE_OFFSET +
                             self.NUM_SVG_END - self.CMD_MOVE)
        self.CMD_TOKEN_START = self.NUM_MASK_AND_EOM + self.NUM_SVG_END
        self.CMD_TOKEN_END = self.PIX_PAD + self.NUM_SVG_END
        self.COORD_TOKEN_START = self.PIX_PAD + self.NUM_SVG_END
        self.COLOR_COORD_BOUNDARY = self.COLOR_TOKEN_START_RAW + 1 + self.BASE_OFFSET
        self.COLOR_THRESHOLD = self.COLOR_TOKEN_START_RAW - self.PIXEL_OFFSET + 1
        # Colour block layout, verified against a live 4B generation:
        #   token 191946 = sentinel (dropped), colours occupy
        #   [191947, 196044] == MAX_COLOR_TOKENS (4098) entries, and the model
        #   does emit 196044 for white (the last, brightest colour).  The
        #   reference reconstruction `vocab_token - BASE_OFFSET - 1` lands the
        #   whole block on 40010..44107, i.e. none / currentColor / 4096 colours.

    def _create_coord_id_table(self) -> np.ndarray:
        """COORD_TOKEN_START + pixel index -> token id, row-major 200x200 grid."""
        x = np.linspace(0, self.BBOX - 1, self.BBOX)
        y = np.linspace(0, self.BBOX - 1, self.BBOX)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(int)
        return xy + self.COORD_PAD + self.NUM_SVG_END

    # ------------------------------------------------------------------ #
    # colour token -> css colour (dataset.py logic)
    # ------------------------------------------------------------------ #
    def token_to_color(self, color_token: int) -> str:
        try:
            if color_token == self.COLOR_TOKEN_START_RAW:
                return "none"
            if color_token == self.COLOR_TOKEN_START_RAW + 1:
                return "currentColor"
            color_index = color_token - (self.COLOR_TOKEN_START_RAW + 2)
            if color_index < 0 or color_index >= self.MAX_COLOR_TOKENS:
                print(f"[OmniSVG] colour token {color_token} out of range")
                return "#808080"
            r = (color_index >> 8) & 0xF
            g = (color_index >> 4) & 0xF
            b = color_index & 0xF
            r, g, b = (r << 4) | r, (g << 4) | g, (b << 4) | b
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception as e:  # pragma: no cover
            print(f"[OmniSVG] token_to_color error: {e}")
            return "#808080"

    # ------------------------------------------------------------------ #
    # token ids -> xy commands
    # ------------------------------------------------------------------ #
    def tokens_to_xy(self, ids: Sequence[int]) -> np.ndarray:
        """
        Port of SVGTokenizer.process_generated_tokens.

        The original received `[BOS] + generated + [EOS]` and dropped both ends;
        the wrapped EOS never contributed anything, so feeding the raw generated
        stream (minus trailing stop tokens) is the same thing.
        """
        toks = list(ids)
        while toks and toks[-1] in (self.EOS_TOKEN_ID, self.BOS_TOKEN_ID):
            toks.pop()

        sample_xys = []
        for pixel in toks:
            if self.CMD_TOKEN_START <= pixel < self.CMD_TOKEN_END:
                sample_xys.append((pixel - self.BASE_OFFSET, pixel - self.BASE_OFFSET))
            elif self.COORD_TOKEN_START <= pixel < self.COLOR_COORD_BOUNDARY:
                index = pixel - self.COORD_TOKEN_START
                if 0 <= index < len(self.coord_ids):
                    x, y = self.coord_ids[index]
                    sample_xys.append((x - self.BASE_OFFSET, y - self.BASE_OFFSET))
            elif (self.ARC_PARAM_START + 1 <= pixel <
                  self.ARC_PARAM_START + 1 + self.ARC_PARAM_RANGE):
                value = pixel - self.ARC_PARAM_START - 1
                sample_xys.append((value, value))
            elif self.COLOR_COORD_BOUNDARY <= pixel < self.ARC_PARAM_START:
                sample_xys.append((pixel - self.BASE_OFFSET, pixel - self.BASE_OFFSET))

        if sample_xys:
            return np.asarray(sample_xys, dtype=np.int64)
        return np.zeros((0, 2), dtype=np.int64)

    # ------------------------------------------------------------------ #
    # xy commands -> 14-wide command rows, grouped per colour
    # ------------------------------------------------------------------ #
    def xy_to_paths(self, pixels: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """Port of SVGTokenizer.raster_svg (same branch order, same ranges)."""
        if len(pixels) == 0:
            return [], []

        pixels = pixels - self.PIXEL_OFFSET
        path_tensors: List[np.ndarray] = []
        color_tensors: List[int] = []
        rows: List[np.ndarray] = []

        i = 0
        n = len(pixels)
        while i < n:
            pix = pixels[i]
            head = pix[0]

            if head == self.CMD_MOVE:
                if i + 2 >= n:
                    break
                row = np.zeros(14)
                row[0] = 0
                row[12:14] = pixels[i + 2]
                rows.append(row)
                i += 3
            elif head == self.CMD_LINE:
                if i + 1 >= n:
                    break
                row = np.zeros(14)
                row[0] = 1
                row[12:14] = pixels[i + 1]
                rows.append(row)
                i += 2
            elif head == self.CMD_CURVE:
                if i + 3 >= n:
                    break
                row = np.zeros(14)
                row[0] = 2
                row[8:10] = pixels[i + 1]
                row[10:12] = pixels[i + 2]
                row[12:14] = pixels[i + 3]
                rows.append(row)
                i += 4
            elif head == self.CMD_ARC:
                if i + 5 >= n:
                    break
                row = np.zeros(14)
                row[0] = 3
                row[1:3] = pixels[i + 1]
                row[3] = pixels[i + 2][0] + self.PIXEL_OFFSET
                row[4] = pixels[i + 3][0] + self.PIXEL_OFFSET
                row[5] = pixels[i + 4][0] + self.PIXEL_OFFSET
                row[12:14] = pixels[i + 5]
                rows.append(row)
                i += 6
            elif head == self.CMD_CLOSE:
                if i + 1 >= n:
                    break
                row = np.zeros(14)
                row[0] = 6
                row[12:14] = pixels[i + 1]
                rows.append(row)
                i += 2
            elif head >= self.COLOR_THRESHOLD:
                if rows:
                    path_tensors.append(np.stack(rows))
                    color_tensors.append(int(pix[0] + self.PIXEL_OFFSET - 1))
                    rows = []
                i += 1
            else:
                i += 1

        if rows:
            path_tensors.append(np.stack(rows))
        return path_tensors, color_tensors

    # ------------------------------------------------------------------ #
    # command rows -> SVG
    # ------------------------------------------------------------------ #
    def _path_d(self, rows: np.ndarray) -> str:
        """
        Emit one `d` string for a command block, mirroring DeepSVG's
        SVGPath.from_commands + SVGPath.to_str:
          - a Move opens a subpath, a Close closes it (DeepSVG additionally put a
            Line back to the Close's end point right before the `Z`; that end
            point is the subpath origin, so it is preserved here);
          - everything after a Close is ignored until the next Move.
        """
        parts: List[str] = []
        open_subpath = False
        for row in rows:
            cmd = _CMD_LETTERS.get(int(row[0]))
            if cmd is None:
                continue
            if cmd == "M":
                parts.append(f"M {_num(row[12])} {_num(row[13])}")
                open_subpath = True
            elif not open_subpath:
                continue
            elif cmd == "L":
                parts.append(f"L {_num(row[12])} {_num(row[13])}")
            elif cmd == "C":
                parts.append("C " + " ".join(_num(v) for v in
                                             (row[8], row[9], row[10], row[11], row[12], row[13])))
            elif cmd == "A":
                # the arc scalars arrive as ARC_PARAM tokens and already carry
                # the +PIXEL_OFFSET shift applied in xy_to_paths, which puts
                # x-axis-rotation in degrees and both flags back on 0/1
                parts.append("A " + " ".join(
                    [_num(row[1]), _num(row[2]), _num(row[3]),
                     str(int(row[4])), str(int(row[5])),
                     _num(row[12]), _num(row[13])]))
            elif cmd == "Z":
                if row[12] != 0 or row[13] != 0:
                    parts.append(f"L {_num(row[12])} {_num(row[13])}")
                parts.append("Z")
                open_subpath = False
        return " ".join(parts)

    def paths_to_svg(self, path_tensors: List[np.ndarray], colors: Optional[List[int]] = None) -> Optional[str]:
        colors = colors or []
        bbox = self.BBOX
        body = []
        for i, rows in enumerate(path_tensors):
            d = self._path_d(rows)
            if not d:
                # nothing but Move-less commands: DeepSVG dropped these too
                continue
            color = self.token_to_color(colors[i]) if i < len(colors) else "none"
            body.append(f'<path fill="{color}" fill-opacity="1.0" d="{d}"></path>')
        if not body:
            return None
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {bbox} {bbox}" '
                f'width="{bbox}" height="{bbox}">' + "\n".join(body) + "</svg>")

    def detokenise(self, ids: Sequence[int]):
        """
        Token ids -> {"svg", "subpaths", "commands", "span_x", "span_y"}.
        Returns None when the stream carries no usable geometry.

        The span numbers are renderer-free: they come straight from the command
        rows, so the node can tell a canvas-filling drawing from a speck without
        rasterising anything.
        """
        try:
            xy = self.tokens_to_xy(ids)
            if len(xy) == 0:
                return None
            path_tensors, colors = self.xy_to_paths(xy)
            if not path_tensors:
                return None
            # paths that were flushed without a colour token default to black,
            # exactly like the original generate_candidates()
            black = self.config["colors"].get("black_color_token", self.COLOR_TOKEN_START_RAW + 2)
            while len(colors) < len(path_tensors):
                colors.append(black)
            svg = self.paths_to_svg(path_tensors, colors)
            if svg is None:
                return None

            n_sub = n_cmd = 0
            pts = []
            for rows in path_tensors:
                n_sub += int((rows[:, 0] == 0).sum())
                n_cmd += len(rows)
                pts.append(rows[:, 12:14])                      # M/L/C/A end points
                cubic = rows[rows[:, 0] == 2]
                if len(cubic):
                    pts.append(cubic[:, 8:12].reshape(-1, 2))   # cubic control points
            pts = np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))
            if len(pts) == 0:
                return None
            span_x = float(pts[:, 0].max() - pts[:, 0].min()) / self.BBOX
            span_y = float(pts[:, 1].max() - pts[:, 1].min()) / self.BBOX
            return {"svg": svg, "subpaths": n_sub, "commands": n_cmd,
                    "span_x": span_x, "span_y": span_y}
        except Exception as e:
            print(f"[OmniSVG] detokenise error: {type(e).__name__}: {e}")
            return None

    def ids_to_svg(self, ids: Sequence[int]) -> Optional[str]:
        """Token ids -> SVG source string (None when the stream has no usable paths)."""
        out = self.detokenise(ids)
        return out["svg"] if out else None


# ---------------------------------------------------------------------- #
# rendering (pillow-lunasvg)
# ---------------------------------------------------------------------- #
HAS_LUNASVG = _LUNASVG_OK


def render_svg(svg_str: str, size: int = 1024) -> Optional[Image.Image]:
    """Rasterise an SVG string into an RGB PIL image on a white background.

    Returns None when pillow-lunasvg is missing or the document cannot be
    parsed; callers treat that as "no raster available" and fall back to the
    geometry based sanity check.
    """
    if not _LUNASVG_OK:
        return None
    try:
        img = Image.open(io.BytesIO(svg_str.encode("utf-8")))
        scale = float(size) / float(img.size[0])
        img.load(scale=scale)
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    except Exception as e:
        print(f"[OmniSVG] SVG render error: {type(e).__name__}: {e}")
        return None


def svg_mean_brightness(img: Image.Image) -> float:
    return float(np.asarray(img, dtype=np.float32).mean())


def geometry_ok(metrics: dict) -> Tuple[bool, str]:
    """
    Renderer-free blank-page gate, used only when no raster backend is present:
    at least one subpath and geometry that actually spans the canvas.
    """
    if metrics["subpaths"] < 1 or metrics["commands"] < 3:
        return False, "no_geometry"
    if metrics["span_x"] < 0.15 and metrics["span_y"] < 0.15:
        return False, "too_small"
    return True, "ok"


def pil_to_tensor(img: Image.Image) -> "torch.Tensor":
    import torch
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "OmniSVG", "config.yaml")
