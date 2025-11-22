#!/usr/bin/env python
"""
Quick example to generate SVG with the 1B/8B model, mirroring the README snippet.

Usage:
  # 1B, sample image 0
  uv run python run_example.py --size 1b --image assets/examples/sample-0.png --max-length 4000 --device auto

  # 8B, sample image 18
  uv run python run_example.py --size 8b --image assets/examples/sample-18.png --max-length 4000 --device auto

  # Specific model id or local path
  uv run python run_example.py --model starvector/starvector-1b-im2svg --image assets/examples/sample-0.png

  # Custom output directory
  uv run python run_example.py --size 1b --out-dir outputs_run
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from starvector.data.util import process_and_rasterize_svg


def parse_args():
    parser = argparse.ArgumentParser(description="Run StarVector Image2SVG example.")
    parser.add_argument(
        "--size",
        choices=["1b", "8b"],
        default="1b",
        help="Model size shorthand (overridden if --model is set).",
    )
    parser.add_argument("--model", default=None, help="HF model id or local path.")
    parser.add_argument("--image", default="assets/examples/sample-0.png", help="Input image path.")
    parser.add_argument("--max-length", type=int, default=4000, help="Max tokens for generation.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to run on.")
    parser.add_argument("--out-dir", default="outputs", help="Directory to save SVG/PNG outputs.")
    return parser.parse_args()


def main():
    args = parse_args()

    model_id = args.model if args.model is not None else f"starvector/starvector-{args.size}-im2svg"
    tag = args.size if args.model is None else Path(model_id).name

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[load] model={model_id} device={device} dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    print("[model] loaded and moved to device")

    processor = model.model.processor
    print(f"[processor] using processor from model ({processor.__class__.__name__})")

    image_path = Path(args.image)
    print(f"[image] loading {image_path}")
    image_pil = Image.open(image_path)
    image = model.process_images([image_pil])[0].to(device)
    print(f"[image] processed via model.process_images -> tensor shape {tuple(image.shape)} dtype {image.dtype} device {image.device}")
    batch = {"image": image}

    print(f"[generate] max_length={args.max_length}")
    with torch.no_grad():
        raw_svg = model.generate_im2svg(batch, max_length=args.max_length)[0]
    print(f"[generate] SVG length (chars): {len(raw_svg)}")

    svg, raster_image = process_and_rasterize_svg(raw_svg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"example-{tag}.svg"
    png_path = out_dir / f"example-{tag}.png"

    svg_path.write_text(svg, encoding="utf-8")
    raster_image.save(png_path)

    print(f"[done] svg -> {svg_path}")
    print(f"[done] raster preview -> {png_path}")
    print(f"[svg head]\n{svg[:400]}...\n")


if __name__ == "__main__":
    main()
