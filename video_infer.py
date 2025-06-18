import os

import config
import cv2
import torch
from model import LitRT4KSR_Rep
from torchvision.transforms import functional as TF
from tqdm import tqdm

from utils import reparameterize, tensor2uint

model_path = config.checkpoint_path_video_infer
save_path = config.video_infer_save_path


def get_device():
  device = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device == "auto"
    else torch.device(config.device)
  )
  print(f"Using device: {device}")
  return device


def load_model(device):
  litmodel = LitRT4KSR_Rep.load_from_checkpoint(
    checkpoint_path=model_path, config=config, map_location=device
  )
  if config.video_infer_reparameterize:
    litmodel.model = reparameterize(config, litmodel.model, device, save_rep_checkpoint=False)
  litmodel.model.to(device)
  litmodel.eval()
  return litmodel


def read_video(video_path):
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  if not cap.isOpened():
    raise ValueError(f"Error opening video file: {video_path}")

  print("Video info:")
  print(f"fps: {fps}")
  print(f"width: {width}")
  print(f"height: {height}")
  print(f"frame_count: {frame_count}")

  return cap, fps, width, height, frame_count


def setup_output(video_path, save_path, fps, width, height):
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  video_name = os.path.basename(video_path).replace(config.video_format, ".mkv")
  video_sr_path = os.path.join(save_path, video_name)
  fourcc = cv2.VideoWriter_fourcc(*"XVID")

  return cv2.VideoWriter(video_sr_path, fourcc, fps, (width * config.scale, height * config.scale))


def process_frames(cap, litmodel, device, video_sr, frame_count, batch_size):
  min_batch_size = 4
  current_batch_size = batch_size
  processed_frames = 0
  progress_bar = tqdm(total=frame_count)

  while processed_frames < frame_count:
    frame_buffer = []

    # Try to load up to current_batch_size frames
    for _ in range(current_batch_size):
      if processed_frames + len(frame_buffer) >= frame_count:
        break  # prevent reading beyond total
      ret, frame = cap.read()
      if not ret:
        break
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      tensor = TF.to_tensor(frame / 255.0).unsqueeze(0).float()
      frame_buffer.append(tensor)

    if not frame_buffer:
      break  # nothing left to process

    # Retry processing the current buffer if OOM occurs
    while True:
      try:
        with torch.no_grad():
          frames_tensor = torch.cat(frame_buffer).to(device)
          sr_frames = litmodel.predict_step(frames_tensor)

          for sr_frame in sr_frames:
            sr_frame = tensor2uint(sr_frame * 255.0)
            sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
            video_sr.write(sr_frame)

        processed_frames += len(frame_buffer)
        progress_bar.update(len(frame_buffer))  # ✅ only update here
        break  # done with this batch

      except RuntimeError as e:
        if "CUDA out of memory" in str(e):
          torch.cuda.empty_cache()
          print(f"[OOM] Reducing batch size from {current_batch_size}")
          current_batch_size = max(current_batch_size - 2, min_batch_size)

          if len(frame_buffer) > current_batch_size:
            frame_buffer = frame_buffer[:current_batch_size]
          continue
        else:
          raise e

  progress_bar.close()
  if current_batch_size != batch_size:
    print(f"Finished processing with reduced batch size: {current_batch_size}")


import argparse


def main():
  parser = argparse.ArgumentParser(description="Process a video with batch size.")
  parser.add_argument("--batch-size", type=int, default=64, help="The size of the batch")
  parser.add_argument("--video-path", type=str, required=True, help="Path to the input video file")
  parser.add_argument(
    "--output-path", type=str, required=True, help="Path to save the output video file"
  )
  args = parser.parse_args()

  device = get_device()
  litmodel = load_model(device)

  cap, fps, width, height, frame_count = read_video(args.video_path)
  video_sr = setup_output(args.video_path, args.output_path, fps, width, height)

  process_frames(cap, litmodel, device, video_sr, frame_count, batch_size=args.batch_size)


if __name__ == "__main__":
  main()
