import os
import cv2
import torch
import argparse
import numpy as np
import av

from tqdm import tqdm
from torchvision.transforms import functional as TF
from model import LitRT4KSR_Rep
from utils import reparameterize, tensor2uint
import config


model_path = config.checkpoint_path_video_infer


def get_available_devices():
  devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
  if not devices:
    devices = [torch.device("cpu")]
  print(f"Using {len(devices)} GPU(s): {[str(d) for d in devices]}")
  return devices


def load_model(device):
  litmodel = LitRT4KSR_Rep.load_from_checkpoint(
    checkpoint_path=model_path, config=config, map_location=device
  )
  if config.video_infer_reparameterize:
    litmodel.model = reparameterize(config, litmodel.model, device, save_rep_checkpoint=False)
  litmodel.model.to(device)
  litmodel.model.eval()
  return litmodel


def process_video(input_path, output_path, litmodels, devices, batch_size):
  container = av.open(input_path)
  video_stream = next(s for s in container.streams if s.type == "video")
  input_audio_streams = [s for s in container.streams if s.type == "audio"]

  output = av.open(output_path, mode="w")
  out_video_stream = output.add_stream("libx264", rate=video_stream.average_rate)
  out_video_stream.width = video_stream.width * config.scale
  out_video_stream.height = video_stream.height * config.scale
  out_video_stream.pix_fmt = "yuv420p"

  # Map input audio stream index to output stream using codec name
  output_audio_streams = {}
  for input_stream in input_audio_streams:
    codec_name = input_stream.codec.name or "aac"
    output_stream = output.add_stream(codec_name, rate=input_stream.rate)
    output_audio_streams[input_stream.index] = output_stream

  out_audio_stream = output.add_stream_from_template(container.streams.audio[0])

  buffer = []
  current_batch_size = batch_size
  min_batch_size = 4
  frame_count = video_stream.frames
  processed_frames = 0
  progress_bar = tqdm(total=frame_count)

  # Copy audio packets from original file into output
  for packet in container.demux(input_audio_streams):
    if packet.dts is None:
      continue
    output.mux(packet)

  container.seek(0)

  for frame in container.decode(video=0):
    frame_rgb = frame.to_rgb().to_ndarray()
    tensor = TF.to_tensor(frame_rgb / 255.0).unsqueeze(0).float()
    buffer.append(tensor)

    if len(buffer) >= current_batch_size:
      while True:
        try:
          with torch.no_grad():
            frames_tensor = torch.cat(buffer)

            chunks = torch.chunk(frames_tensor, len(devices))
            sr_chunks = []

            for model, chunk, gpu in zip(litmodels, chunks, devices):
              chunk = chunk.to(gpu)
              sr_out = model.model(chunk)
              sr_chunks.append(sr_out.cpu())

            sr_frames = torch.cat(sr_chunks)

            for sr_frame in sr_frames:
              sr_img = tensor2uint(sr_frame * 255.0)
              video_frame = av.VideoFrame.from_ndarray(sr_img, format="rgb24")
              for packet in out_video_stream.encode(video_frame):
                output.mux(packet)

          processed_frames += len(buffer)
          progress_bar.update(len(buffer))
          buffer = []
          break

        except RuntimeError as e:
          if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            print(f"[OOM] Reducing batch size from {current_batch_size}")
            current_batch_size = max(current_batch_size // 2, min_batch_size)
            buffer = buffer[:current_batch_size]
            continue
          else:
            raise e

  if buffer:
    with torch.no_grad():
      frames_tensor = torch.cat(buffer)
      chunks = torch.chunk(frames_tensor, len(devices))
      sr_chunks = []

      for model, chunk, gpu in zip(litmodels, chunks, devices):
        chunk = chunk.to(gpu)
        sr_out = model.model(chunk)
        sr_chunks.append(sr_out.cpu())

      sr_frames = torch.cat(sr_chunks)
      for sr_frame in sr_frames:
        sr_img = tensor2uint(sr_frame * 255.0)
        video_frame = av.VideoFrame.from_ndarray(sr_img, format="rgb24")
        for packet in out_video_stream.encode(video_frame):
          output.mux(packet)

  for packet in out_video_stream.encode():
    output.mux(packet)

  progress_bar.close()
  output.close()
  container.close()


def main():
  parser = argparse.ArgumentParser(description="Process a video with batch size.")
  parser.add_argument("--batch-size", type=int, default=64, help="The size of the batch")
  parser.add_argument("--video-path", type=str, required=True, help="Path to the input video file")
  parser.add_argument(
    "--output-path", type=str, required=True, help="Path to save the output video file"
  )
  args = parser.parse_args()

  devices = get_available_devices()
  litmodels = [load_model(device) for device in devices]

  process_video(args.video_path, args.output_path, litmodels, devices, args.batch_size)


if __name__ == "__main__":
  main()
