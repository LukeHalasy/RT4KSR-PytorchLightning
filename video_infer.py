import os
import cv2
import torch
from model import LitRT4KSR_Rep
from torchvision.transforms import functional as TF
from tqdm import tqdm

from utils import reparameterize, tensor2uint
import config


model_path = config.checkpoint_path_video_infer
save_path = config.video_infer_save_path
video_path = config.video_infer_video_path


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

  video_name = os.path.basename(video_path).replace(config.video_format, "_SR.avi")
  video_sr_path = os.path.join(save_path, video_name)
  fourcc = (
    cv2.VideoWriter_fourcc(*"VP80")
    if config.video_format == ".webm"
    else cv2.VideoWriter_fourcc(*"XVID")
  )

  return cv2.VideoWriter(video_sr_path, fourcc, fps, (width * config.scale, height * config.scale))


def process_frames(cap, litmodel, device, video_sr, frame_count, batch_size):
  frame_buffer = []
  for i in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
      break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_buffer.append(TF.to_tensor(frame / 255.0).unsqueeze(0).float())

    if len(frame_buffer) == batch_size or i == frame_count - 1:
      with torch.no_grad():
        frames_tensor = torch.cat(frame_buffer).to(device)
        sr_frames = litmodel.predict_step(frames_tensor)
        for sr_frame in sr_frames:
          sr_frame = tensor2uint(sr_frame * 255.0)
          sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
          video_sr.write(sr_frame)
      frame_buffer.clear()


def main():
  device = get_device()
  litmodel = load_model(device)

  cap, fps, width, height, frame_count = read_video(video_path)
  video_sr = setup_output(video_path, save_path, fps, width, height)

  process_frames(cap, litmodel, device, video_sr, frame_count, batch_size=64)


if __name__ == "__main__":
  main()
