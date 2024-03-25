import ultralytics
from ultralytics.utils.plotting import save_one_box
import cv2
import numpy as np
import pafy
import concurrent.futures
from collections import defaultdict
import types
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F

from counter import Counter
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend

# get input argument
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', action='store_true', help='use webcam')               # webcam usually is 0
parser.add_argument('--camera', type=int, default=None, help='camera port number')    # you can find it using find_port.py
parser.add_argument('--video-file', type=str, default=None, help='video filenames')   # example: "dataset_cam1.mp4"
parser.add_argument('--rtsp', type=str, default=None, help='rtsp link')               # example: "rtsp://192.168.1.136:8554/"
parser.add_argument('--youtube', type=str, default=None, help='youtube link')         # example: "http://www.youtube.com/watch?v=q0kPBRIPm6o"
parser.add_argument('--roi-xyxy', type=str, default=None, help='x1y1x2y2 of geofencing region of interest (in range 0 to 1), i.e.: [0.3,0.5,0.3,0.5]')
parser.add_argument('--stream-idx', type=int, default=0, help='Index for this video streaming')
opt = parser.parse_args()

# Define the source
WEBCAM = opt.webcam
CAMERA = opt.camera
VIDEO_FILE = opt.video_file
RTSP = opt.rtsp
YOUTUBE = opt.youtube # need ssl to be set

# Other arguments
ROI_XYXY   = opt.roi_xyxy
STREAM_IDX = opt.stream_idx
SAVE       = False


# load video source
if WEBCAM:
   cap = cv2.VideoCapture(0) # usually webcam is 0
elif CAMERA is not None: 
   cap = cv2.VideoCapture(CAMERA)
elif VIDEO_FILE:
   cap = cv2.VideoCapture(VIDEO_FILE)
elif RTSP:
   cap = cv2.VideoCapture(RTSP)
elif YOUTUBE:
   video = pafy.new(YOUTUBE)
   best = video.getbest(preftype="mp4")
   cap = cv2.VideoCapture(best.url)   
else:
   assert False, "You do not specificy input video source!"


# resize your input video frame size (smaller -> faster, but less accurate)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
resize_width = 1280   # Adjust based on your needs
resize_height = 720  # Adjust based on your needs
if frame_width > 0:
   resize_height = int((resize_width / frame_width) * frame_height)


# save crop, to overwrite in ultralytics.engine.results.save_crop
def save_crop(self, save_dir, file_name=Path("im.jpg")):
    """
    Save cropped predictions to `save_dir/cls/file_name.jpg`.

    Args:
        save_dir (str | pathlib.Path): Save path.
        file_name (str | pathlib.Path): File name.
    """
    if self.probs is not None:
        LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
        return
    if self.obb is not None:
        LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
        return

    # variables
    boxes = self.boxes
    try:
        track_ids = self.boxes.id.int().cpu().tolist()
    except:
        track_ids = [None] * len(boxes)

    # save crops
    for d, id in zip(boxes, track_ids):
        if id is None:
            continue
        crop = save_one_box(
                            d.xyxy,
                            self.orig_img.copy(),
                            file=Path(save_dir) / f"stream_{STREAM_IDX}" / str(id) / f"{Path(file_name)}.jpg",
                            BGR=True,
                           )     
        # ReID
        crop = torch.from_numpy(crop)
        crop = crop.unsqueeze(0)
        crop = crop.permute(0, 3, 1, 2)


# get model
def get_model(opt):
    # replace save crop function
    ultralytics.engine.results.Results.save_crop = save_crop 
    
    # Load the YOLO model
    chosen_model = ultralytics.YOLO("yolov8n_face.pt")  # Adjust model version as needed

    # Load the ReID model
    chosen_model.reid = ReIDDetectMultiBackend(weights=Path("backbone_10000.onnx"), device=torch.device(0), fp16=True)

    # ReID dataset
    chosen_model.database = None

    # Load counter for geofencing based on ROI
    ROI_XYXY   = opt.roi_xyxy
    STREAM_IDX = opt.stream_idx
    if ROI_XYXY is not None:
       xyxy = ROI_XYXY.split(',')
       assert len(xyxy) == 4, 'xyxy should be 4 coordinates'
       xyxy = [float(item) for item in xyxy]
       x1, y1, x2, y2  = xyxy
       chosen_model.my_counter = Counter(x1, y1, x2, y2, STREAM_IDX)
    else:
       chosen_model.my_counter = None
       
    return chosen_model
    
    
# draw roi
def draw_roi(chosen_model, img):
   # img shape
   img_shape = img.shape
   
   # draw roi
   x1 = chosen_model.my_counter.roi_x1 * img_shape[1]
   y1 = chosen_model.my_counter.roi_y1 * img_shape[0]
   x2 = chosen_model.my_counter.roi_x2 * img_shape[1]
   y2 = chosen_model.my_counter.roi_y2 * img_shape[0]

   pts = [[x1,y1],[x1,y2],[x2,y2],[x2,y1]]
   pts = np.array(pts, int)
   pts = pts.reshape((-1, 1, 2))
   img = cv2.polylines(img, [pts], True, (0,0,255), 5)

   # put text
   text = f'in: {chosen_model.my_counter.count_in}'
   font = cv2.FONT_HERSHEY_SIMPLEX
   font_scale = int(img.shape[0] * 0.002)
   font_thickness = 2
   origin = (int(img.shape[0]*0.35), int(img.shape[1]*0.5))
   x, y = origin
   text_color = (255, 255, 255)
   text_color_bg = (0, 0, 0)
        
   text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
   text_w, text_h = text_size

   cv2.rectangle(img, origin, (x + text_w, y + text_h), text_color_bg, -1)
   cv2.putText(img,text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
   
   return img


# reid
def reid(xyxys, img):
   # ReID model inference
   feats = chosen_model.reid.get_features(xyxys.numpy(), img)

   # normalize feats
   feats = torch.from_numpy(feats)
   feats = F.normalize(feats, dim=1)

   # init database
   if chosen_model.database is None:
      chosen_model.database = feats
      return

   # pair-wise scores
   #scores = torch.sum(chosen_model.database * feats, dim=1).tolist()
   cosine_sim = torch.mm(chosen_model.database, feats.transpose(0, 1))
   print(cosine_sim)


# predict
def predict(chosen_model, img, classes=[], conf=0.5):
   #resiz the image to 640x480
   img = cv2.resize(img, (resize_width, resize_height))
   if classes:
       results = chosen_model.track(img, classes=classes, conf=conf, save_txt=SAVE, persist=True, verbose=False, save_crop=SAVE)
   else:
       results = chosen_model.track(img, conf=conf, save_txt=SAVE, persist=True, verbose=False, save_crop=SAVE)

   return results


# predict and detect
def predict_and_detect(chosen_model, track_history, img, classes=[], conf=0.5):
   # resiz the image to 640x480
   img = cv2.resize(img, (resize_width, resize_height))
   img_shape = img.shape

   # get results   
   results = predict(chosen_model, img, classes, conf=conf)

   # Get the boxes and track IDs
   boxes = results[0].boxes.xywh.cpu()
   xyxys = results[0].boxes.xyxy.cpu()
   img = results[0].orig_img.copy()
   try:
      track_ids = results[0].boxes.id.int().cpu().tolist()
   except:
      # draw roi
      if chosen_model.my_counter is not None:  
         img = draw_roi(chosen_model, img)

      # log   
      return img, results

   # reid
   reid(xyxys, img)

   # visualize
   annotated_frame = results[0].plot()
   for box, track_id in zip(boxes, track_ids):
      x, y, w, h = box
      track = track_history[track_id]
      track.append((float(x), float(y)))  # x, y center point
      if len(track) > 30:  # retain 90 tracks for 90 frames
         track.pop(0)

      # Draw the tracking lines
      # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
      # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

   if chosen_model.my_counter is not None:  
      # counter
      chosen_model.my_counter.update(img_shape, results[0])
   
      # draw roi
      annotated_frame = draw_roi(chosen_model, annotated_frame)
   
      # log
      chosen_model.my_counter.log()
   
   return annotated_frame, results


# process frame
def process_frame(track_history, frame):
   result_frame, _ = predict_and_detect(chosen_model, track_history, frame)
   return result_frame


# main
def main():
   skip_frames = 2  # Number of frames to skip before processing the next one
   frame_count = 0  

   # Store the track history
   track_history = defaultdict(lambda: [])

   with concurrent.futures.ThreadPoolExecutor() as executor:
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           frame_count = 1+frame_count
           if frame_count % skip_frames != 0:
               continue  # Skip this frame

           # Submit the frame for processing
           future = executor.submit(process_frame, track_history, frame)
           result_frame = future.result()

           # Display the processed frame
           cv2.imshow(f"Video Stream {STREAM_IDX}", result_frame)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break

   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   chosen_model = get_model(opt)
   main()
