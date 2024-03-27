import ultralytics
from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import cv2
import numpy as np
import pafy
from copy import deepcopy
import concurrent.futures
from collections import defaultdict
import types
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F

from counter import Counter
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend

#------------------------------------------------------------------------------------------------------
# Arguments
#------------------------------------------------------------------------------------------------------
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
THRESH     = 0.5

#------------------------------------------------------------------------------------------------------
# Video streaming
#------------------------------------------------------------------------------------------------------
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



#------------------------------------------------------------------------------------------------------
# Overwrite original ultralytics function
#------------------------------------------------------------------------------------------------------
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


# plot, to overwrite in ultralytics.engine.results.plot
def plot(
    self,
    reid_dict=None,
    conf=True,
    line_width=None,
    font_size=None,
    font="Arial.ttf",
    pil=False,
    img=None,
    im_gpu=None,
    kpt_radius=5,
    kpt_line=True,
    labels=True,
    boxes=True,
    masks=True,
    probs=True,
    show=False,
    save=False,
    filename=None,
):

    if img is None and isinstance(self.orig_img, torch.Tensor):
        img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

    names = self.names
    is_obb = self.obb is not None
    pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
    pred_masks, show_masks = self.masks, masks
    pred_probs, show_probs = self.probs, probs
    annotator = Annotator(
        deepcopy(self.orig_img if img is None else img),
        line_width,
        font_size,
        font,
        pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names,
    )

    # Plot Segment results
    if pred_masks and show_masks:
        if im_gpu is None:
            img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
            im_gpu = (
                torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                .permute(2, 0, 1)
                .flip(0)
                .contiguous()
                / 255
            )
        idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
        annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

    # Plot Detect results
    if pred_boxes is not None and show_boxes:
        for d in reversed(pred_boxes):
            c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            if reid_dict is not None:
                id = reid_dict[id]
            name = ("" if id is None else f"id:{id} ") + names[c]
            label = (f"{name} {conf:.2f}" if conf else name) if labels else None
            box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
            annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)

    # Plot Classify results
    if pred_probs is not None and show_probs:
        text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
        x = round(self.orig_shape[0] * 0.03)
        annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

    # Plot Pose results
    if self.keypoints is not None:
        for k in reversed(self.keypoints.data):
            annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

    # Show results
    if show:
        annotator.show(self.path)

    # Save results
    if save:
        annotator.save(filename)

    return annotator.result()

#------------------------------------------------------------------------------------------------------
# ReID function
#------------------------------------------------------------------------------------------------------
# reid
class ReIDManager:
    def __init__(self):
        self.feat_database = None
        self.total_unique_id = 0
        self.unique_track_ids = [-1] * 300
        self.track_to_match_id = {}
        
    def matching(self, track_ids, xyxys, img):
        # ReID model inference
        feats = chosen_model.reid.get_features(xyxys.numpy(), img)

        # normalize feats
        feats = torch.from_numpy(feats)
        feats = F.normalize(feats, dim=1)

        # init
        if self.feat_database is None:
            self.feat_database = feats
            for track_id in track_ids:
                self.total_unique_id += 1
                self.unique_track_ids.append(track_id)
                self.track_to_match_id[track_id] = self.total_unique_id
            return

        # cosine similarity scores, and matches
        cosine_sim = torch.mm(self.feat_database, feats.transpose(0, 1))
        match_ids        = torch.argmax(cosine_sim, dim=0).cpu().tolist()
        match_thresholds = torch.any(cosine_sim > THRESH, dim=0).cpu().tolist()
        
        #print(match_ids, match_thresholds)
        reid_dict = {}
        for idx, (track_id, match_id, match_threshold) in enumerate(zip(track_ids, match_ids, match_thresholds)):
            #print(idx, match_id, match_threshold)
            
            # skip this track_id, if it has been recorded
            if track_id in self.unique_track_ids:
                reid_dict[track_id] = self.track_to_match_id[track_id]
                continue
            
            # register track_id
            self.unique_track_ids.append(track_id)
            self.unique_track_ids.pop(0)            
            
            # if match, then use the match_id
            if match_threshold:
                reid_dict[track_id] = match_id + 1
                self.track_to_match_id[track_id] = match_id + 1
            else:
                self.total_unique_id += 1
                reid_dict[track_id] = self.total_unique_id
                self.feat_database = torch.cat((self.feat_database, feats[idx].unsqueeze(0)), axis=0)
                self.track_to_match_id[track_id] = self.total_unique_id
                #print(self.feat_database.shape)
        return reid_dict
            


#------------------------------------------------------------------------------------------------------
# Functions for video streaming
#------------------------------------------------------------------------------------------------------
# get model
def get_model(opt):
    # overwrite function
    ultralytics.engine.results.Results.save_crop = save_crop 
    ultralytics.engine.results.Results.plot      = plot
    
    # Load the YOLO model
    chosen_model = ultralytics.YOLO("yolov8n_face.pt")  # Adjust model version as needed

    # Load the ReID model
    chosen_model.reid = ReIDDetectMultiBackend(weights=Path("custom_backbone_10000.onnx"), device=torch.device(0), fp16=True)

    # ReID magager
    chosen_model.reid_manager = ReIDManager()

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
    reid_dict = chosen_model.reid_manager.matching(track_ids, xyxys, img)

    # visualize
    annotated_frame = results[0].plot(reid_dict)
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
