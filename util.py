import os
import cv2
import numpy as np

def make_savepath(save_dirs,classes):
    for path in save_dirs.values():
        if os.path.isdir(path):
            continue
        os.makedirs(path, exist_ok=True)
        class_txt_path=f"{path}/classes.txt"
        if not os.path.isfile(class_txt_path):
            with open(class_txt_path, "w") as f:
                for idx in sorted(classes.keys()):
                    f.write(classes[idx] + "\n")

def plot_label_image(img, annots,class_names):
    colors = ([0,100,0],[0,0,255],[0,215,255],[0,128,128],[0,0,139],[30,105,210],[66,10,255],[30,0,255])
    for anno in annots:
        if len(anno) == 0:
            continue  # 결과가 비어 있으면 건너뜁니다.
        if len(anno) < 4:
            print("Invalid annotation:", anno)
            continue  # 결과가 예상한 형식이 아니면 건너뜁니다.
        label = f"{class_names[int(anno[-1])]}: {anno[-2]:.2f}"
        cv2.rectangle(img, (int(anno[0]),int(anno[1])), (int(anno[2]),int(anno[3])), color=colors[int(anno[-1])], thickness=1)
            #cv2.putText(img, str(anno[4]), (anno[0]+5,anno[1]+18), 1, 1, color=colors[int(anno[5])], thickness=2)
        cv2.putText(img, label, (int(anno[0])+5,int(anno[1])+18), 1, 1, color=colors[int(anno[-1])], thickness=2)

def new_letterbox(old_img, new_shape=320):
    img = np.array(old_img)
    h,w = img.shape[:2]
    if h>new_shape or w>new_shape:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    aspect = w/h

    if aspect > 1:
        new_w = new_shape
        new_h = np.round(new_w / aspect).astype(int)

        pad_vert = (new_shape - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = new_shape
        new_w = np.round(new_h * aspect).astype(int)

        pad_horz = (new_shape - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = new_shape, new_shape
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=(127,127,127))

    return scaled_img, (pad_top, pad_bot, pad_left, pad_right), (new_w, new_h)

# def postprocess_yolo(results, shape, ratio):
#     result = []
#     num_dets = results.as_numpy("num_dets")[0][0]
#     bboxes = results.as_numpy("det_boxes")[0]
#     scores = results.as_numpy("det_scores")[0]
#     classes = results.as_numpy("det_classes")[0]
#     cnt = 0
#     f_count=0
#     ret = []
    
#     if num_dets == 0:
#         return ret,0

#     for bbox, score, clss in zip(bboxes, scores, classes):
#         cnt += 1
#         #print(clss, infer_class)
#         if cnt > num_dets:break
#         x1 = np.clip((bbox[0]-ratio[2][0])*ratio[0],0,shape[1])
#         y1 = np.clip((bbox[1]-ratio[2][1])*ratio[1],0,shape[0])
#         x2 = np.clip((bbox[2]-ratio[2][0])*ratio[0],0,shape[1])
#         y2 = np.clip((bbox[3]-ratio[2][1])*ratio[1],0,shape[0])
#         if int(clss)==2:
#             f_count+=1
#         ret.append([round(x1),round(y1),round(x2),round(y2),score,int(clss)])
       
#     return ret,f_count

def postprocess_yolo(results, shape, ratio, conf_score=0.3):
    num_dets = results.as_numpy("num_dets")[0][0]
    bboxes = results.as_numpy("det_boxes")[0]
    scores = results.as_numpy("det_scores")[0]
    classes = results.as_numpy("det_classes")[0]

    if num_dets == 0:
        return [], []

    # confidence 필터링
    valid_indices = scores > conf_score
    valid_bboxes = bboxes[valid_indices]
    valid_scores = scores[valid_indices]
    valid_classes = classes[valid_indices]

    # 좌표 변환을 벡터화
    x1 = np.clip((valid_bboxes[:, 0] - ratio[2][0]) * ratio[0], 0, shape[1])
    y1 = np.clip((valid_bboxes[:, 1] - ratio[2][1]) * ratio[1], 0, shape[0])
    x2 = np.clip((valid_bboxes[:, 2] - ratio[2][0]) * ratio[0], 0, shape[1])
    y2 = np.clip((valid_bboxes[:, 3] - ratio[2][1]) * ratio[1], 0, shape[0])
    
    # 모든 결과를 하나의 배열로 생성
    all_results = np.column_stack([
        np.round(x1), np.round(y1), np.round(x2), np.round(y2), 
        valid_scores, valid_classes.astype(int)
    ])
    
    # 클래스별로 분리

    pose_mask = np.isin(valid_classes, [0, 1])
    
    pose_result = all_results[pose_mask] if np.any(pose_mask) else []
    farrow_result = all_results[~pose_mask] if np.any(~pose_mask) else []

    return pose_result, farrow_result

def preprocess_yolo(img, half=False, h=320, w=320):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """

    new_img = img.copy()
    if img.shape[2] == 3:
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    one_shape = new_img.shape[:2]

    # make resized letterbox
    resized, padding, rat = new_letterbox(new_img,h)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    typed = resized.astype(np.float32)

    scaled = typed / 255.0

    # Swap to CHW
    ordered = np.transpose(scaled, (2, 0, 1))
    
    if padding[0]!=0 and padding[2]==0: # horizontal image
        ratio = (img.shape[1]/w, img.shape[0]/rat[1], (0, padding[0]))
    elif padding[0]==0 and padding[2]!=0: # portrait image
        ratio = (img.shape[1]/rat[0], img.shape[0]/h, (padding[2], 0))
    elif padding[0]==0 and padding[2]==0: # square image
        ratio = (img.shape[1]/w, img.shape[0]/h, (0,0))

    if half:
        ordered = ordered.astype(np.float16)

    return (np.expand_dims(ordered, axis=0), one_shape, ratio) 

def convert_to_yolo(bboxes, img_width, img_height, save_path,farrowkey):
    with open(save_path, "w") as f:
        for box in bboxes:
            x1, y1, x2, y2, score, cls = box
            if cls in [2,5,6]:
                if farrowkey=='s':
                    #mgs = f"Det class : {cls} saved : 5"
                    cls=5
                    print(f"Save class {cls} -> 5(Farrow_S)")
                    
                elif farrowkey=='e':
                    #mgs = f"Det class : {cls} saved : 6"
                    cls=6
                    print(f"Save class {cls} -> 6(Farrow_E)")
                elif farrowkey=='f':
                    #mgs = f"Det class : {cls} saved : 2"
                    cls=2
                    print(f"Save class {cls} -> 2(Farrowing)") 

            x_center = (x1 + x2) / 2.0 / img_width
            y_center = (y1 + y2) / 2.0 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
