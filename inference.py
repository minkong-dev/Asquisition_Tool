from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import numpy as np
import time
from util import make_savepath,preprocess_yolo,postprocess_yolo,plot_label_image,convert_to_yolo
import cv2
import os
import gc

access_models = ["yolo_model_v11","yolo_model_v12","farrow"]
def nms_boxes(dets, iou_threshold=0.5):
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    scores=dets[:,4]
    sorted_indices = np.argsort(scores)[::-1]
    keep = []
    while len(sorted_indices) > 0:
        # 가장 높은 스코어 선택
        current = sorted_indices[0]
        keep.append(current)
        others = sorted_indices[1:]
        if len(sorted_indices) == 1:
            break
        # 교집합 계산
        xx1 = np.maximum(x1[current], x1[others])
        yy1 = np.maximum(y1[current], y1[others])
        xx2 = np.minimum(x2[current], x2[others])
        yy2 = np.minimum(y2[current], y2[others])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        # IoU 계산
        union = areas[current] + areas[others] - intersection
        iou = intersection / union
        # IoU가 임계값보다 낮은 박스만 유지
        indices = others[iou <= iou_threshold]
        sorted_indices = indices

    return dets[keep]

class Inference_server:
    def __init__(self,url,modelname,log_callback=None):
        self.server_url = url
        self.modelname = modelname
        self.model_version = "1"  # 기본 버전
        self.log_callback = log_callback
        self.demo=False
        self.annotation_list = []  # 어노테이션 메모리 (프레임번호 + 키만 저장)
        self.annotation_dict = {}  # 딕셔너리: 빠른 조회용 (frame_idx -> annotation_item)

        if modelname=="yolo_model_v11_base":
            self.classes={
                    0: "Standing",
                    1: "Lying",
                    2: "Farrowing",
                    3: "Nothing",
                    4: "Womb"
                }
        elif modelname == "yolo_model_v11":
            self.modelname ="yolo_model_v11"
            self.classes={   
                    0: "Standing",
                    1: "Lying",
                    2: "Farrowing",
                    3: "Nothing",
                    4: "Womb",
                    5: "Farrow_start",
                    6: "Farrow_end",
                    7: "Hand"
                }
            self.demo=True
        else:
            self.classes={
                    0: "sow_normal",
                    1: "sow_farrowing"
                }

        self.label_classes={
                0: "Standing",
                1: "Lying",
                2: "Farrowing",
                3: "Nothing",
                4: "Womb",
                5: "Farrow_S",
                6: "Farrow_E",
                7: "hand"
            }
    def health_check(self, timeout=5.0, interval=0.5):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with grpcclient.InferenceServerClient(
                    url=self.server_url,
                    verbose=False,
                ) as client:
                    if not client.is_server_live():
                        raise Exception("Triton 서버가 Live 상태가 아닙니다.")
                    if not client.is_server_ready():
                        raise Exception("Triton 서버가 Ready 상태가 아닙니다.")
                    # if not client.is_model_ready(self.modelname):
                    #     raise Exception(f"모델 '{self.modelname}'이(가) Ready 상태가 아닙니다.")
                check=True
                break
            except Exception as e:
                check=False
                # 연결 실패 시 잠시 대기 후 재시도
                time.sleep(interval)
        return check
        #raise TimeoutError(f"서버 연결 실패: {timeout}초 내에 연결할 수 없음")

    def get_model_list(self, timeout=5.0, interval=0.5):
        """트리톤 서버에서 모델 리스트를 가져와서 파싱합니다."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with grpcclient.InferenceServerClient(
                    url=self.server_url,
                    verbose=False,
                ) as client:
                    repository_index = client.get_model_repository_index()
                    #print("Raw repository index:", repository_index)
                    
                    # 모델 정보 파싱
                    models = []
                    for model in repository_index.models:
                        
                        model_info = {
                            'name': model.name,
                            'version': model.version,
                            'state': model.state
                        }
                        if model.name in access_models:
                            models.append(model_info)
                    
                    return models
                    
            except Exception as e:
                print(f"모델 리스트 가져오기 실패: {e}")
                # 연결 실패 시 잠시 대기 후 재시도
                time.sleep(interval)
        
        return []

    def cal_pose(self,result_array):
        # 좌표와 점수 추출
        x1, y1, x2, y2, scores, classes = result_array.T
        #print(result_array.T)
        # 중심점 계산 (벡터화)
        cx = (x1 + x2) / 2 
        cy = (y1 + y2) / 2
        
        # 거리 계산 (벡터화)
        dist = np.sqrt((cx - self.center[0])**2 + (cy - self.center[1])**2)
        
        # 거리 점수 계산 (벡터화)
        dist_scores = 1 - (dist / self.max_dist)
        
        # 최종 점수 계산 (벡터화)
        final_scores = self.alpha * dist_scores + self.beta * scores
        
        # 최고 점수 인덱스 찾기
        best_idx = np.argmax(final_scores)
        best_score = final_scores[best_idx]
        best_box = result_array[best_idx].tolist()
        status = int(classes[best_idx])

        return [best_box]
    def cal_farrow(self,farrow_results):

        if len(farrow_results) == 0:
            out_result=[]
        else:
            target_classes = [2, 5, 6]

            mask = np.isin(farrow_results[:, 5], target_classes)
            
            # target_classes에 포함된 것들만 (기존)
            output_results = farrow_results[mask]
            remaining_results = farrow_results[~mask]

            if len(output_results)>1:
                output_results = nms_boxes(output_results)
            scores = output_results[:, 4]
            clss = output_results[:, 5]
            bboxes = output_results[:, :4]
            detections = [[round(bbox[0],2),round(bbox[1],2),round(bbox[2],2),round(bbox[3],2), round(score,3), int(cls)] for bbox, score,cls in zip(bboxes, scores,clss)]
            out_result = detections+remaining_results.tolist()
            
        return out_result

    def start_label(self, video_path, save_path, model_name=None, version=None):

        # 모델명과 버전 설정
        if model_name:
            self.modelname = model_name

        if version:
            self.model_version = str(version)
        else:
            self.model_version = "1"  # 기본 버전
        
        print(f"사용 모델: {self.modelname}, 버전: {self.model_version}")
        
        # 어노테이션 메모리 초기화 (영상 시작 시)
        self.annotation_list = []  # 리스트: 저장 순서 유지 (롤백용)
        self.annotation_dict = {}  # 딕셔너리: 빠른 조회용 (frame_idx -> annotation_item)
        
        # 저장 경로 설정
        save_dirs = {
            'l': f'{save_path}/stand_lying',
            's': f'{save_path}/farrow_start',
            'e': f'{save_path}/farrow_end',
            'f': f'{save_path}/farrowing',
            "w":f'{save_path}/womb',
            "n":f'{save_path}/nothing',
            "h":f'{save_path}/hand',
        }

        make_savepath(save_dirs,self.label_classes)
        save_class = {
            'l':"Lying",
            's':"Farrow_Start",
            'f':"Farrowing",
            'e':"Farrow_End",
            'w':"Womb",
            'n':"Nothing",
            "h":"Hand"
        }
        save_count = {
            'l':0,
            's':0,
            'f':0,
            'e':0,
            'w':0,
            'n':0,
            "h":0
        }
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 원본 동영상의 너비
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 원본 동영상의 높이
        
        self.center = (width // 2, height // 2)  # (320, 240)
        self.max_dist = np.sqrt(self.center[0]**2 + self.center[1]**2)
        self.alpha = 0.6
        self.beta = 0.4

        frame_skip_L = 60
        frame_skip_S = 24
        frame_idx = 0
        paused = False

        base_name = os.path.basename(video_path)
        filename, ext = os.path.splitext(base_name)
        window_name = f'Video : {base_name}'
        full=False
        screen=False
        label = True
        resize_w,resize_h = 1280,720
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 창 크기 조절 가능하게 만듦
        cv2.resizeWindow(window_name, resize_w, resize_h)    
        check_frame=0
        model_infer_time,cnt=0,0
        
        # 저장 함수 정의 (start_label 내부 함수)
        def save_annotation_list():
            """annotation_list에 기록된 프레임들을 실제로 저장"""
            if not self.annotation_list:
                self.log_callback({"log": "저장할 어노테이션이 없습니다."})
                return 0  # 저장된 개수 0 반환
            
            saved_count = 0
            failed_count = 0
            is_first_successful_save = True  # 첫 저장 성공 플래그
            
            for item in self.annotation_list:
                frame_idx = item["frame_idx"]
                key = item["annotation_key"]
                save_frame = item["frame"]  # 저장된 프레임 이미지 사용
                draw_result = item["draw_result"]  # 저장된 추론 결과 사용
                # frame.shape = (height, width, channels)
                save_height, save_width = save_frame.shape[:2]
                
                # 파일 경로 생성
                save_txt = os.path.join(save_dirs[key], f"{filename}_{frame_idx}.txt")
                save_img = os.path.join(save_dirs[key], f"{filename}_{frame_idx}.jpg")
                
                # 중복 체크
                if os.path.isfile(save_txt) or os.path.isfile(save_img):
                    self.log_callback({"log": f"파일이 이미 존재합니다: {save_img}"})
                    continue
                
                # 저장 수행 (저장된 프레임과 추론 결과 사용)
                try:
                    convert_to_yolo(draw_result, save_width, save_height, save_txt, key)
                    cv2.imwrite(save_img, save_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    saved_count += 1
                    
                    # 저장 로그 (첫 저장 성공 시에만 CSV 기록 요청)
                    self.log_callback({
                        "log": f"일괄 저장 중: {saved_count}/{len(self.annotation_list)}",
                        "type": save_class[key],
                        "img": save_img,
                        "label": save_txt,
                        "record_to_csv": is_first_successful_save,  # 첫 저장 성공 시에만 True
                        "video_path": video_path
                    })
                    is_first_successful_save = False  # 첫 저장 후 False로 변경
                except Exception as e:
                    failed_count += 1
                    self.log_callback({"log": f"저장 실패 (프레임 {frame_idx}): {str(e)}"})
            
            # 완료 메시지
            if failed_count > 0:
                self.log_callback({
                    "log": f"일괄 저장 완료: {saved_count}개 저장됨, {failed_count}개 실패"
                })
            else:
                self.log_callback({
                    "log": f"일괄 저장 완료: 총 {saved_count}개 저장됨"
                })
            
            # 메모리 정리: 저장된 항목들의 큰 데이터를 명시적으로 삭제
            for item in self.annotation_list:
                if "frame" in item:
                    del item["frame"]
                if "draw_result" in item:
                    del item["draw_result"]
            
            # 리스트와 딕셔너리 초기화 (참조 해제)
            self.annotation_list.clear()
            self.annotation_list = []
            self.annotation_dict.clear()
            self.annotation_dict = {}
            
            # 가비지 컬렉션 강제 실행
            gc.collect()
            
            return saved_count  # 저장된 개수 반환
        
        with grpcclient.InferenceServerClient(url=self.server_url, verbose=False) as client:
            ## dynamic input
            ## dynamic input
            if self.modelname in access_models:
                w,h = 640,640
                #model_inputs = [grpcclient.InferInput("images", (1,3,320,320), np_to_triton_dtype(np.float16)),]
            else:    
                w,h = 320,320
            model_inputs = [grpcclient.InferInput("images", (1,3,w,h), np_to_triton_dtype(np.float16)),]
            model_outputs= [
                            grpcclient.InferRequestedOutput("num_dets"),
                            grpcclient.InferRequestedOutput("det_boxes"),
                            grpcclient.InferRequestedOutput("det_scores"),
                            grpcclient.InferRequestedOutput("det_classes"),
                            ]
            msgg = "Start Labeling"
            while True:
                #if not paused:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print("영상 읽기 실패 또는 오류")
                    # 예기치 못한 종료: 안전장치로 저장 후 종료
                    saved_count = save_annotation_list()
                    cap.release()
                    cv2.destroyAllWindows()
                    # 메모리 정리
                    gc.collect()
                    return (None, saved_count)  # 오류 상태 + 저장된 개수
                cnt+=1
                check_frame+=1
                start = time.time()
                ordered, one_shape, ratio = preprocess_yolo(frame,True,h,w)
                model_inputs[0].set_data_from_numpy(ordered)

                response = client.infer(model_name = self.modelname,
                                            model_version = self.model_version,
                                            inputs = model_inputs,
                                            outputs = model_outputs)
                pose_result,farrow_results = postprocess_yolo(response, one_shape,ratio)

                f_count=len(farrow_results)
                end = time.time()

                #print(f"yolo box result : {results}")
                frame_display = frame.copy()
                
                # 현재 프레임에 저장된 어노테이션이 있는지 확인 (딕셔너리로 빠른 조회)
                saved_annotation = self.annotation_dict.get(frame_idx, None)
                
                # 추론 결과를 사용하여 박스 표시 (저장된 프레임이어도 현재 상태를 보기 위해 항상 표시)
                pose_best = self.cal_pose(pose_result)
                farrow_best = self.cal_farrow(farrow_results)
                draw_result = pose_best+farrow_best
                
                # 저장된 어노테이션이 있으면 클래스명 가져오기
                marked_class = None
                if saved_annotation:
                    annotation_key = saved_annotation["annotation_key"]
                    marked_class = save_class.get(annotation_key, "Unknown")
                
                #print(f"Pose : {pose_best}, Farrow : {farrow_best}, Draw : {draw_result}")
                # 추론 결과 박스 표시
                plot_label_image(frame_display, draw_result,self.classes)
                
                # 저장된 어노테이션이 있으면 "Mark: 클래스명" 텍스트 추가 표시
                if marked_class:
                    cv2.putText(frame_display, f"Mark : {marked_class}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 30, 68), 2)

                infer_time = end - start
                model_infer_time+=infer_time
                avg_infer_time= model_infer_time/cnt
                fps_text = f"{frame_idx}/{total_frames} {avg_infer_time:.4f}"

                cv2.putText(frame_display, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.putText(frame_display, f"Check: {check_frame}, S:{save_count['s']},F:{save_count['f']},E:{save_count['e']},N:{save_count['n']},W:{save_count['w']},h:{save_count['h']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 255), 2)
                cv2.putText(frame_display, msgg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # Latest: 마지막 기록한 어노테이션 (스택 최상단)
                if self.annotation_list:
                    latest = self.annotation_list[-1]  # 마지막 항목
                    latest_frame = latest["frame_idx"]
                    latest_key = latest["annotation_key"]
                    latest_status = save_class.get(latest_key, latest_key)
                    cv2.putText(frame_display, f"Latest: {latest_frame} ({latest_status})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    cv2.putText(frame_display, "Latest: None", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame_display, f"Pending: {len(self.annotation_list)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 원하는 크기로 지정
                if label:
                    cv2.imshow(window_name, frame_display)
                else:
                    cv2.imshow(window_name, frame)
                #cv2.imshow(f'Video : {filename}', frame_display)

                key = cv2.waitKey(0 if paused else 30)& 0xFF# ← 이 줄이 반드시 있어야 함!
                #print("key",key)
                if key == ord('q'):
                    # 저장 수행
                    saved_count = save_annotation_list()
                    msg = {"log":f"종료 시그널을 받았습니다."}
                    self.log_callback(msg)
                    cap.release()
                    cv2.destroyAllWindows()
                    # 메모리 정리
                    gc.collect()
                    return ('quit', saved_count)  # 완전 종료 + 저장된 개수
                elif key == ord('0'):  # 숫자 0: 다음 영상으로
                    # 저장 수행
                    saved_count = save_annotation_list()
                    msg = {"log":f"다음 영상으로 넘어갑니다."}
                    self.log_callback(msg)
                    cap.release()
                    cv2.destroyAllWindows()
                    # 메모리 정리
                    gc.collect()
                    return ('next', saved_count)  # 다음 영상 + 저장된 개수
                elif key == 13:
                    label = not label
                    print("Change Screen")
                elif key == 45:
                    screen = not screen
                    if not full:
                        if screen:
                            resize_w,resize_h = 640,480
                        else:
                            resize_w,resize_h = 1280,720
                        self.log_callback({"log":f" 스크린 사이즈 {resize_w}x{resize_h}"})
                        msgg = f" Screen Size {resize_w}x{resize_h}"
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, resize_w, resize_h)
                    else:
                        self.log_callback({"log":f" 전체화면을 먼저 해제하세요"})

                elif key == 43:
                    full = not full
                    #print(full)
                    if full:
                        self.log_callback({"log":f" 전체화면"})
                        msgg = f"Full Screen"
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        self.log_callback({"log":f" 전체화면 취소"})
                        msgg = f"Small Screen"
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, resize_w, resize_h)
                    
                elif key == 32 or key==53:  # Space: Pause toggle / 5
                    self.log_callback({"log":f" 일시정지"})
                    msgg = f"STOP"
                    paused = not paused
                elif key == 52:  # 4
                    self.log_callback({"log":f"< 1 프레임 전으로"})
                    frame_idx = max(0, frame_idx - 1)
                    msgg = f"< 1 Frame Before"
                    #paused = True
                #elif key == 83:  # Right arrow
                elif key == 54:  # 6
                    self.log_callback({"log":f"> 1 프레임 후로"})
                    frame_idx = min(total_frames - 1, frame_idx + 1)
                    msgg = f"> 1 Frame After"
                #elif key == 81:  # 4
                elif key == 55:  # 7
                    self.log_callback({"log":f"<<< {frame_skip_L} 프레임 전으로"})
                    frame_idx = max(0, frame_idx - frame_skip_L)
                    msgg = f"<<< {frame_skip_L}  Frame Before"
                    #paused = True
                #elif key == 83:  # 6
                elif key == 57:  # 9
                    self.log_callback({"log":f">>> {frame_skip_L} 프레임 후로"})
                    frame_idx = min(total_frames - 1, frame_idx + frame_skip_L)
                    msgg = f">>> {frame_skip_L} Frame After"
                    #paused = True
                elif key == 49:  # 3
                    self.log_callback({"log":f"<< {frame_skip_S} 프레임 전으로"})
                    frame_idx = max(0, frame_idx - frame_skip_S)
                    msgg = f"<< {frame_skip_S}  Frame Before"
                    #paused = True
                #elif key == 83:  # Right arrow
                elif key == 51:  # 1
                    self.log_callback({"log":f">> {frame_skip_S} 프레임 후로"})
                    msgg = f">> {frame_skip_S} Frame After"
                    frame_idx = min(total_frames - 1, frame_idx + frame_skip_S)
                elif key == 8:  # backspace: 롤백
                    if self.annotation_list:
                        removed = self.annotation_list.pop()
                        removed_frame_idx = removed['frame_idx']
                        # 딕셔너리에서도 제거
                        if removed_frame_idx in self.annotation_dict:
                            del self.annotation_dict[removed_frame_idx]
                        # 해당 키의 save_count 감소
                        removed_key = removed['annotation_key']
                        if removed_key in save_count:
                            save_count[removed_key] = max(0, save_count[removed_key] - 1)
                        
                        self.log_callback({
                            "log": f"롤백: 프레임 {removed_frame_idx}의 {save_class.get(removed_key, 'Unknown')} 어노테이션 취소됨 (남은 개수: {len(self.annotation_list)})"
                        })
                        msgg = f"Undo: Frame {removed_frame_idx}"
                    else:
                        self.log_callback({"log": "롤백할 어노테이션이 없습니다."})
                        msgg = "No Annotation to Undo"
                elif key > 0 and chr(key) in save_dirs:
                    # 메모리에 기록 (프레임 번호, 키, 프레임 이미지, 추론 결과)
                    annotation_item = {
                        "frame_idx": frame_idx,
                        "annotation_key": chr(key),
                        "frame": frame.copy(),  # 현재 프레임 이미지 저장 (RAM) - frame.shape에서 width/height 추출 가능
                        "draw_result": draw_result[:] if isinstance(draw_result, list) else draw_result.copy() if hasattr(draw_result, 'copy') else draw_result  # 추론 결과 복사 (RAM)
                    }
                    self.annotation_list.append(annotation_item)
                    # 딕셔너리에도 추가 (빠른 조회용)
                    self.annotation_dict[frame_idx] = annotation_item
                    
                    # UI 피드백만 (실제 저장은 하지 않음)
                    save_count[chr(key)] += 1
                    msgtype = save_class[chr(key)]
                    msg = {
                        "log": f"어노테이션 기록됨 (프레임 {frame_idx}, 총 {len(self.annotation_list)}개 대기)",
                        "type": msgtype,
                        "img": None,  # 아직 저장 안함
                        "label": None
                    }
                    msgg = f"Recorded:{frame_idx} ({len(self.annotation_list)} pending)"
                    self.log_callback(msg)
                    
                if not paused:
                    frame_idx += 1
                    if frame_idx >= total_frames:
                        print("마지막 프레임 도달 - 무한재생")
                        frame_idx = 0  # 처음으로 리셋
                        self.log_callback({"log": "영상 끝 - 처음부터 반복재생합니다"})
                        msgg = "Looping Video"
                        # break 제거, 계속 재생

            # 이 코드는 실행되지 않음 (while True 루프는 q/0 키 또는 오류로만 종료됨)
            # 하지만 안전장치로 남겨둠
            cap.release()
            cv2.destroyAllWindows()
            
            # 혹시 모를 남은 어노테이션도 저장 (안전장치)
            saved_count = save_annotation_list()
            
            # 메모리 정리
            gc.collect()
            
            # 실제로는 이 코드에 도달하지 않음 (무한재생 + q/0 키 처리로 인해)
            # 하지만 혹시 모를 경우를 대비해 남겨둠
            return (None, saved_count)


    # def start_label_demo(self, video_path, save_path, model_name=None, version=None):
    #     # 모델명과 버전 설정
    #     if model_name:
    #         self.modelname = model_name
    #     if version:
    #         self.model_version = str(version)
    #     else:
    #         self.model_version = "1"  # 기본 버전
        
    #     print(f"사용 모델: {self.modelname}, 버전: {self.model_version}")
        
    #     # 저장 경로 설정
    #     save_dirs = {
    #         'l': f'{save_path}/stand_lying',
    #         's': f'{save_path}/farrow_start',
    #         'e': f'{save_path}/farrow_end',
    #         'f': f'{save_path}/farrowing',
    #         "w":f'{save_path}/womb',
    #         "n":f'{save_path}/nothing',
    #         "h":f'{save_path}/hand'
    #     }

    #     make_savepath(save_dirs,self.label_classes)
    #     save_class = {
    #         'l':"Lying",
    #         's':"Farrow_Start",
    #         'f':"Farrowing",
    #         'e':"Farrow_End",
    #         'w':"Womb",
    #         'n':"Nothing",
    #         "h":"Hand"
    #     }

    #     cap = cv2.VideoCapture(video_path)
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 원본 동영상의 너비
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 원본 동영상의 높이

    #     frame_skip_L = 60
    #     frame_skip_S = 24
    #     frame_idx = 0
    #     paused = False

    #     base_name = os.path.basename(video_path)
    #     filename, ext = os.path.splitext(base_name)
    #     window_name = f'Video : {base_name}'
    #     full=False
    #     screen=False
    #     label = True
    #     resize_w,resize_h = 1280,720
    #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 창 크기 조절 가능하게 만듦
    #     cv2.resizeWindow(window_name, resize_w, resize_h)    

    #     model_infer_time,cnt=0,0
    #     with grpcclient.InferenceServerClient(url=self.server_url, verbose=False) as client:
    #         ## dynamic input
    #         model_inputs = [grpcclient.InferInput("images", (1,3,640,640), np_to_triton_dtype(np.float16)),]
    #         model_outputs= [
    #                         grpcclient.InferRequestedOutput("num_dets"),
    #                         grpcclient.InferRequestedOutput("det_boxes"),
    #                         grpcclient.InferRequestedOutput("det_scores"),
    #                         grpcclient.InferRequestedOutput("det_classes"),
    #                         ]
  
    #         while True:
    #             #if not paused:
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    #             ret, frame = cap.read()
    #             if not ret:
    #                 print("영상 끝 또는 오류")
    #                 break
    #             cnt+=1
    #             start = time.time()
    #             ordered, one_shape, ratio = preprocess_yolo(frame,True,640,640)
    #             model_inputs[0].set_data_from_numpy(ordered)
    #             response = client.infer(model_name = self.modelname,
    #                                         model_version = self.model_version,
    #                                         inputs = model_inputs,
    #                                         outputs = model_outputs)
    #             pose_result,farrow_results = postprocess_yolo(response, one_shape,ratio)

    #             f_count=len(farrow_results)
    #             end = time.time()

    #             #print(f"yolo box result : {results}")
    #             frame_display = frame.copy()

    #             if len(farrow_results) == 0:
    #                 detections = []
    #                 remaining_results=[]
    #             else:
    #                 target_classes = [2, 5, 6]

    #                 mask = np.isin(farrow_results[:, 5], target_classes)
                    
    #                 # target_classes에 포함된 것들만 (기존)
    #                 output_results = farrow_results[mask]
    #                 remaining_results = farrow_results[~mask]

    #                 if len(output_results)>1:
    #                     output_results = nms_boxes(output_results)
  
    #                 farrow_result = output_results+remaining_results
                    
    #             draw_result = pose_result.tolist()+farrow_result.tolist()
    #             plot_label_image(frame_display, draw_result,self.classes)

    #             infer_time = end - start
    #             model_infer_time+=infer_time
    #             avg_infer_time= model_infer_time/cnt
    #             fps_text = f"{frame_idx}/{total_frames} {avg_infer_time:.4f}"

    #             cv2.putText(frame_display, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    #             if f_count >0:
    #                 cv2.putText(frame_display, "Farrowing", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    #             else:
    #                 cv2.putText(frame_display, "Nothing", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 255), 2)

    #             # 원하는 크기로 지정
    #             if label:
    #                 cv2.imshow(window_name, frame_display)
    #             else:
    #                 cv2.imshow(window_name, frame)
    #             #cv2.imshow(f'Video : {filename}', frame_display)

    #             key = cv2.waitKey(0 if paused else 30)& 0xFF# ← 이 줄이 반드시 있어야 함!
    #             #print("key",key)
    #             if key == ord('q'):
    #                 msg = {"log":f"종료 시그널을 받았습니다."}
    #                 self.log_callback(msg)
    #                 break
    #             elif key == 13:
    #                 label = not label
    #                 print("Change Screen")
    #             elif key == 45:
    #                 screen = not screen
    #                 if not full:
    #                     if screen:
    #                         resize_w,resize_h = 640,480
    #                     else:
    #                         resize_w,resize_h = 1280,720
    #                     self.log_callback({"log":f" 스크린 사이즈 {resize_w}x{resize_h}"})
    #                     cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    #                     cv2.resizeWindow(window_name, resize_w, resize_h)
    #                 else:
    #                     self.log_callback({"log":f" 전체화면을 먼저 해제하세요요"})
    #             elif key == 43:
    #                 full = not full
    #                 #print(full)
    #                 if full:
    #                     self.log_callback({"log":f" 전체화면"})
    #                     cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #                 else:
    #                     self.log_callback({"log":f" 전체화면 취소"})
    #                     cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    #                     cv2.resizeWindow(window_name, resize_w, resize_h)
                    
    #             elif key == 32 or key==53:  # Space: Pause toggle / 5
    #                 self.log_callback({"log":f" 일시정지"})
    #                 paused = not paused
    #             elif key == 52:  # 4
    #                 self.log_callback({"log":f"< 1 프레임 전으로"})
    #                 frame_idx = max(0, frame_idx - 1)
    #                 #paused = True
    #             #elif key == 83:  # Right arrow
    #             elif key == 54:  # 6
    #                 self.log_callback({"log":f"> 1 프레임 후로"})
    #                 frame_idx = min(total_frames - 1, frame_idx + 1)

    #             #elif key == 81:  # 4
    #             elif key == 55:  # 7
    #                 self.log_callback({"log":f"<<< {frame_skip_L} 프레임 전으로"})
    #                 frame_idx = max(0, frame_idx - frame_skip_L)
    #                 #paused = True
    #             #elif key == 83:  # 6
    #             elif key == 57:  # 9
    #                 self.log_callback({"log":f">>> {frame_skip_L} 프레임 후로"})
    #                 frame_idx = min(total_frames - 1, frame_idx + frame_skip_L)

    #                 #paused = True
    #             elif key == 49:  # 3
    #                 self.log_callback({"log":f"<< {frame_skip_S} 프레임 전으로"})
    #                 frame_idx = max(0, frame_idx - frame_skip_S)
    #                 #paused = True
    #             #elif key == 83:  # Right arrow
    #             elif key == 51:  # 1
    #                 self.log_callback({"log":f">> {frame_skip_S} 프레임 후로"})
    #                 frame_idx = min(total_frames - 1, frame_idx + frame_skip_S)
    #             elif key > 0 and chr(key) in save_dirs:
    #                 # 'l': f'{save_path}/stand_lying',
    #                 # 's': f'{save_path}/farrow_start',
    #                 # 'e': f'{save_path}/farrow_end',
    #                 # 'f': f'{save_path}/farrowing',
    #                 # "w":f'{save_path}/womb',
    #                 # "n":f'{save_path}/nothing',
    #                 # "h":f'{save_path}/hand'
    #                 save_txt = os.path.join(save_dirs[chr(key)], f"{filename}_{frame_idx}.txt")
    #                 save_path = os.path.join(save_dirs[chr(key)], f"{filename}_{frame_idx}.jpg")
    #                 msgtype = save_class[chr(key)]
                    
    #                 if os.path.isfile(save_txt):
    #                     msg = {"log":f"파일이 이미 존재합니다. -- {save_txt}"}
    #                     self.log_callback(msg)
    #                     continue
    #                     # for savenum in range(50):
    #                     #     save_txt = os.path.join(save_dirs[chr(key)], f"{filename}_{frame_idx}_{savenum}.txt")
    #                     #     if os.path.isfile(save_txt):
    #                     #         continue
    #                     #     else:
    #                     #         break

    #                 if os.path.isfile(save_path):
    #                     msg = {"log":f"파일이 이미 존재합니다. -- {save_path}"}
    #                     self.log_callback(msg)
    #                     continue
    #                     # for savenum in range(50):
    #                     #     save_path = os.path.join(save_dirs[chr(key)], f"{filename}_{frame_idx}_{savenum}.jpg")
    #                     #     if os.path.isfile(save_path):
    #                     #         continue
    #                     #     else:
    #                     #         break

    #                 msg = {
    #                     "log":"파일을 저장합니다.",
    #                     "type":msgtype,
    #                     "img":save_path,
    #                     "label":save_txt
    #                 }
                    
    #                 self.log_callback(msg)
    #                 convert_to_yolo(draw_result, width, height, save_txt,chr(key))
    #                 cv2.imwrite(save_path, frame,[cv2.IMWRITE_JPEG_QUALITY, 100])
    #                 # print(f"프레임 저장: {save_path}")
    #                 # print(f"라벨 저장: {save_txt}")
                    
    #             if not paused:
    #                 frame_idx += 1
    #                 if frame_idx >= total_frames:
    #                     print("마지막 프레임 도달")
    #                     self.log_callback({"log":f"마지막 프레임 도달, 비디오를 종료합니다."})
    #                     break

    #         cap.release()
    #         cv2.destroyAllWindows()