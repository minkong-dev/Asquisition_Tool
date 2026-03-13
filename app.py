import sys
import os
import glob
import csv
from inference import Inference_server
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QTextEdit,QLineEdit,
    QVBoxLayout, QWidget, QComboBox, QHBoxLayout, QGroupBox, QMessageBox, QRadioButton, QButtonGroup
)
from PyQt5.QtWidgets import QSizePolicy
# from PyQt6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox,QComboBox
# )
from PyQt5.QtCore import Qt
import cv2

serverlist = ["inference_server","remote_server" ,"local","직접 입력"]
class LabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Labeling Tool")
        self.setGeometry(100, 100, 1100, 600)


        self.save_path=None
        self.video_path=None
        self.video_folder_path=None  # 폴더 모드용
        self.healthcheck=False
        self.models_info = {}  # 모델 정보 저장용
        self.save_class = {
            "Lying":0,
            "Farrow_Start":0,
            "Farrowing":0,
            "Farrow_End":0,
            "Womb":0,
            "Nothing":0,
            "Hand":0
        }
        self.savenum=0
        self.video_list = []  # 폴더 모드용 영상 리스트 

        # === 전체 레이아웃 ===
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        #right_log_layout=QVBoxLayout()
        # === 왼쪽: 설정 그룹 ===
        setting_group = QGroupBox("[설정] 설정")
        setting_layout = QVBoxLayout()

        self.server_input = QComboBox()
        self.server_input.addItems(serverlist)
        
        setting_layout.addWidget(QLabel("추론서버 선택"))
        setting_layout.addWidget(self.server_input)

        # 직접 입력용 QLineEdit (처음엔 숨김)
        self.server_input_custom = QLineEdit(self)
        self.server_input_custom.setPlaceholderText("서버 주소 입력 (예: 192.168.1.100)")
        self.server_input_custom.hide()  # 처음엔 숨김
        setting_layout.addWidget(self.server_input_custom)
        # 콤보박스 변경 시 이벤트 연결
        self.server_input.currentTextChanged.connect(self.on_server_input_changed)


        self.protocol = QComboBox()
        self.protocol.addItems(["grpc", "http"])
        setting_layout.addWidget(QLabel("프로토콜 선택"))
        setting_layout.addWidget(self.protocol)

        # 모델과 버전 선택을 같은 라인에 배치
        model_version_layout = QHBoxLayout()
        
        # 모델 선택
        model_label = QLabel("모델 선택")
        self.model_select = QComboBox()
        self.model_select.addItems(["서버 연결 후 모델을 불러오세요..."])
        model_version_layout.addWidget(model_label)
        model_version_layout.addWidget(self.model_select)
        
        # 버전 선택
        version_label = QLabel("버전 선택")
        self.version_select = QComboBox()
        self.version_select.addItems(["모델을 먼저 선택하세요"])
        model_version_layout.addWidget(version_label)
        model_version_layout.addWidget(self.version_select)
        
        setting_layout.addLayout(model_version_layout)
        
        # 모델 선택 시 버전 업데이트 이벤트 연결
        self.model_select.currentTextChanged.connect(self.on_model_changed)

        # Triton 서버 입력
        self.farm_input = QLineEdit(self)
        self.farm_input.setPlaceholderText("농장코드(저장폴더): FARM001")
        setting_layout.addWidget(QLabel("농장코드 입력"))
        setting_layout.addWidget(self.farm_input)
        #self.farm_input
        setting_group.setLayout(setting_layout)
        left_layout.addWidget(setting_group)

        # === 왼쪽: 기능 그룹 ===
        action_group = QGroupBox("[기능] 기능")
        action_layout = QVBoxLayout()

        self.server_alive = QPushButton("서버 연결 확인")
        self.btn_select_save_path = QPushButton("[폴더] 저장 경로 선택")
        
        # 영상 모드 선택 라디오 버튼
        video_mode_label = QLabel("영상 모드:")
        self.radio_single = QRadioButton("단일 영상")
        self.radio_folder = QRadioButton("폴더 모드")
        self.radio_single.setChecked(True)  # 기본값: 단일 영상
        
        video_mode_layout = QHBoxLayout()
        video_mode_layout.addWidget(video_mode_label)
        video_mode_layout.addWidget(self.radio_single)
        video_mode_layout.addWidget(self.radio_folder)
        
        self.video_button = QPushButton("[영상] 비디오 선택")
        self.start_button = QPushButton("> 시작")
        self.btn_exit = QPushButton("X 종료")

        action_layout.addWidget(self.server_alive)
        action_layout.addWidget(self.btn_select_save_path)
        action_layout.addLayout(video_mode_layout)
        action_layout.addWidget(self.video_button)
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.btn_exit)
        
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)
        left_layout.addStretch()

        main_layout.addLayout(left_layout, 3)

        # === 오른쪽: 가이드 ===
        guide_group = QGroupBox("[가이드] 단축키 가이드")
        guide_layout = QVBoxLayout()
        self.guide_text = QTextEdit()
        self.guide_text.setReadOnly(True)
        self.guide_text.setText(
            """
            [영상] 영상 재생 제어 (숫자키패드)
            ━━━━━━━━━━━━━━
            일시정지/재생     Space or NUM(5)
            1 프레임 이동     NUM(4) / NUM(6)
            60 프레임 이동    NUM(7) / NUM(9)
            24 프레임 이동    NUM(1) / NUM(3)

            [화면] 화면 설정
            ━━━━━━━━━━━━━━
            전체화면          NUM(+)
            화면 크기 조정    NUM(-)
            라벨 표시 토글    Enter

            📝 어노테이션 저장 (키보드)
            ━━━━━━━━━━━━━━
            Lying             L
            Farrow_Start      S
            Farrowing         F
            Farrow_End        E
            Womb              W
            Nothing           N
            Hand              H

            [기능] 작업 제어
            ━━━━━━━━━━━━━━
            롤백 (Undo)       Backspace
            다음 영상 (폴더)  NUM(0)
            완전 종료         Q

        """
        )
        guide_layout.addWidget(self.guide_text)
        guide_group.setLayout(guide_layout)
        right_layout.addWidget(guide_group)
        self.guide_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # [OK] [추가] 처리 현황 로그 창
        # right_layout = QVBoxLayout()
        # guide_label = QLabel("[가이드] 사용 가이드")
        # guide_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        # right_layout.addWidget(guide_label)

        # [현황] 처리 현황 GroupBox
        log_group = QGroupBox("[현황] 처리 현황")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #f4f4f4; font-size: 16px;")
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_text.setText(
            "-------- 처리현황 -------\n"
            f"Message: 라벨링 작업 전\n"
            f" Stand/Lying : 0\n"
            f" Farrow_Start : 0\n"
            f" Farrowing : 0\n"
            f" Farrow_End : 0\n"
            f" Womb : 0\n"
            f" Nothing : 0\n"
            f" Total Save : 0\n"
        )

        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)

        # [현황] 진행 상황 표시 (폴더 모드용)
        progress_group = QGroupBox("[진행] 진행 상황 (폴더 모드)")
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("전체: 0 | 처리 완료: 0 | 남음: 0")
        self.progress_label.setStyleSheet("font-size: 16px; color: #2a72b5; font-weight: bold;")
        progress_layout.addWidget(self.progress_label)
        
        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)

        right_layout.addStretch()

        main_layout.addLayout(right_layout, 3)

        # === 최종 설정 ===
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # === 스타일 적용 ===
        style = """
        QPushButton {
            font-size: 14px;
            padding: 8px;
        }
        QPushButton:hover {
            background-color: #d0f0ff;
        }
        QComboBox {
            font-size: 14px;
            padding: 4px;
        }
        QLabel {
            font-weight: bold;
        }
        QGroupBox {
            font-size: 15px;
            font-weight: bold;
            border: 1px solid #bbbbbb;
            border-radius: 5px;
            margin-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            color: #2a72b5;
        }
        QTextEdit {
            background-color: #fcfcfc;
            font-size: 14px;
            padding: 10px;
            border: 1px solid #dddddd;
            border-radius: 4px;
        }
        """
        # style = """
        # QPushButton {
        #     font-size: 14px;
        #     padding: 8px;
        # }
        # QPushButton:hover {
        #     background-color: #d0f0ff;
        # }
        # QComboBox {
        #     font-size: 14px;
        #     padding: 4px;
        # }
        # QLabel {
        #     font-weight: bold;
        # }
        # QGroupBox {
        #     font-size: 15px;
        #     font-weight: bold;
        #     border: 1px solid #cccccc;
        #     border-radius: 5px;
        #     margin-top: 10px;
        # }
        # QGroupBox::title {
        #     subcontrol-origin: margin;
        #     subcontrol-position: top left;
        #     padding: 0 5px;
        # }
        # QTextEdit {
        #     background-color: #f8f8f8;
        #     font-size: 16px;
        #     font-weight: bold;
        #     padding: 12px;
        # }
        # """

        self.setStyleSheet(style)
        self.video_button.clicked.connect(self.select_video)
        # 저장경로 선택 버튼
        self.btn_select_save_path.clicked.connect(self.select_save_path)
        # health check
        self.server_alive.clicked.connect(self.health_check)
        # 종료 버튼
        self.btn_exit.clicked.connect(self.close_app)
        # 라벨링시작 버튼
        self.start_button.clicked.connect(self.start_labeling)
        # 라디오 버튼 이벤트
        self.radio_single.toggled.connect(self.on_video_mode_changed)
        self.radio_folder.toggled.connect(self.on_video_mode_changed)

    def select_save_path(self):
        folder = QFileDialog.getExistingDirectory(self, "저장할 폴더 선택")
        if folder:
            self.save_path = folder
            self.btn_select_save_path.setText(f"[폴더] 저장 경로: {folder}")
        else:
            QMessageBox.information(self, "알림", "저장 경로를 선택하지 않았습니다.")

    def close_app(self):
        reply = QMessageBox.question(
            self, "종료 확인", "정말 종료하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.close()

    def on_video_mode_changed(self):
        """영상 모드 변경 시 UI 업데이트"""
        if self.radio_single.isChecked():
            self.video_button.setText("[영상] 비디오 선택")
        else:
            self.video_button.setText("[폴더] 영상 폴더 선택")
    
    def select_video(self):
        if self.radio_single.isChecked():
            # 단일 영상 모드
            video_path, _ = QFileDialog.getOpenFileName(self, "비디오 선택", "", "Video Files (*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV)")
            if video_path:
                print("선택한 비디오:", video_path)
                self.video_path = video_path
                base_name = os.path.basename(video_path)    
                self.video_button.setText(f"[영상] 선택한 비디오: {base_name}")
            else:
                self.video_path=None
                self.video_button.setText(f"[영상] 비디오 선택")
        else:
            # 폴더 모드
            folder_path = QFileDialog.getExistingDirectory(self, "영상 폴더 선택")
            if folder_path:
                print("선택한 영상 폴더:", folder_path)
                self.video_folder_path = folder_path
                folder_name = os.path.basename(folder_path)
                self.video_button.setText(f"[폴더] 선택한 폴더: {folder_name}")
                
                # 폴더 내 모든 영상 파일 찾기
                video_files = []
                extensions = ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV']
                for ext in extensions:
                    video_files.extend(glob.glob(os.path.join(folder_path, ext)))
                
                if video_files:
                    self.video_list = sorted(video_files)
                    self.log_text.setText(f"[폴더] 폴더 내 {len(self.video_list)}개 영상 발견\n영상을 시작하면 미완료 영상만 표시됩니다.")
                else:
                    QMessageBox.warning(self, "알림", "선택한 폴더에 영상 파일이 없습니다.")
                    self.video_folder_path = None
                    self.video_list = []
                    self.video_button.setText("[폴더] 영상 폴더 선택")
            else:
                self.video_folder_path = None
                self.video_list = []
                self.video_button.setText("[폴더] 영상 폴더 선택")

    def set_saveclass(self):
        self.save_class = {
            "Lying":0,
            "Farrow_Start":0,
            "Farrowing":0,
            "Farrow_End":0,
            "Womb":0,
            "Nothing":0,
            "Hand":0
        }
        self.savenum=0
        self.log_text.setText(
            "-------- 처리현황 -------\n"
            f"Message: 라벨링 작업 전\n"
            f" Stand/Lying : 0\n"
            f" Farrow_Start : 0\n"
            f" Farrowing : 0\n"
            f" Farrow_End : 0\n"
            f" Womb : 0\n"
            f" Nothing : 0\n"
            f" Total Save : 0\n"
        )

    def log(self, message):
        #self.log_text.append(message)
        msg = message["log"]
        save_type = message.get("type")
        save_path = message.get("img")
        save_txt = message.get("label")
        record_to_csv = message.get("record_to_csv", False)  # CSV 기록 요청
        video_path = message.get("video_path", self.video_path)  # 영상 경로 (저장 시점에 전달)

        # lying_count=self.save_class[save_type]
        # lying_count=self.save_class["Lying"]

        if save_path is not None:
            # 실제 저장이 완료된 경우
            self.save_class[save_type]+=1
            self.savenum+=1
            
            # [현황] CSV 기록 (요청된 경우만, 중복 체크 포함)
            if record_to_csv and self.radio_folder.isChecked() and video_path:
                video_name = os.path.basename(video_path)
                # 중복 체크: 이미 CSV에 있으면 기록 안함
                completed_videos = self.read_completed_videos()
                if video_name not in completed_videos:
                    print("[알림] 첫 저장 완료 - CSV에 기록합니다")
                    self.save_completed_video(video_name)
                else:
                    print(f"[INFO]  이미 CSV에 기록된 영상: {video_name}")

            lying_count = self.save_class["Lying"]
            fs_count = self.save_class["Farrow_Start"]
            f_count = self.save_class["Farrowing"]
            fe_count = self.save_class["Farrow_End"]
            w_count = self.save_class["Womb"]
            n_count = self.save_class["Nothing"]

            self.log_text.setText(
                "-------- 처리현황 -------\n"
                f"Message : {msg}\n"
                f"이미지 저장 : {save_path}\n"
                f"라벨 저장 : {save_txt}\n"
                f" Stand/Lying : {lying_count}\n"
                f" Farrow_Start : {fs_count}\n"
                f" Farrowing : {f_count}\n"
                f" Farrow_End : {fe_count}\n"
                f" Womb : {w_count}\n"
                f" Nothing : {n_count}\n"
                f" Total Save : {self.savenum}"
            )
        else:
            lying_count = self.save_class["Lying"]
            fs_count = self.save_class["Farrow_Start"]
            f_count = self.save_class["Farrowing"]
            fe_count = self.save_class["Farrow_End"]
            w_count = self.save_class["Womb"]
            n_count = self.save_class["Nothing"]

            self.log_text.setText(
                "-------- 처리현황 -------\n"
                f"Message : {msg}\n"
                f" Stand/Lying : {lying_count}\n"
                f" Farrow_Start : {fs_count}\n"
                f" Farrowing : {f_count}\n"
                f" Farrow_End : {fe_count}\n"
                f" Womb : {w_count}\n"
                f" Nothing : {n_count}\n"
                f" Total Save : {self.savenum}"
            )

    # def on_server_input_changed(self, text):
    #     if text == "직접 입력":
    #         self.server_input_custom.show()
    #     else:
    #         self.server_input_custom.hide()
    def on_server_input_changed(self, text):
        if text == "직접 입력":
            self.server_input_custom.show()
        else:
            self.server_input_custom.hide()
        
        # 서버 선택이 바뀌면 연결 상태 초기화
        self.healthcheck = False
        self.server_alive.setText("서버 연결 확인")
        
        # 모델과 버전 선택 초기화
        self.models_info = {}  # 모델 정보 초기화
        self.model_select.clear()
        self.model_select.addItems(["서버 연결 후 모델을 불러오세요..."])
        self.version_select.clear()
        self.version_select.addItems(["모델을 먼저 선택하세요"])
        
        # 로그도 초기화 (선택사항)
        self.log_text.setText(
            "-------- 처리현황 -------\n"
            f"Message: 서버를 다시 선택하세요\n"
            f" Stand/Lying : 0\n"
            f" Farrow_Start : 0\n"
            f" Farrowing : 0\n"
            f" Farrow_End : 0\n"
            f" Womb : 0\n"
            f" Nothing : 0\n"
            f" Total Save : 0\n"
        )
    def health_check(self):

        if self.protocol.currentText() =="http":
            self.port = "8000"
        else:
            self.port = "8001"

        if self.server_input.currentText() =="inference_server":
            self.ip = "192.168.0.100"
        elif self.server_input.currentText() == "remote_server":
            self.ip = "example-server.local"
            if self.protocol.currentText() =="http":
                self.port = "8080"
            else:
                self.port = "8081"
        elif self.server_input.currentText() =="local":
            self.ip = "localhost"
        elif self.server_input.currentText() == "직접 입력":
            self.ip = self.server_input_custom.text().strip()
            if not self.ip:
                QMessageBox.warning(self, "입력 오류", "서버 주소를 입력하세요.")
                return
        else:
            self.ip = "localhost"


        # if self.server_input.currentText() =="http":
        #     self.port = "8000"
        # else:
        #     self.port = "8001"

        server_url = f"{self.ip}:{self.port}"   
        #print(f"Server : {server_url}")
        # if not self.video_path:
        #     QMessageBox.warning(self, "비디오 없음", "비디오를 선택하세요.")
        #     return
        
        # if self.save_path is None:
        #     QMessageBox.warning(self, "저장경로 없음", "저장경로를 선택하세요.")
        #     return
        
        if not server_url:
            QMessageBox.warning(self, "서버 주소 없음", "서버 주소를 입력하세요.")
            return

        # 임시로 기본 모델명 사용 (나중에 동적으로 변경됨)
        self.AI = Inference_server(server_url, "yolo_model_v11", log_callback=self.log)
        connect = self.AI.health_check()

        if connect:
            self.healthcheck=True
            QMessageBox.information(self, "server 체크", "연결 성공! 비디오를 선택하세요")
            self.server_alive.setText(f"추론서버: {server_url} 연결 성공!")
        else:
            self.healthcheck=False
            QMessageBox.critical(self, "server 체크", "연결 실패! AI팀에 문의하세요")
            self.server_alive.setText(f"추론서버: 연결 실패!")

        if self.healthcheck:
            # 모델 리스트 파싱 및 업데이트
            self.parse_and_update_models()
    
    def parse_and_update_models(self):
        """트리톤 서버 응답을 파싱하여 모델과 버전 정보를 업데이트합니다."""
        try:
            # 트리톤 서버에서 모델 리스트 가져오기 (실제 구현은 inference.py에서)
            model_data = self.AI.get_model_list()
            
            if model_data:
                # 모델 정보 파싱
                self.models_info = {}
                ready_models = []
                
                for model in model_data:
                    if model.get('state') == 'READY':
                        model_name = model.get('name')
                        version = int(model.get('version', 1))
                        
                        if model_name not in self.models_info:
                            self.models_info[model_name] = []
                        if version not in self.models_info[model_name]:  # 중복 방지
                            self.models_info[model_name].append(version)
                        if model_name not in ready_models:  # 중복 방지
                            ready_models.append(model_name)
                
                # 모델 선택 콤보박스 업데이트
                self.model_select.clear()
                self.model_select.addItems(ready_models)
                
                # 버전 선택 초기화
                self.version_select.clear()
                max_version = max(self.models_info[ready_models[0]])
                version_items = [f"버전 {max_version}"]
                self.version_select.clear()
                self.version_select.addItems(version_items)
                #self.version_select.addItems(["모델을 먼저 선택하세요"])
                self.version_select.setCurrentIndex(0)
                QMessageBox.information(self, "모델 로드", f"연결 성공! {len(ready_models)}개 모델을 찾았습니다.")
            else:
                QMessageBox.warning(self, "모델 로드", "연결 성공했지만 모델을 찾을 수 없습니다.")
                
        except Exception as e:
            QMessageBox.critical(self, "모델 로드 오류", f"모델 정보를 가져오는 중 오류가 발생했습니다: {str(e)}")
    
    def on_model_changed(self, model_name):
        """모델 선택이 바뀔 때 버전 목록을 업데이트합니다."""
        if model_name in self.models_info:
            # 해당 모델의 최대 버전 찾기
            max_version = max(self.models_info[model_name])
            
            # 1부터 최대 버전까지의 모든 버전 생성
            all_versions = list(range(1, max_version + 1))
            version_items = [f"버전 {max_version}"]# [f"버전 {max_version}" for v in all_versions]
            
            self.version_select.clear()
            self.version_select.addItems(version_items)
            #self.version_select.setCurrentIndex(max_version)
            # # 최신 버전(최대 버전) 자동 선택
            if all_versions:
                self.version_select.setCurrentIndex(0)  # 마지막(최신) 버전 선택
        else:
            self.version_select.clear()
            self.version_select.addItems(["모델을 먼저 선택하세요"])
    
    def get_progress_csv_path(self):
        """진행상황 CSV 파일 경로 반환"""
        if not self.save_path:
            print("X save_path (저장 경로)가 설정되지 않았습니다.")
            return None
        
        savedir = self.farm_input.text()
        if savedir:
            # 농장코드가 있으면 서브폴더에 저장
            csv_path = os.path.join(self.save_path, savedir, "progress_list.csv")
        else:
            # 농장코드가 없으면 바로 저장 경로에 저장
            csv_path = os.path.join(self.save_path, "progress_list.csv")
        return csv_path
    
    def read_completed_videos(self):
        """진행상황 CSV에서 완료된 영상 목록 읽기"""
        csv_path = self.get_progress_csv_path()
        print(f"📖 CSV 경로: {csv_path}")
        if not csv_path:
            print("X CSV 경로가 None입니다")
            return []
        if not os.path.isfile(csv_path):
            print(f"X CSV 파일이 존재하지 않습니다: {csv_path}")
            return []
        
        print(f"[OK] CSV 파일 존재 확인: {csv_path}")
        completed = []
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 건너뛰기
                for row in reader:
                    if row:
                        completed.append(row[0])  # 첫 번째 컬럼이 영상명
            print(f"[OK] 완료된 영상 {len(completed)}개 읽음: {completed}")
        except Exception as e:
            print(f"X 진행상황 파일 읽기 실패: {e}")
        return completed
    
    def save_completed_video(self, video_name):
        """완료된 영상을 CSV에 추가"""
        csv_path = self.get_progress_csv_path()
        if not csv_path:
            print("X CSV 경로가 없습니다. save_path 또는 farm_input이 설정되지 않았습니다.")
            return False
        
        print(f"[OK] CSV 저장 경로: {csv_path}")
        print(f"[OK] 저장할 영상명: {video_name}")
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        try:
            # 파일이 없으면 헤더와 함께 생성
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['video_name'])  # 헤더
                writer.writerow([video_name])
            print(f"[OK] 진행상황 파일 저장 성공!")
            return True
        except Exception as e:
            print(f"X 진행상황 파일 저장 실패: {e}")
            return False
    
    def filter_incomplete_videos(self):
        """폴더 모드: 미완료 영상만 필터링"""
        if not self.video_list:
            return []
        
        completed_videos = self.read_completed_videos()
        # completed_videos는 이미 파일명만 저장됨
        
        incomplete = []
        for video_path in self.video_list:
            video_name = os.path.basename(video_path)
            if video_name not in completed_videos:
                incomplete.append(video_path)
            else:
                print(f">>  완료된 영상 제외: {video_name}")
        
        print(f"📋 필터링 결과: {len(incomplete)}개 영상 남음 (전체 {len(self.video_list)}개 중)")
        return incomplete
            
    def start_labeling(self):
        savedir = self.farm_input.text()

        label_path = f"{self.save_path}/{savedir}"
        
        if self.save_path is None:
            QMessageBox.warning(self, "저장경로 없음", "저장경로를 선택하세요.")
            return
        
        if not self.healthcheck:
            QMessageBox.warning(self, "추론서버 연결 없음", "추론서버 연결상태를 확인하세요.")
            return
        
        # 선택된 모델과 버전 정보 가져오기
        selected_model = self.model_select.currentText()
        selected_version_text = self.version_select.currentText()
        
        if selected_version_text.startswith("버전 "):
            selected_version = int(selected_version_text.replace("버전 ", ""))
        else:
            selected_version = 1  # 기본값
        
        print(f"선택된 모델: {selected_model}, 버전: {selected_version}")
        
        # 선택된 모델로 AI 인스턴스 업데이트
        self.AI.modelname = selected_model
        
        # 영상 모드에 따라 분기
        if self.radio_folder.isChecked():
            # 폴더 모드: 미완료 영상만 필터링
            incomplete_videos = self.filter_incomplete_videos()
            if not incomplete_videos:
                QMessageBox.information(self, "알림", "처리할 영상이 없습니다. 모든 영상이 완료되었습니다.")
                return
            
            # 진행도 업데이트
            total = len(self.video_list)
            completed = len(self.video_list) - len(incomplete_videos)
            remaining = len(incomplete_videos)
            self.progress_label.setText(f"전체: {total} | 처리 완료: {completed} | 남음: {remaining}")
            
            # 영상 순회 처리
            self.process_videos_folder_mode(incomplete_videos, label_path, selected_model, selected_version)
        else:
            # 단일 영상 모드
            if not self.video_path:
                QMessageBox.warning(self, "비디오 없음", "비디오를 선택하세요.")
                return
            
            print(f"비디오 경로: {self.video_path}")
            self.set_saveclass()
            self.AI.start_label(self.video_path, label_path, model_name=selected_model, version=selected_version)
    
    def process_videos_folder_mode(self, video_list, label_path, selected_model, selected_version):
        """폴더 모드: 영상 순회 처리"""
        total = len(self.video_list)
        # CSV에 이미 저장된 완료 영상 수
        completed = len(self.video_list) - len(video_list)
        
        for idx, video_path in enumerate(video_list):
            self.video_path = video_path
            
            current_video_idx = idx + 1
            print(f"\n=== 영상 {current_video_idx}/{len(video_list)}: {os.path.basename(video_path)} ===")
            
            # 진행도 업데이트 (현재 작업 중인 영상 포함)
            remaining = len(video_list) - idx  # 남은 영상 수 (현재 영상 포함)
            self.progress_label.setText(f"전체: {total} | 처리 완료: {completed} | 남음: {remaining}")
            
            self.set_saveclass()
            result = self.AI.start_label(video_path, label_path, model_name=selected_model, version=selected_version)
            
            # 결과 처리
            if isinstance(result, tuple):
                status, saved_count = result
            else:
                status = result
                saved_count = 0
            
            if status == 'quit':
                # q 누름: 완전 종료
                # 어노테이션이 저장되었으면 완료 카운트 올림
                if saved_count > 0:
                    completed += 1
                    print(f"사용자가 종료했습니다. (저장된 어노테이션: {saved_count}개)")
                else:
                    print("사용자가 종료했습니다.")
                return
            elif status == 'next':
                # 0 누름: 다음 영상으로
                # 어노테이션이 저장되었으면 완료 카운트 올림
                if saved_count > 0:
                    completed += 1
                continue
            elif status is None:
                # 예기치 못한 오류 (영상 읽기 실패 등)
                # 어노테이션이 저장되었으면 완료 카운트 올림
                if saved_count > 0:
                    completed += 1
                    print(f"⚠️  영상 처리 중 오류 발생: {os.path.basename(video_path)} (저장된 어노테이션: {saved_count}개)")
                else:
                    print(f"⚠️  영상 처리 중 오류 발생: {os.path.basename(video_path)}")
                continue
        
        # 모든 영상 완료
        QMessageBox.information(self, "완료", f"모든 영상 처리가 완료되었습니다!\n총 {total}개 영상 처리 완료")
        self.progress_label.setText(f"전체: {total} | 처리 완료: {total} | 남음: 0")
        
    def play_video(self, video_path):
        basepath = f"{self.save_path}/{self.video_path}"
        self.AI.start_label(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "비디오 오류", "비디오를 열 수 없습니다.")
            return

        cv2.namedWindow("Video Preview", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Video Preview", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelingApp()
    window.show()
    sys.exit(app.exec_())