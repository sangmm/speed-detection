# -*- coding: UTF-8 -*-
import sys
import codecs
import random
import threading
import time
import cv2
import numpy
import torch
import torch.backends.cudnn as cudnn
import pyttsx3  # 用于语音输出
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size
from utils.torch_utils import select_device
from detector import Detector
from v2xanalysix import HighwayAnalyse
import tracker
from deep_sort.tools.tools.highway_detection import Tracker

# 模型路径
model_path = './weights/car_weight1.pt'

# 语音引擎初始化
engine = pyttsx3.init()

# 设置语音警报的内容
def voice_alert(alert_text):
    engine.say(alert_text)
    engine.runAndWait()

# 主窗口类
class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('基于YOLOv8的智能交通监控系统')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("./UI/xf.jpg"))
        self.output_size = 480
        self.img2predict = ""
        self.device = ''
        self.threading = None
        self.jump_threading = False
        self.image_size = 640
        self.confidence = 0.25
        self.iou_threshold = 0.45
        self.model = self.model_load(weights=model_path, device=self.device)
        self.initUI()
        self.reset_vid()

    @torch.no_grad()
    def model_load(self, weights="", device=''):
        device = self.device = select_device(device)
        half = device.type != 'cpu'
        model = attempt_load(weights, device)
        self.stride = int(model.stride.max())
        self.image_size = check_img_size(self.image_size, s=self.stride)
        if half:
            model.half()
        if device.type != 'cpu':
            model(torch.zeros(1, 3, self.image_size, self.image_size).to(device).type_as(next(model.parameters())))
        print("模型加载完成!")
        return model

    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.left_vid_img.setPixmap(QPixmap("./UI/up.jpeg"))
        self.vid_source = '0'
        self.disable_btn(self.det_img_button)
        self.disable_btn(self.vid_start_stop_btn)
        self.jump_threading = False

    def initUI(self):
        # 初始化界面逻辑（略）

        # 添加用于显示超速警告的 QLabel
        self.speed_warning_label = QLabel("")
        self.speed_warning_label.setFont(QFont('楷体', 16))
        self.speed_warning_label.setAlignment(Qt.AlignCenter)
        self.speed_warning_label.setStyleSheet("QLabel{color: red;}")

        # 将 QLabel 添加到界面布局中（略）

    def detect(self, source: str, left_img: QLabel, right_img: QLabel):
        # 检测逻辑（略）

        for path, img, im0s, vid_cap in dataset:
            # 检测框逻辑（略）

            # 超速判断
            for item_bbox in list_bboxs:
                # 设定某个阈值来判断是否超速，这里假设速度超过某值为超速
                speed = self.calculate_speed(item_bbox)
                if speed > speed_limit:
                    alert_text = f"车辆ID {item_bbox[-1]} 超速！当前速度：{speed} km/h"
                    self.speed_warning_label.setText(alert_text)
                    
                    # 启动语音警报
                    threading.Thread(target=voice_alert, args=(alert_text,)).start()

            # 其他逻辑（略）

    def calculate_speed(self, bbox):
        # 计算速度的逻辑（略）
        return speed

    # 其他方法（略）

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
