# -*- coding: UTF-8 -*-
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import random
import sys
import threading
import time

import cv2
import numpy
import torch
import torch.backends.cudnn as cudnn
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QLabel

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size
from utils.torch_utils import select_device
from detector import Detector
from v2xanalysix import HighwayAnalyse
import tracker
from deep_sort.tools.tools.highway_detection import Tracker
model_path = './weights/car_weight1.pt'
# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
	# 基本配置不动，然后只动第三个界面
	def __init__(self):
		# 初始化界面
		super().__init__()
		self.setWindowTitle('基于YOLOv8的智能交通监控系统')
		self.resize(1200, 800)
		self.setWindowIcon(QIcon("./UI/xf.jpg"))
		# 图片读取进程
		self.output_size = 480
		self.img2predict = ""
		# 空字符串会自己进行选择，首选cuda
		self.device = ''
		# # 初始化视频读取线程
		self.vid_source = '0'  # 初始设置为摄像头
		# 检测视频的线程
		self.threading = None
		# 是否跳出当前循环的线程
		self.jump_threading: bool = False

		self.image_size = 640
		self.confidence = 0.25
		self.iou_threshold = 0.45
		# 指明模型加载的位置的设备
		self.model = self.model_load(weights=model_path,
		                             device=self.device)
		self.initUI()
		self.reset_vid()


	@torch.no_grad()
	def model_load(self,
	               weights="",  # model.pt path(s)
	               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
	               ):
		"""
		模型初始化
		"""
		device = self.device = select_device(device)
		half = device.type != 'cpu'  # half precision only supported on CUDA

		# Load model
		model = attempt_load(weights, device)  # load FP32 model
		self.stride = int(model.stride.max())  # model stride
		self.image_size = check_img_size(self.image_size, s=self.stride)  # check img_size
		if half:
			model.half()  # to FP16
		# Run inference
		if device.type != 'cpu':
			print("Run inference")
			model(torch.zeros(1, 3, self.image_size, self.image_size).to(device).type_as(
				next(model.parameters())))  # run once
		print("模型加载完成!")
		return model

	def reset_vid(self):
		"""
		界面重置事件
		"""
		self.webcam_detection_btn.setEnabled(True)
		self.mp4_detection_btn.setEnabled(True)
		self.left_vid_img.setPixmap(QPixmap("./UI/up.jpeg"))
		self.vid_source = '0'
		self.disable_btn(self.det_img_button)
		self.disable_btn(self.vid_start_stop_btn)
		self.jump_threading = False

	def initUI(self):
		"""
		界面初始化
		"""
		# 图片检测子界面
		font_title = QFont('楷体', 16)
		font_main = QFont('楷体', 14)
		font_general = QFont('楷体', 10)
		# 图片识别界面, 两个按钮，上传图片和显示结果
		img_detection_widget = QWidget()
		img_detection_layout = QVBoxLayout()
		img_detection_title = QLabel("图片识别功能")
		img_detection_title.setFont(font_title)
		mid_img_widget = QWidget()
		mid_img_layout = QHBoxLayout()
		self.left_img = QLabel()
		self.right_img = QLabel()
		self.left_img.setPixmap(QPixmap("./UI/up.jpeg"))
		self.right_img.setPixmap(QPixmap("./UI/right.jpeg"))
		self.left_img.setAlignment(Qt.AlignCenter)
		self.right_img.setAlignment(Qt.AlignCenter)
		self.left_img.setMinimumSize(480, 480)
		self.left_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
		mid_img_layout.addWidget(self.left_img)
		self.right_img.setMinimumSize(480, 480)
		self.right_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
		mid_img_layout.addStretch(0)
		mid_img_layout.addWidget(self.right_img)
		mid_img_widget.setLayout(mid_img_layout)
		self.up_img_button = QPushButton("上传图片")
		self.det_img_button = QPushButton("开始检测")
		self.up_img_button.clicked.connect(self.upload_img)
		self.det_img_button.clicked.connect(self.detect_img)
		self.up_img_button.setFont(font_main)
		self.det_img_button.setFont(font_main)
		self.up_img_button.setStyleSheet("QPushButton{color:white}"
		                                 "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                 "QPushButton{background-color:rgb(48,124,208)}"
		                                 "QPushButton{border:2px}"
		                                 "QPushButton{border-radius:5px}"
		                                 "QPushButton{padding:5px 5px}"
		                                 "QPushButton{margin:5px 5px}")
		self.det_img_button.setStyleSheet("QPushButton{color:white}"
		                                  "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                  "QPushButton{background-color:rgb(48,124,208)}"
		                                  "QPushButton{border:2px}"
		                                  "QPushButton{border-radius:5px}"
		                                  "QPushButton{padding:5px 5px}"
		                                  "QPushButton{margin:5px 5px}")
		img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
		img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
		img_detection_layout.addWidget(self.up_img_button)
		img_detection_layout.addWidget(self.det_img_button)
		img_detection_widget.setLayout(img_detection_layout)

		# 视频识别界面
		# 视频识别界面的逻辑比较简单，基本就从上到下的逻辑
		vid_detection_widget = QWidget()
		vid_detection_layout = QVBoxLayout()
		vid_title = QLabel("视频检测功能")
		vid_title.setFont(font_title)
		self.left_vid_img = QLabel()
		self.right_vid_img = QLabel()
		self.left_vid_img.setPixmap(QPixmap("./UI/up.jpeg"))
		self.right_vid_img.setPixmap(QPixmap("./UI/right.jpeg"))
		self.left_vid_img.setAlignment(Qt.AlignCenter)
		self.left_vid_img.setMinimumSize(480, 480)
		self.left_vid_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
		self.right_vid_img.setAlignment(Qt.AlignCenter)
		self.right_vid_img.setMinimumSize(480, 480)
		self.right_vid_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
		mid_img_widget = QWidget()
		mid_img_layout = QHBoxLayout()
		mid_img_layout.addWidget(self.left_vid_img)
		mid_img_layout.addStretch(0)
		mid_img_layout.addWidget(self.right_vid_img)
		mid_img_widget.setLayout(mid_img_layout)
		self.webcam_detection_btn = QPushButton("摄像头实时监测")
		self.mp4_detection_btn = QPushButton("视频文件检测")
		self.vid_start_stop_btn = QPushButton("启动/停止检测")
		self.webcam_detection_btn.setFont(font_main)
		self.mp4_detection_btn.setFont(font_main)
		self.vid_start_stop_btn.setFont(font_main)
		self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
		                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                        "QPushButton{background-color:rgb(48,124,208)}"
		                                        "QPushButton{border:2px}"
		                                        "QPushButton{border-radius:5px}"
		                                        "QPushButton{padding:5px 5px}"
		                                        "QPushButton{margin:5px 5px}")
		self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
		                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                     "QPushButton{background-color:rgb(48,124,208)}"
		                                     "QPushButton{border:1px}"
		                                     "QPushButton{border-radius:5px}"
		                                     "QPushButton{padding:5px 5px}"
		                                     "QPushButton{margin:5px 5px}")
		self.vid_start_stop_btn.setStyleSheet("QPushButton{color:white}"
		                                      "QPushButton:hover{background-color: rgb(2,110,180);}"
		                                      "QPushButton{background-color:rgb(48,124,208)}"
		                                      "QPushButton{border:2px}"
		                                      "QPushButton{border-radius:5px}"
		                                      "QPushButton{padding:5px 5px}"
		                                      "QPushButton{margin:5px 5px}")
		self.webcam_detection_btn.clicked.connect(self.open_cam)
		self.mp4_detection_btn.clicked.connect(self.open_mp4)
		self.vid_start_stop_btn.clicked.connect(self.start_or_stop)

		# 添加fps显示
		fps_container = QWidget()
		fps_container.setStyleSheet("QWidget{background-color: #f6f8fa;}")
		fps_container_layout = QHBoxLayout()
		fps_container.setLayout(fps_container_layout)
		# 左容器
		fps_left_container = QWidget()
		fps_left_container.setStyleSheet("QWidget{background-color: #f6f8fa;}")
		fps_left_container_layout = QHBoxLayout()
		fps_left_container.setLayout(fps_left_container_layout)

		# 右容器
		fps_right_container = QWidget()
		fps_right_container.setStyleSheet("QWidget{background-color: #f6f8fa;}")
		fps_right_container_layout = QHBoxLayout()
		fps_right_container.setLayout(fps_right_container_layout)

		# 将左容器和右容器添加到fps_container_layout中
		fps_container_layout.addWidget(fps_left_container)
		fps_container_layout.addStretch(0)
		fps_container_layout.addWidget(fps_right_container)

		# 左容器中添加fps显示
		raw_fps_label = QLabel("原始帧率:")
		raw_fps_label.setFont(font_general)
		raw_fps_label.setAlignment(Qt.AlignLeft)
		raw_fps_label.setStyleSheet("QLabel{margin-left:80px}")
		self.raw_fps_value = QLabel("0")
		self.raw_fps_value.setFont(font_general)
		self.raw_fps_value.setAlignment(Qt.AlignLeft)
		fps_left_container_layout.addWidget(raw_fps_label)
		fps_left_container_layout.addWidget(self.raw_fps_value)

		# 右容器中添加fps显示
		detect_fps_label = QLabel("检测帧率:")
		detect_fps_label.setFont(font_general)
		detect_fps_label.setAlignment(Qt.AlignRight)
		self.detect_fps_value = QLabel("0")
		self.detect_fps_value.setFont(font_general)
		self.detect_fps_value.setAlignment(Qt.AlignRight)
		self.detect_fps_value.setStyleSheet("QLabel{margin-right:96px}")
		fps_right_container_layout.addWidget(detect_fps_label)
		fps_right_container_layout.addWidget(self.detect_fps_value)

		# 添加组件到布局上
		vid_detection_layout.addWidget(vid_title, alignment=Qt.AlignCenter)
		vid_detection_layout.addWidget(fps_container)
		vid_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
		vid_detection_layout.addWidget(self.webcam_detection_btn)
		vid_detection_layout.addWidget(self.mp4_detection_btn)
		vid_detection_layout.addWidget(self.vid_start_stop_btn)
		vid_detection_widget.setLayout(vid_detection_layout)

		# 关于界面
		about_widget = QWidget()
		about_layout = QVBoxLayout()
		about_title = QLabel('欢迎使用车辆检测系统\n')  # 修改欢迎词语
		about_title.setFont(QFont('楷体', 18))
		about_title.setAlignment(Qt.AlignCenter)
		about_img = QLabel()
		about_img.setPixmap(QPixmap('./UI/qq.png'))
		about_img.setAlignment(Qt.AlignCenter)

		label_super = QLabel()  # 更换作者信息
		label_super.setText("1023010302-桑猛猛")
		label_super.setFont(QFont('楷体', 16))
		label_super.setOpenExternalLinks(True)
		# label_super.setOpenExternalLinks(True)
		label_super.setAlignment(Qt.AlignRight)
		about_layout.addWidget(about_title)
		about_layout.addStretch()
		about_layout.addWidget(about_img)
		about_layout.addStretch()
		about_layout.addWidget(label_super)
		about_widget.setLayout(about_layout)

		self.addTab(img_detection_widget, '图片检测')
		self.addTab(vid_detection_widget, '视频检测')
		self.setTabIcon(0, QIcon('./UI/lufei.png'))
		self.setTabIcon(1, QIcon('./UI/lufei.png'))

	def disable_btn(self, pushButton: QPushButton):
		pushButton.setDisabled(True)
		pushButton.setStyleSheet("QPushButton{background-color: rgb(2,110,180);}")

	def enable_btn(self, pushButton: QPushButton):
		pushButton.setEnabled(True)
		pushButton.setStyleSheet(
			"QPushButton{background-color: rgb(48,124,208);}"
			"QPushButton{color:white}"
		)

	def detect(self, source: str, left_img: QLabel, right_img: QLabel):
		"""
		@param source: file/dir/URL/glob, 0 for webcam
		@param left_img: 将左侧QLabel对象传入，用于显示图片
		@param right_img: 将右侧QLabel对象传入，用于显示图片
		"""
		model = self.model
		img_size = [self.image_size, self.image_size]  # inference size (pixels)
		conf_threshold = self.confidence  # confidence threshold
		iou_threshold = self.iou_threshold  # NMS IOU threshold
		device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
		classes = None # filter by class: --class 0, or --class 0 2 3
		agnostic_nms = False  # class-agnostic NMS
		augment = False  # augmented inference
		# 实例化1个追踪器
		tracker_l = Tracker()
		# 实例化分析类
		highway = HighwayAnalyse()
		hide_labels = False

		f = open("log/OverSpeed.txt", 'a',encoding='utf-8')
		f.write("------------------------------------------------------------------------")
		f.write("\n")
		f.write("开始记录，记录时间为{}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
		f.write("\n")
		f.close()

		half = device.type != 'cpu'  # half precision only supported on CUDA

		if source == "":
			self.disable_btn(self.det_img_button)
			QMessageBox.warning(self, "请上传", "请先上传视频或图片再进行检测")
		else:
			source = str(source)
			webcam = source.isnumeric()

			# Set Dataloader
			if webcam:
				cudnn.benchmark = True  # set True to speed up constant image size inference
				dataset = LoadStreams(source, img_size=img_size, stride=self.stride)
			else:
				dataset = LoadImages(source, img_size=img_size, stride=self.stride)
			# Get names and colors
			names = model.module.names if hasattr(model, 'module') else model.names
			colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

			# 用来记录处理的图片数量
			count = 0

			# 根据视频尺寸，填充第一个polygon，供撞线计算使用
			mask_image_temp = numpy.zeros((1080, 1920), dtype=numpy.uint8)
			# 初始化2个撞线polygon
			# 初始
			list_pts_blue = [[0, 450], [1920, 450], [1920, 460], [0, 460]]  # 两条竖线
			ndarray_pts_blue = numpy.array(list_pts_blue, numpy.int32)
			polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
			polygon_blue_value_1 = polygon_blue_value_1[:, :, numpy.newaxis]

			# 填充第二个polygon
			mask_image_temp = numpy.zeros((1080, 1920), dtype=numpy.uint8)
			list_pts_red = [[0, 550], [1920, 550], [1920, 560], [0, 560]]  # 两条竖线
			ndarray_pts_red = numpy.array(list_pts_red, numpy.int32)
			polygon_red_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_red], color=2)
			polygon_red_value_2 = polygon_red_value_2[:, :, numpy.newaxis]

			# 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
			polygon_mask_blue_and_red = polygon_blue_value_1 + polygon_red_value_2

			# 缩小尺寸，1920x1080->960x540
			polygon_mask_blue_and_red = cv2.resize(polygon_mask_blue_and_red, (960, 540))

			# 蓝 色盘 b,g,r
			blue_color_plate = [255, 0, 0]
			# 蓝 polygon图片
			blue_image = numpy.array(polygon_blue_value_1 * blue_color_plate, numpy.uint8)

			# 红 色盘
			red_color_plate = [0, 0, 255]
			# 红 polygon图片
			red_image = numpy.array(polygon_red_value_2 * red_color_plate, numpy.uint8)

			# 彩色图片（值范围 0-255）
			color_polygons_image = blue_image + red_image
			# 缩小尺寸，1920x1080->960x540
			color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

			# list 与蓝色polygon重叠
			list_overlapping_blue_polygon = []

			# list 与黄色polygon重叠
			list_overlapping_red_polygon = []

			# 进入数量
			down_count = 0
			# 离开数量
			up_count = 0

			car_count_up = 0
			bus_count_up = 0
			truck_count_up = 0
			others_count_up = 0
			car_count_down = 0
			bus_count_down = 0
			truck_count_down = 0
			others_count_down = 0



			font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
			draw_text_postion = (int(960 * 0.01), int(540 * 0.05))
			# 初始化 yolov8
			detector = Detector()
			# 计算帧率开始时间
			frame_index = -2

			for path, img, im0s, vid_cap in dataset:
				yolo_bboxs_l = []

				# 直接跳出for，结束线程
				if self.jump_threading:
					# 清除状态
					self.jump_threading = False
					break
				count += 1
				# 打开视频
				# 读取每帧图片
				# 缩小尺寸，1920x1080->960x540
				fps_start_time = time.time()

				if source == '0': im0s = im0s[0]
				im = cv2.resize(im0s, (960, 540))
				list_bboxs = []
				bboxes = detector.detect(im)


				# 如果画面中 有bbox
				if len(bboxes) > 0:
					list_bboxs = tracker.update(bboxes, im)
					# 画框
					output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
					pass
				else:
					# 如果画面中 没有bbox
					output_image_frame = im
				pass
				# 输出图片
				output_image_frame = cv2.add(output_image_frame, color_polygons_image)
				if len(list_bboxs) > 0:
					# ----------------------判断撞线----------------------
					for item_bbox in list_bboxs:
						x1, y1, x2, y2, label, track_id = item_bbox
						yolo_bboxs_l.append((x1, y1, x2, y2, label, track_id))
						# 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
						y1_offset = int(y1 + ((y2 - y1) * 0.6))
						# 撞线的点
						y = y1_offset
						x = x1
						if polygon_mask_blue_and_red[y, x] == 1:
							# 如果撞 蓝polygon
							if track_id not in list_overlapping_blue_polygon:
								list_overlapping_blue_polygon.append(track_id)
							pass
							# 判断 黄polygon list 里是否有此 track_id
							# 有此 track_id，则 认为是 外出方向
							if track_id in list_overlapping_red_polygon:
								# 外出+1
								up_count += 1
								if label == "car":
									car_count_up += 1
								elif label == "bus":
									bus_count_up += 1
								elif label == "truck":
									truck_count_up += 1
								else:
									others_count_up += 1
								# ledClose()
								# print(
								# 	f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_red_polygon}')
								# 删除 黄polygon list 中的此id
								list_overlapping_red_polygon.remove(track_id)
								pass
							else:
								# 无此 track_id，不做其他操作
								pass

						elif polygon_mask_blue_and_red[y, x] == 2:
							# 如果撞 黄polygon
							if track_id not in list_overlapping_red_polygon:
								list_overlapping_red_polygon.append(track_id)
							pass

							# 判断 蓝polygon list 里是否有此 track_id
							# 有此 track_id，则 认为是 进入方向
							if track_id in list_overlapping_blue_polygon:
								# 进入+1
								if label == "car":
									car_count_down += 1
								elif label == "bus":
									bus_count_down += 1
								elif label == "truck":
									truck_count_down += 1
								else:
									others_count_down += 1
								down_count += 1
								# ledOpen()

								# print(
								# 	f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')
								# 删除 蓝polygon list 中的此id
								list_overlapping_blue_polygon.remove(track_id)

								pass
							else:
								# 无此 track_id，不做其他操作
								pass
							pass
						else:
							pass
						pass
					sp = im.shape
					tracker_bboxs_l = tracker_l.update_tracker(output_image_frame, yolo_bboxs_l)
					if (frame_index % 15) == 0:
						highway.update_id_info(sp, tracker_bboxs_l, 'left')
					highway.plot_bboxes_1(output_image_frame, tracker_bboxs_l, 'left')
					frame_index += 1
					pass

					# ----------------------清除无用id----------------------
					list_overlapping_all = list_overlapping_red_polygon + list_overlapping_blue_polygon
					for id1 in list_overlapping_all:
						is_found = False
						for _, _, _, _, _, bbox_id in list_bboxs:
							if bbox_id == id1:
								is_found = True
								break
							pass
						pass

						if not is_found:
							# 如果没找到，删除id
							if id1 in list_overlapping_red_polygon:
								list_overlapping_red_polygon.remove(id1)
							pass
							if id1 in list_overlapping_blue_polygon:
								list_overlapping_blue_polygon.remove(id1)
							pass
						pass
					list_overlapping_all.clear()
					pass

					# 清空list
					list_bboxs.clear()

					pass
				else:
					# 如果图像中没有任何的bbox，则清空list
					list_overlapping_blue_polygon.clear()
					list_overlapping_red_polygon.clear()
					pass
				pass

				fps = int(1 / (time.time() - fps_start_time))

				text_draw = 'veh_type'
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01), int(540 * 0.05)),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)
				text_draw = 'come'
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01) + 200, int(540 * 0.05)),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)
				text_draw = 'go'
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01) + 300, int(540 * 0.05)),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)
				text_draw = 'car'
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01), int(540 * 0.05) + 25),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)

				text_draw = str(car_count_down) + '    ' + str(car_count_up)
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01) + 200, int(540 * 0.05) + 25),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)

				text_draw = 'bus'
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01), int(540 * 0.05) + 50),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)
				text_draw = str(bus_count_down) + '    ' + str(bus_count_up)
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01) + 200, int(540 * 0.05) + 50),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)
				text_draw = 'truck'
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01), int(540 * 0.05) + 75),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)
				text_draw = str(truck_count_down) + '    ' + str(truck_count_up)
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01) + 200, int(540 * 0.05) + 75),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)
				text_draw = 'Total'
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01), int(540 * 0.05) + 100),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)
				text_draw = str(down_count) + '    ' + str(up_count)
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01) + 200, int(540 * 0.05) + 100),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)
				text_draw = 'FPS: ' + str(fps) + 'pics/s'
				output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
												 org=(int(960 * 0.01), int(540 * 0.05) + 125),
												 fontFace=font_draw_number,
												 fontScale=1, color=(255, 255, 255), thickness=2)
				self.detect_fps_value.setText(str(fps))
				# cv2.imshow('im', im0s)
				img = self.resize_img(im0s)
				img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1],
							 QImage.Format_RGB888)
				left_img.setPixmap(QPixmap.fromImage(img))
				# cv2.imshow('output_image_frame', output_image_frame)
				img = self.resize_img(output_image_frame)
				img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1],
							 QImage.Format_RGB888)
				right_img.setPixmap(QPixmap.fromImage(img))



				if cv2.waitKey(1) == ord('q'):  # q to quit
					raise StopIteration

				pass
			pass
			# 使用完摄像头释放资源

			f = open("log/OverSpeed.txt", 'a',encoding='utf-8')
			f.write("结束记录，记录时间为{}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
			f.write("\n")
			f.write("------------------------------------------------------------------------")
			f.write("\n")
			f.close()
			if webcam:
				for cap in dataset.caps:
					cap.release()
			else:
				dataset.cap and dataset.cap.release()

	def resize_img(self, img):
		"""
		调整图片大小，方便用来显示
		@param img: 需要调整的图片
		"""
		resize_scale = min(self.output_size / img.shape[0], self.output_size / img.shape[1])
		img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img

	def upload_img(self):
		"""
		上传图片
		"""
		# 选择录像文件进行读取
		fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
		if fileName:
			self.img2predict = fileName
			# 将上传照片和进行检测做成互斥的
			self.enable_btn(self.det_img_button)
			self.disable_btn(self.up_img_button)
			# 进行左侧原图展示
			img = cv2.imread(fileName)
			# 应该调整一下图片的大小
			img = self.resize_img(img)
			img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format_RGB888)
			self.left_img.setPixmap(QPixmap.fromImage(img))
			# 上传图片之后右侧的图片重置
			self.right_img.setPixmap(QPixmap("./UI/right.jpeg"))

	def detect_img(self):
		"""
		检测图片
		"""
		# 重置跳出线程状态，防止其他位置使用的影响
		self.jump_threading = False
		self.detect(self.img2predict, self.left_img, self.right_img)
		# 将上传照片和进行检测做成互斥的
		self.enable_btn(self.up_img_button)
		self.disable_btn(self.det_img_button)

	def open_mp4(self):
		"""
		开启视频文件检测事件
		"""
		print("开启视频文件检测")
		fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
		if fileName:
			self.disable_btn(self.webcam_detection_btn)
			self.disable_btn(self.mp4_detection_btn)
			self.enable_btn(self.vid_start_stop_btn)
			# 生成读取视频对象
			cap = cv2.VideoCapture(fileName)
			# 获取视频的帧率
			fps = cap.get(cv2.CAP_PROP_FPS)
			# 显示原始视频帧率
			self.raw_fps_value.setText(str(fps))
			if cap.isOpened():
				# 读取一帧用来提前左侧展示
				ret, raw_img = cap.read()
				cap.release()
			else:
				QMessageBox.warning(self, "需要重新上传", "请重新选择视频文件")
				self.disable_btn(self.vid_start_stop_btn)
				self.enable_btn(self.webcam_detection_btn)
				self.enable_btn(self.mp4_detection_btn)
				return
			# 应该调整一下图片的大小
			img = self.resize_img(numpy.array(raw_img))
			img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format_RGB888)
			self.left_vid_img.setPixmap(QPixmap.fromImage(img))
			# 上传图片之后右侧的图片重置
			self.right_vid_img.setPixmap(QPixmap("./UI/right.jpeg"))
			self.vid_source = fileName
			self.jump_threading = False

	def open_cam(self):
		"""
		打开摄像头事件
		"""
		print("打开摄像头")
		self.disable_btn(self.webcam_detection_btn)
		self.disable_btn(self.mp4_detection_btn)
		self.enable_btn(self.vid_start_stop_btn)
		self.vid_source = "0"
		self.jump_threading = False
		# 生成读取视频对象
		cap = cv2.VideoCapture(0)
		# 获取视频的帧率
		fps = cap.get(cv2.CAP_PROP_FPS)
		# 显示原始视频帧率
		self.raw_fps_value.setText(str(fps))
		if cap.isOpened():
			# 读取一帧用来提前左侧展示
			ret, raw_img = cap.read()
			cap.release()
		else:
			QMessageBox.warning(self, "需要重新上传", "请重新选择视频文件")
			self.disable_btn(self.vid_start_stop_btn)
			self.enable_btn(self.webcam_detection_btn)
			self.enable_btn(self.mp4_detection_btn)
			return
		# 应该调整一下图片的大小
		img = self.resize_img(numpy.array(raw_img))
		img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format_RGB888)
		self.left_vid_img.setPixmap(QPixmap.fromImage(img))
		# 上传图片之后右侧的图片重置
		self.right_vid_img.setPixmap(QPixmap("./UI/right.jpeg"))

	def start_or_stop(self):
		"""
		启动或者停止事件
		"""
		print("启动或者停止")
		if self.threading is None:
			# 创造并启动一个检测视频线程
			self.jump_threading = False
			self.threading = threading.Thread(target=self.detect_vid)
			self.threading.start()
			self.disable_btn(self.webcam_detection_btn)
			self.disable_btn(self.mp4_detection_btn)
		else:
			# 停止当前线程
			# 线程属性置空，恢复状态
			self.threading = None
			self.jump_threading = True
			self.enable_btn(self.webcam_detection_btn)
			self.enable_btn(self.mp4_detection_btn)

	def detect_vid(self):
		"""
		视频检测
		视频和摄像头的主函数是一样的，不过是传入的source不同罢了
		"""
		print("视频开始检测")
		self.detect(self.vid_source, self.left_vid_img, self.right_vid_img)
		print("视频检测结束")
		# 执行完进程，刷新一下和进程有关的状态，只有self.threading是None，
		# 才能说明是正常结束的线程，需要被刷新状态
		if self.threading is not None:
			self.start_or_stop()

	def closeEvent(self, event):
		"""
		界面关闭事件
		"""
		reply = QMessageBox.question(
			self,
			'quit',
			"Are you sure?",
			QMessageBox.Yes | QMessageBox.No,
			QMessageBox.No
		)
		if reply == QMessageBox.Yes:
			self.jump_threading = True
			self.close()
			event.accept()
		else:
			event.ignore()






if __name__ == "__main__":
	app = QApplication(sys.argv)
	mainWindow = MainWindow()
	mainWindow.show()
	sys.exit(app.exec_())
