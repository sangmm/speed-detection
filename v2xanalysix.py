from deep_sort.tools.tools.highway_detection import SpeedEstimate
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyttsx3
import threading
from queue import Queue

engine = pyttsx3.init()

class HighwayAnalyse:
    def __init__(self):
        # 初始化消息队列
        self.alert_queue = Queue()
        # 启动后台线程处理警报
        threading.Thread(target=self.process_alert_queue, daemon=True).start()

        self.speed_esti = SpeedEstimate()
        # 记录左右车道上次位置、速度 { id :{'last_pos':(123,234),'speed':12.34}  , 2: ........} 
        self.left_ids_info = {} 
        self.right_ids_info = {}
        self.lock = threading.Lock()
        # 初始化语音引擎
        self.engine = pyttsx3.init()
        # 设置语音属性（可选）
        self.engine.setProperty('rate', 150)    #设置语速
        self.engine.setProperty('volume', 1.0)  #设置音量
        
        # 中文label图像
        #self.zh_label_img_list = self.getPngList()

        # 记录车辆数量
        self.vehicle_num = {'car':0,'truck':0}  
        # 汽车尾部
        self.track_tail_points = {'left':{},'right':{}}
        # 速度计量
        self.track_speeds = {'<90':0,'90-110':0,'>110':0}

    def getPngList(self):
        """
        获取PNG图像列表

        @return numpy array list
        """
        overlay_list = []
        # 遍历文件
        for i in range(2):
            fileName = './label_png/%s.png' % (i)
            overlay = cv2.imread(fileName)
            # overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            overlay_list.append(overlay)

        return overlay_list
    
    def plot_bboxes_1(self,image, bboxes,side='left'):
        """
        绘制，展示速度
        """
        alpha = 0.9
        this_ids_info = self.left_ids_info if side == 'left' else self.right_ids_info
        line_thickness = round(
            0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1
        font_thickness = max(line_thickness - 1, 1)
        overSpeedIds = []

        for (l, t, r, b, track_id) in bboxes:

            scale = (r- l)/40 * 0.6
            if track_id in this_ids_info and this_ids_info[track_id]['speed'] != 0:
                speed = round(this_ids_info[track_id]['speed'],1)
                 # 在图像上绘制车辆ID和速度
                cv2.putText(image, 'ID-{}'.format(track_id), (l + 55, t - 2), 0, line_thickness / 3,
                            [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)
                cv2.putText(image, '{}km/h'.format(speed), (l,t-30), cv2.FONT_ITALIC,scale,(0, 0, 255),2)
                if speed > 5:
                    # 绘制红色的警示框
                    cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), thickness=line_thickness)
                    # 在图像上显示警告文字
                    cv2.putText(image, 'Warning!!!', (l, t-60), cv2.FONT_ITALIC, scale, (0, 0, 255), 2)
                    self.trigger_alert(track_id,speed)
                    overSpeedIds.append(track_id)
                    f = open("log/OverSpeed.txt", 'a',encoding='utf-8')
                    f.write("超速车辆id: {} | 速度 {}".format(track_id, speed))
                    f.write("\n")
                    f.close()
        # 在图像上显示所有超速车辆的ID列表
        cv2.putText(img=image, text="OverSpeedIds: " + str(overSpeedIds),
                     org=(int(960 * 0.01), int(540 * 0.05) + 500),
                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                     fontScale=1, color=(255, 255, 255), thickness=2)

#定义警报器播放内容    
    #def trigger_alert(self, speed):
        #alert_message = f"警报！车速超过 {15} km/h."
        #engine.say(alert_message)
        #engine.runAndWait()
    

    

    def update_left_info(self, key, value):
        with self.lock:
            self.left_ids_info[key] = value

    def update_right_info(self, key, value):
        with self.lock:
            self.right_ids_info[key] = value
   
    def trigger_alert(self,track_id,speed):
        alert_message = f"警报！车辆ID-{track_id} 车速超过 {5} km/h."
        # 启动一个新线程来播放语音
        threading.Thread(target=self._play_alert, args=(alert_message,)).start()
        # 将警报消息添加到队列中
        self.alert_queue.put(alert_message)
    
    def _play_alert(self, message):
        with self.lock:
            if  self.engine.isBusy():
                self.engine.say(message)
                self.engine.runAndWait()

    def process_alert_queue(self):
        while True:
            # 从队列中获取警报消息并播放
            message = self.alert_queue.get()
            with self.lock:
                self.engine.say(message)
                self.engine.runAndWait()
            self.alert_queue.task_done()
            
   # 添加中文
    def cv2AddChineseText(self,img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "./fonts/MSYH.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)



    def update_id_info(self,shape, bboxes,side='left', ):
        """
        获取当前画面各ID的位置、车速信息
        """
        last_frame_info = self.left_ids_info if side == 'left' else self.right_ids_info
        # 本帧位置
        this_frame_info = {}
        # for (l, t, r, b, cls_name, track_id) in bboxes:
        for (l, t, r, b,  track_id) in bboxes:

            if side == 'left':
                # 尾部位置（左）
                head_pos = l+int((r-l)/2),t
            else:
                head_pos = l+int((r-l)/2),b
            # 初始化
            this_frame_info[track_id] = {'last_pos':head_pos,'speed':0}

        if len(last_frame_info) > 0:
            # 更新
            # 上：1、2、3、4
            # 本：3、4、5、6
            # 需要：更新3、4速度，插入 5、6 记录

            # 更新后的信息
            update_frame_info = {}
            update_num = 0
            insert_num  = 0

            for key,val in this_frame_info.items():
                
                if key in last_frame_info:
                    # 更新
                    # 本帧位置
                    this_frame_pos = val['last_pos']
                    scal = 1
                    # 上帧位置
                    last_frame_pos = last_frame_info[key]['last_pos']
                    # 计算距离
                    distance = self.speed_esti.pixelDistance(this_frame_pos[0], this_frame_pos[1], last_frame_pos[0], last_frame_pos[1])
                    # 速度
                    speed = distance * 0.00673 * 3.6 * 60 #1m/s=3.6km/h；6.73mm=0.00673m；两帧的时间差为1/60s

                    update_frame_info[key] = {'last_pos':this_frame_pos,'speed':speed}

                    update_num +=1
                else:
                    # 插入
                    # 本帧位置
                    this_frame_pos = val['last_pos']
                    update_frame_info[key] = {'last_pos':this_frame_pos,'speed':0}
                    insert_num +=1
            # print("刷新{}辆车信息，新增{}辆车位置信息".format(update_num,insert_num))
            # f = open("log/log.txt", 'a')
            # f.write("刷新{}辆车信息，新增{}辆车位置信息".format(update_num,insert_num))
            # f.write("\n")
            # f.close()
            last_frame_info = update_frame_info
        else:
            # 初始化
            last_frame_info = this_frame_info
            # print("新增：{}辆车位置信息".format(len(last_frame_info)))
            # f = open("log/log.txt", 'a')
            # f.write("新增：{}辆车位置信息".format(len(last_frame_info)))
            # f.write("\n")
            # f.close()
        # 重新赋值
        if side == 'left':
            self.left_ids_info = last_frame_info
            # print("===========",last_frame_info)
            # f = open("log/log.txt", 'a')
            # f.write((str)(last_frame_info))
            # f.write("\n")
            # f.close()


        else:
            self.right_ids_info = last_frame_info

