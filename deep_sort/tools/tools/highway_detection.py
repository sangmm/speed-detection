import torch
import numpy as np
import cv2
from deep_sort.tools.utils.parser import get_config
from deep_sort.tools.deep_sort import DeepSort
from haversine import haversine


class Tracker:
    """
    deepsort追踪
    """
    def __init__(self):

        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)


    def update_tracker(self,image, yolo_bboxes):

        bbox_xywh = []
        confs = []
        clss = []

        for x1, y1, x2, y2, cls_id, conf in yolo_bboxes:

            obj = [
                int((x1+x2)/2), int((y1+y2)/2),
                x2-x1, y2-y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)
            clss.append(cls_id.split(" ")[0])


        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        #更新追踪结果
        outputs = self.deepsort.update_speed(xywhs, confss, clss, image)

        bboxes2draw = []
        for value in list(outputs):
            x1, y1, x2, y2, track_id = value
            # x1, y1, x2, y2, cls_, track_id = value
            bboxes2draw.append(
                (x1, y1, x2, y2,  track_id)
                # (x1, y1, x2, y2, cls_, track_id)
            )


        return bboxes2draw





class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """
    def __init__(self, pixel_array, lonlat_array):
        assert pixel_array.shape==(4,2), "Need (4,2) input array"
        assert lonlat_array.shape==(4,2), "Need (4,2) input array"
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array),np.float32(lonlat_array))
        self.invM = cv2.getPerspectiveTransform(np.float32(lonlat_array),np.float32(pixel_array))
        
    def pixel_to_lonlat(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1,2)
        assert pixel.shape[1]==2, "Need (N,2) input array" 
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0],1))], axis=1)
        lonlat = np.dot(self.M,pixel.T)
        
        return (lonlat[:2,:]/lonlat[2,:]).T
    
    def lonlat_to_pixel(self, lonlat):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1,2)
        assert lonlat.shape[1]==2, "Need (N,2) input array" 
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0],1))], axis=1)
        pixel = np.dot(self.invM,lonlat.T)
        
        return (pixel[:2,:]/pixel[2,:]).T


class SpeedEstimate:
    def __init__(self):

        # 配置相机画面与地图的映射点，需要根据自己镜头和地图上的点重新配置
        quad_coords = {
            "lonlat": np.array([
                [39.749918, 116.5162439], # top left
                [39.7499493,116.5162564], # top right
                [39.7500739,116.5156966], # bottom left
                [39.7500426,116.5156838] # bottom right
            ]),
            "pixel": np.array([
                [2417,8094],# top left
                [6258,8310], # top right
                [5320,1827], # bottom left
                [4305,1788] # bottom right
            ])
        }

        self.pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])

    def pixel2lonlat(self,x,y):
        # 像素坐标转为经纬度
        return self.pm.pixel_to_lonlat((x,y))[0]
    
    def pixelDistance(self,pa_x,pa_y,pb_x,pb_y):
        # 相机画面两点在地图上实际的距离

        lonlat_a = self.pm.pixel_to_lonlat((pa_x,pa_y))
        lonlat_b = self.pm.pixel_to_lonlat((pb_x,pb_y))
        
        lonlat_a = tuple(lonlat_a[0])
        lonlat_b = tuple(lonlat_b[0])
        
        return haversine(lonlat_a, lonlat_b, unit='m')


    

    
