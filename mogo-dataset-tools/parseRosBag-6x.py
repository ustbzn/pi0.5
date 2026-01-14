#!/usr/bin/python3
import sys
import os
import pickle
import json
import bisect
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import rosbag
import cv2
from cv_bridge import CvBridge

sys.path.append('/opt/ros/noetic/lib/python3/dist-packages/')
sys.path.append('/home/mogo/data/liqinbu/d6_ws/install/lib/python3/dist-packages/')
for path in sys.path:
    if 'install' in path and 'common' not in path:
        sys.path.append(path + '/common')
from header_pb2 import Header
from localization_pb2 import Localization
from trajectory_pb2 import Trajectory


# 定义camera view和topic
camera_view_topic_dict = {
    "/sensor/camera/sensing/image_raw_120/nvjpeg":"front",
    "/sensor/camera/sensing/image_raw_left/nvjpeg":"left",
    "/sensor/camera/sensing/image_raw_right/nvjpeg":"right",
    "/sensor/camera/sensing/image_raw_back/nvjpeg":"back"
}

# 计算header时间戳（秒）
def calculate_header_timestamp(header: Header) -> float:
    """Calculate ROS Header timestamp in seconds."""
    sec = float(header.stamp.sec)
    nsec = float(header.stamp.nsec) if hasattr(header.stamp, 'nsec') else 0.0
    return sec + nsec * 1.0e-9

class PklDatasetWriter:
    """
    负责累积样本并保存 pkl
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.samples: list = []

    def add_sample(self, sample: Dict[str, Any]):
        self.samples.append(sample)

    def save(self):
        if not self.samples:
            print("[WARN] 没有可保存的样本，跳过写入")
            return
        try:
            with open(self.output_path, 'wb') as f:
                pickle.dump(self.samples, f)
            print(f"[INFO] PKL写入完成：{self.output_path}（{len(self.samples)}条）")
        except Exception as e:
            print(f"[ERROR] PKL写入失败：{e}（路径：{self.output_path}）")
            raise

class BagDataHandler:
    """
    负责加载 rosbag 文件，并提供解析相机与定位数据的接口。

    使用方式示例：
        handler = BagDataHandler(bag_dir)
        handler.parse_cameras()        # 解析并缓存相机图像
        handler.parse_localization()   # 解析定位与时间戳

    解析完成后可直接使用：
        handler.master_bag
        handler.slave_103_bag
        handler.camera_cache
        handler.valid_timestamps
        handler.valid_locations
        handler.json_filename
        handler.pkl_writer
    """

    def __init__(self, bag_dir: str, work_dir: Optional[str] = None):
        self.bag_dir = bag_dir
        self.work_dir = work_dir or os.getcwd()

        # bag 文件路径
        self.bags_file_master: Optional[str] = None
        self.bags_file_103: Optional[str] = None
        self.bags_file_107: Optional[str] = None

        # 已打开的 bag
        self.master_bag: Optional[rosbag.Bag] = None
        self.slave_103_bag: Optional[rosbag.Bag] = None
        self.slave_107_bag: Optional[rosbag.Bag] = None

        # 相机相关
        self.camera_topic_dict: Dict[str, str] = {}
        self.camera_cache: Dict[str, Dict[str, list]] = {}

        # 定位相关
        self.valid_timestamps: List[float] = []
        self.valid_locations: List[Localization] = []

        # 输出文件
        self.json_filename: str = os.path.join(self.work_dir, "test_data.json")
        self.pkl_filename: str = os.path.join(self.work_dir, "test_data.pkl")
        self.pkl_writer = PklDatasetWriter(self.pkl_filename)

        self._check_and_find_bags()
        self._open_bags()

    # ---------------- internal helpers ----------------
    def _check_and_find_bags(self):
        """检查目录并查找 master/slave bag 文件。"""
        if not os.path.isdir(self.bag_dir):
            raise FileNotFoundError(f"目录{self.bag_dir}无效")

        for filename in os.listdir(self.bag_dir):
            if filename.endswith(".bag"):
                fp = os.path.join(self.bag_dir, filename)
                if "-rosmaster-" in filename:
                    self.bags_file_master = fp
                elif "-rosslave-103" in filename:
                    self.bags_file_103 = fp
                elif "-rosslave-107" in filename:
                    self.bags_file_107 = fp

        if not self.bags_file_master or not self.bags_file_103 or not self.bags_file_107:
            raise FileNotFoundError("未找到指定Bag文件")

        print(f"[INFO] Master Bag路径：{self.bags_file_master}")
        print(f"[INFO] Slave103 Bag路径：{self.bags_file_103}")
        print(f"[INFO] Slave107 Bag路径：{self.bags_file_107}")

    def _open_bags(self):
        """打开 master 与 slave bag。"""
        try:
            self.master_bag = rosbag.Bag(self.bags_file_master, "r")
        except Exception as e:
            raise RuntimeError(f"打开Master Bag失败：{e}")

        try:
            self.slave_103_bag = rosbag.Bag(self.bags_file_103, "r")
        except Exception as e:
            if self.master_bag is not None:
                self.master_bag.close()
            raise RuntimeError(f"打开Slave103 Bag失败：{e}")

        try:
            self.slave_107_bag = rosbag.Bag(self.bags_file_107, "r")
        except Exception as e:
            if self.master_bag is not None:
                self.master_bag.close()
            if self.slave_103_bag is not None:
                self.slave_103_bag.close()
            raise RuntimeError(f"打开Slave107 Bag失败：{e}")

    # ---------------- public APIs ----------------
    def parse_cameras(self, camera_rosbag: Optional[str]):
        """
        解析相机 topic，并将图片写入工作目录下的各相机子目录。

        结果保存在：
            self.camera_topic_dict
            self.camera_cache
        """
        assert camera_rosbag is not None 

        slave_topics = camera_rosbag.get_type_and_topic_info()[1].keys()
        print(f"[INFO] Slave Bag的topic：{slave_topics}")
        for topic in slave_topics:
            if topic in camera_view_topic_dict.keys():
                camera_view = camera_view_topic_dict[topic]
                self.camera_topic_dict[camera_view] = topic
                
        bridge = CvBridge()
        for key in self.camera_topic_dict.keys():
            for topic, msg, t in camera_rosbag.read_messages(self.camera_topic_dict[key]):
                if topic.endswith("nvjpeg"):
                    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                else:
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                camera_dir = os.path.join(self.bag_dir, key)
                if not os.path.exists(camera_dir):
                    os.makedirs(camera_dir, exist_ok=True)
                ts = msg.header.stamp.to_sec()
                timestr = "%.9f" % ts                
                camera_path = os.path.join(camera_dir, timestr + ".jpg")
                cv2.imwrite(camera_path, cv_image)
                if key not in self.camera_cache:
                    self.camera_cache[key] = {"times": [], "paths": []}
                self.camera_cache[key]["times"].append(ts)
                self.camera_cache[key]["paths"].append(camera_path)

        print(f"[INFO] 相机解析完成：共{len(self.camera_topic_dict)}路相机")

    def parse_localization(self):
        """
        解析定位数据并预处理时间戳。

        结果保存在：
            self.valid_timestamps
            self.valid_locations
        """
        assert self.master_bag is not None

        self.valid_timestamps = []
        self.valid_locations = []
        self.valid_loc_speed = []

        print("[INFO] 解析定位数据并预处理时间戳...")
        for topic, msg, t in self.master_bag.read_messages(topics=["/localization/global"]):
            try:
                loc_data = Localization()
                loc_data.ParseFromString(msg.data)
                ts = calculate_header_timestamp(loc_data.header)
                self.valid_timestamps.append(ts)
                self.valid_locations.append(loc_data)
                vx_global = getattr(loc_data, "longitudinal_v", 0.0)
                vy_global = getattr(loc_data, "lateral_v", 0.0)
                vz_global = getattr(loc_data, "vertical_v", 0.0)
                vel_xy = np.array([vx_global, vy_global], dtype=np.float64)
                speed_val = float(np.linalg.norm(vel_xy))
                self.valid_loc_speed.append(speed_val)

            except Exception as e:
                print(f"[WARN] 解析定位消息失败：{e}")

        if not self.valid_timestamps:
            raise RuntimeError("定位数据中无有效时间戳信息")
        print(f"[INFO] 定位数据处理完成：有效数据{len(self.valid_timestamps)}条")
    
    def align_camera_cache_by_front(self, max_delta: float = 0.05) -> Dict[float, Dict[str, str]]:
        """
        根据 front 时间戳对齐其他视角 camera 的图片路径。

        功能：
        - 以 front 相机的时间戳序列为基准；
        - 对每个 front 时间戳，在其他相机中寻找时间差 <= max_delta(秒) 的最近帧；
        - 将 front 及所有匹配到的其他视角组成一个 dict：{视角名: path}；
        - 存入新的容器：key 为 front 的时间戳，value 为该时间戳下所有视角的 path dict。

        返回：
            Dict[front_ts, {cam_key: path}]
        """
        if "front" not in self.camera_cache or not self.camera_cache["front"]["times"]:
            print("[WARN] align_camera_cache_by_front: front 相机数据为空")
            self.camera_group_by_front: Dict[float, Dict[str, str]] = {}
            return self.camera_group_by_front

        front_times: List[float] = self.camera_cache["front"]["times"]
        front_paths: List[str] = self.camera_cache["front"]["paths"]

        # 先确保每路相机的 times 与 paths 一一对应且按时间升序
        for key, data in self.camera_cache.items():
            times = data.get("times", [])
            paths = data.get("paths", [])
            if len(times) != len(paths):
                print(f"[WARN] 相机 {key} 的 times 与 paths 长度不一致，跳过排序")
                continue
            if not times:
                continue
            # 如果不是升序，则按时间重新排序
            if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
                paired = sorted(zip(times, paths), key=lambda x: x[0])
                data["times"] = [p[0] for p in paired]
                data["paths"] = [p[1] for p in paired]

        grouped: Dict[float, Dict[str, str]] = {}

        # 遍历 front 的时间轴，对其它相机做 max_delta 内的最近匹配
        for t_front, path_front in zip(front_times, front_paths):
            # 第一个一定是 front 自己
            paths_at_t: Dict[str, str] = {"front": path_front}

            for cam_key, data in self.camera_cache.items():
                if cam_key == "front":
                    continue
                times = data.get("times", [])
                paths = data.get("paths", [])
                if not times or len(times) != len(paths):
                    continue

                # 二分查找最接近 t_front 的索引
                idx = bisect.bisect_left(times, t_front)
                candidates = []
                if idx == 0:
                    candidates.append(0)
                elif idx == len(times):
                    candidates.append(len(times) - 1)
                else:
                    candidates.extend([idx - 1, idx])

                best_idx = None
                best_diff = float("inf")
                for ci in candidates:
                    diff = abs(times[ci] - t_front)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = ci

                if best_idx is not None and best_diff <= max_delta:
                    paths_at_t[cam_key] = paths[best_idx]
            # 确保paths_at_t中有所有的 camera，否则舍弃此帧所有图像（TODO）

            grouped[t_front] = paths_at_t

        self.camera_group_by_front: Dict[float, Dict[str, str]] = grouped
        print(f"[INFO] 对齐相机帧完成：front 时间戳 {len(grouped)} 个")
        return grouped

    def close(self):
        """关闭已打开的 bag。"""
        if self.master_bag is not None:
            self.master_bag.close()
            self.master_bag = None
        if self.slave_103_bag is not None:
            self.slave_103_bag.close()
            self.slave_103_bag = None

    def write_to_json(self):
        """
        按照 example.json 的格式，将对齐好的相机帧与定位速度写入 json 文件。
        规则：
            - token: 使用 front 相机时间戳（字符串形式，保留 9 位小数）
            - image: front 相机对应图片的路径
            - images: 当前时间戳下所有相机图片路径列表
            - 将与当前相机时间戳在 10ms(0.01s) 内匹配到的 speed_val 写入 conversations 中
        """
        # 前置检查
        if not hasattr(self, "camera_group_by_front") or not self.camera_group_by_front:
            print("[WARN] write_to_json: camera_group_by_front 为空，请先调用 align_camera_cache_by_front()")
            return
        if not getattr(self, "valid_timestamps", None) or not getattr(self, "valid_loc_speed", None):
            print("[WARN] write_to_json: 定位数据为空，请先调用 parse_localization()")
            return

        # 会话模板，与 example.json 中保持一致，其中 %.2f 为速度占位
        human_template = (
            "<FRONT VIEW>:\\n<image>\\n"
            "<FRONT LEFT VIEW>:\\n<image>\\n"
            "<FRONT RIGHT VIEW>:\\n<image>\\n"
            "<BACK LEFT VIEW>:\\n<image>\\n"
            "<BACK RIGHT VIEW>:\\n<image>\\n"
            "<BACK VIEW>:\\n<image>\\n "
            "Your current speed is %.2f m/s, the navigation command is 'go straight', "
            "based on the understanding of the driving scene and the navigation information, "
            "what is your plan for the next three seconds? "
            "Please answer your SPEED plan and your PATH plan. "
            "SPEED includes KEEP, ACCELERATE, DECELERATE, STOP. "
            "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. "
            "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
        )

        max_delta = 0.01  # 10ms

        json_list = []

        # 确保定位时间戳有序（通常已经按时间递增，但再保险）
        loc_times = self.valid_timestamps
        loc_speeds = self.valid_loc_speed
        if any(loc_times[i] > loc_times[i + 1] for i in range(len(loc_times) - 1)):
            paired = sorted(zip(loc_times, loc_speeds), key=lambda x: x[0])
            loc_times = [p[0] for p in paired]
            loc_speeds = [p[1] for p in paired]

        # 按时间排序遍历 front 帧
        for front_ts in sorted(self.camera_group_by_front.keys()):
            cam_paths_dict = self.camera_group_by_front[front_ts]

            # 匹配 10ms 内的 speed_val
            idx = bisect.bisect_left(loc_times, front_ts)
            candidates = []
            if idx == 0:
                candidates.append(0)
            elif idx == len(loc_times):
                candidates.append(len(loc_times) - 1)
            else:
                candidates.extend([idx - 1, idx])

            best_idx = None
            best_diff = float("inf")
            for ci in candidates:
                diff = abs(loc_times[ci] - front_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = ci

            # 若没有在 10ms 内匹配到定位，则跳过该帧
            if best_idx is None or best_diff > max_delta:
                continue

            speed_val = float(loc_speeds[best_idx])

            # 构造 json sample
            token = "%.9f" % front_ts
            front_image_path = cam_paths_dict.get("front", "")
            # images 中填写所有 camera 的 path
            images = list(cam_paths_dict.values())

            conversations = [
                {
                    "from": "human",
                    "value": human_template % speed_val,
                },
                {
                    "from": "gpt",
                    "value": "-, -\n",
                },
            ]

            sample = {
                "token": token,
                "image": front_image_path,
                "images": images,
                "prev": "",
                "next": "",
                "scene_token": "",
                "fut_valid_flag": "",
                "conversations": conversations,
                "ego_navi_cmd": "go straight",
                "id": "",
            }

            json_list.append(sample)

        # 写入 json 文件
        try:
            with open(self.json_filename, "w", encoding="utf-8") as f:
                json.dump(json_list, f, ensure_ascii=False, indent=2)
            print(f"[INFO] JSON 写入完成：{self.json_filename}（{len(json_list)} 条样本）")
        except Exception as e:
            print(f"[ERROR] 写入 JSON 失败：{e}")
            raise

    def run(self):
        # 主逻辑
        # 解析图片
        self.parse_cameras(self.slave_103_bag)
        self.parse_cameras(self.slave_107_bag)
        # 图片按时间戳拼帧
        self.align_camera_cache_by_front()
        # 解析定位topic
        self.parse_localization()
        # 保存到json
        self.write_to_json()
        # 关闭 bag
        self.close()

if __name__ == '__main__':
    print("[INFO] 脚本启动")
    try:
        bag_dir = sys.argv[1]
        print(f"[INFO] Bag目录：{bag_dir}")
    except IndexError:
        bag_dir = input("input bag directory: ")
        print(f"[INFO] Bag目录：{bag_dir}")
    try:
        pass
        handler = BagDataHandler(bag_dir)
        handler.run()
    except Exception as e:
        print(f"[ERROR] 初始化BagDataHandler失败：{e}")
        sys.exit(1)

