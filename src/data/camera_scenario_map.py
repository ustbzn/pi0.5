from enum import IntEnum

class ExpertType(IntEnum):
    MERGING = 0
    PARKING_EXIT = 1
    OVERTAKING = 2
    EMERGENCY_BRAKE = 3
    GIVEWAY = 4
    TRAFFIC_SIGN = 5
    NORMAL = 6

def get_scenario_id(scenario_name):
    mapping = {
        "MERGING": ExpertType.MERGING,
        "MERGING_HIGHWAY": ExpertType.MERGING,
        "MERGING_JUNCTION": ExpertType.MERGING,
        "PARKING_EXIT": ExpertType.PARKING_EXIT,
        "OVERTAKING": ExpertType.OVERTAKING,
        "EMERGENCY_BRAKE": ExpertType.EMERGENCY_BRAKE,
        "GIVEWAY": ExpertType.GIVEWAY,
        "GIVEWAY_HIGHWAY": ExpertType.GIVEWAY,
        "TRAFFIC_LIGHT": ExpertType.TRAFFIC_SIGN,
        "TRAFFIC_SIGN": ExpertType.TRAFFIC_SIGN,
        "NORMAL": ExpertType.NORMAL,
    }
    return mapping.get(scenario_name)


def get_cam_id(cam_name):
    cam_dict = {
        "NULL": 2,
        "CAM_FRONT_LEFT": 0,
        "CAM_FRONT_RIGHT": 1,
        "CAM_BACK": 2,
        "CAM_BACK_LEFT": 3,
        "CAM_BACK_RIGHT": 4, 
    }
    return cam_dict.get(cam_name)