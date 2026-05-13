import os
import pickle
from tqdm import tqdm
from collections import Counter

from src.data.camera_scenario_map import get_cam_id, get_scenario_id

SCENARIO_LIST = [
    "CrossingBicycleFlow",
    "EnterActorFlow",
    "HighwayExit",
    "InterurbanActorFlow",
    "HighwayCutIn",
    "InterurbanAdvancedActorFlow",
    "MergerIntoSlowTrafficV2",
    "MergerIntoSlowTraffic", 
    "NonSignalizedJunctionLeftTurn",
    "NonSignalizedJunctionRightTurn", 
    "NonSignalizedJunctionLeftTurnEnterFlow", 
    "LaneChange",
    "SignalizedJunctionLeftTurn", 
    "SignalizedJunctionRightTurn",
    "SignalizedJunctionLeftTurnEnterFlow",
    "ParkingExit",
    "Accident",
    "AccidentTwoWays",
    "ConstructionObstacle",
    "ConstructionObstacleTwoWays",
    "HazardAtSideLaneTwoWays",
    "HazardAtSideLane",
    "ParkedObstacleTwoWays",
    "ParkedObstacle",
    "VehicleOpensDoorTwoWays",
    "BlockedIntersection",
    "DynamicObjectCrossing",
    "HardBreakRoute",
    "OppositeVehicleTakingPriority", 
    "OppositeVehicleRunningRedLight", 
    "ParkingCutIn",
    "PedestrianCrossing",
    "ParkingCrossingPedestrian", 
    "StaticCutIn", 
    "VehicleTurningRoute",
    "VehicleTurningRoutePedestrian", 
    "ControlLoss",
    "InvadingTurn",
    "YieldToEmergencyVehicle",
    "TJunction",
    "VanillaSignalizedTurnEncounterGreenLight",
    "VanillaSignalizedTurnEncounterRedLight",
    "VanillaNonSignalizedTurn",
    "VanillaNonSignalizedTurnEncounterStopsign",
]

def check_cam_id(data_path):
    cam_id_counts = Counter()
    
    list_data_dict = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    for i in tqdm(range(len(list_data_dict))):
        pkl_data_path = os.path.join(data_path, list_data_dict[i])
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            cam_id = get_cam_id(all_data['his_camera_id'].numpy().decode('utf-8'))
            cam_id_counts[cam_id] += 1
    for cam_id, count in sorted(cam_id_counts.items()):
        print(f"ID {cam_id}: {count}")
        
def check_scenario_id(data_path):
    scenario_id_counts = Counter()
    
    list_data_dict = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    for i in tqdm(range(len(list_data_dict))):
        pkl_data_path = os.path.join(data_path, list_data_dict[i])
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            scenario_id = get_scenario_id(all_data['his_scenario_label'].numpy().decode('utf-8'))
            scenario_id_counts[scenario_id] += 1
    for scenario_id, count in sorted(scenario_id_counts.items()):
        print(f"ID {scenario_id}: {count}")

def count_scenario_files(scenario_list, data_path):
    all_files = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    sorted_scenarios = sorted(scenario_list, key=len, reverse=True)
    
    stats = {scenario: 0 for scenario in scenario_list}
    
    for file_name in tqdm(all_files):
        for scenario in sorted_scenarios:
            if file_name.startswith(scenario):
                stats[scenario] += 1
                break 

    total_count = 0
    for scenario, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{scenario:<45} | {count:<10}")
        total_count += count
    print("="*55)
    print(f"{'total':<45} | {total_count:<10}")
    

if __name__ == "__main__":
    check_cam_id("exp/b2d_action/train")
    check_scenario_id("exp/b2d_action/train")
    count_scenario_files(SCENARIO_LIST, "exp/b2d_action/train")

# Camera ID
# ID 0: 35105
# ID 1: 8865
# ID 2: 161717
# ID 3: 13671
# ID 4: 5990

# Scenario ID
# ID 0: 13304
# ID 1: 1036
# ID 2: 29921
# ID 3: 12064
# ID 4: 5200
# ID 5: 45332
# ID 6: 118491

# AccidentTwoWays                               | 12726     
# TJunction                                     | 11169     
# HardBreakRoute                                | 10752     
# CrossingBicycleFlow                           | 10308     
# HazardAtSideLane                              | 9207      
# ParkingCutIn                                  | 9120      
# VanillaSignalizedTurnEncounterRedLight        | 8791      
# ConstructionObstacleTwoWays                   | 8773      
# ParkedObstacleTwoWays                         | 7871      
# BlockedIntersection                           | 7218      
# VanillaSignalizedTurnEncounterGreenLight      | 7093      
# NonSignalizedJunctionLeftTurn                 | 7076      
# ParkingExit                                   | 6714      
# PedestrianCrossing                            | 6532      
# ParkedObstacle                                | 6383      
# Accident                                      | 6187      
# HazardAtSideLaneTwoWays                       | 6027      
# ParkingCrossingPedestrian                     | 5958      
# VehicleTurningRoute                           | 5406      
# DynamicObjectCrossing                         | 4889      
# ConstructionObstacle                          | 4845      
# StaticCutIn                                   | 4427      
# EnterActorFlow                                | 4274      
# HighwayExit                                   | 4161      
# OppositeVehicleTakingPriority                 | 4013      
# VehicleTurningRoutePedestrian                 | 3960      
# SignalizedJunctionLeftTurnEnterFlow           | 3796      
# VanillaNonSignalizedTurnEncounterStopsign     | 3690      
# HighwayCutIn                                  | 3402      
# YieldToEmergencyVehicle                       | 3247      
# OppositeVehicleRunningRedLight                | 3099      
# SignalizedJunctionLeftTurn                    | 3041      
# ControlLoss                                   | 3005      
# MergerIntoSlowTrafficV2                       | 2422      
# LaneChange                                    | 2326      
# SignalizedJunctionRightTurn                   | 2271      
# InvadingTurn                                  | 2106      
# NonSignalizedJunctionLeftTurnEnterFlow        | 2025      
# InterurbanAdvancedActorFlow                   | 1888      
# NonSignalizedJunctionRightTurn                | 1848      
# InterurbanActorFlow                           | 1720      
# MergerIntoSlowTraffic                         | 979       
# VehicleOpensDoorTwoWays                       | 603       
# VanillaNonSignalizedTurn                      | 0         
# =======================================================
# total                                         | 225348 