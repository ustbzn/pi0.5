# Data Directory

This directory should contain the Bench2Drive dataset and associated labels.

## Required Downloads

### 1. Bench2Drive Dataset
Download the [Bench2Drive dataset](https://huggingface.co/datasets/rethinklab/Bench2Drive) and place it in this directory.

### 2. Camera Labels
Download the [camera labels](https://huggingface.co/rethinklab/DriveMoE/tree/main/labels/camera_labels) into this directory.

We developed a set of heuristic rules based on annotation information from the Bench2Drive dataset to identify special driving scenarios, enabling effective camera-view-level supervision. We select camera views contextually, defaulting to the rear view if no critical view is identified.

#### Intersection Turning
When the ego-vehicle is required to turn at an intersection (i.e., `is_in_junction` is true and the current command is either "turn left" or "turn right"), we annotate the front-side camera view pointing toward the intended exit of the intersection.

#### Lane Change
When a lane change is required, identified by conditions such as:
- The current command being "change left" or "change right"
- An obstacle appearing within a certain distance ahead in the current lane
- The ego-vehicle not being in the target lane

The annotation depends on lane direction:
- **Same direction:** If the target lane is in the same direction as the ego-vehicle's current movement, we annotate the corresponding rear-side camera.
- **Opposing lane:** If the ego-vehicle must temporarily occupy the opposing lane, we annotate the corresponding front-side camera.

#### Highway Merging and Cut-in
In scenarios such as highway merging or vehicle cut-ins (scenario labeled as "merging" or "cut-in"), we determine the merging location based on the ego-vehicle's lane position and distance to the junction, annotating the side camera facing the merging location.

#### Yielding to Emergency Vehicles
If a high-speed emergency vehicle is present in the scenario, the ego-vehicle must yield, and we annotate the camera facing the direction of the approaching emergency vehicle.

### 3. Scenario Labels
Download the [scenario labels](https://huggingface.co/rethinklab/DriveMoE/tree/main/labels/scenario_labels) into this directory.

We define 7 scenario categories (0-6) to classify driving situations in the Bench2Drive dataset. These labels enable fine-grained analysis of ego-vehicle behavior under different conditions.

#### 0 - MERGING

**Description:** Scenarios requiring the ego-vehicle to merge into traffic flow, including highway merges, junction merges, and general merging situations.

**Original Scenarios:**
- `LaneChange`
- `HighwayExit`
- `InterurbanActorFlow`
- `MergerIntoSlowTraffic`
- `MergerIntoSlowTrafficV2`
- `CrossingBicycleFlow`
- `EnterActorFlow`
- `NonSignalizedJunctionLeftTurn`
- `NonSignalizedJunctionRightTurn`
- `NonSignalizedJunctionLeftTurnEnterFlow`
- `SignalizedJunctionLeftTurn`
- `SignalizedJunctionRightTurn`
- `SignalizedJunctionLeftTurnEnterFlow`

**Key Behaviors:** Lane changes, junction navigation, traffic flow integration

#### 1 - PARKING_EXIT

**Description:** Scenarios where the ego-vehicle exits from a parking lane or parking area.

**Original Scenarios:**
- `ParkingExit`

**Key Behaviors:** Leaving parking lane, merging from parking area

#### 2 - OVERTAKING

**Description:** Scenarios requiring overtaking obstacles, including accidents, construction, parked vehicles, and other static or dynamic obstacles.

**Original Scenarios:**
- `Accident`
- `AccidentTwoWays`
- `ConstructionObstacle`
- `ConstructionObstacleTwoWays`
- `HazardAtSideLane`
- `HazardAtSideLaneTwoWays`
- `ParkedObstacle`
- `ParkedObstacleTwoWays`
- `VehicleOpensDoor`
- `VehicleOpensDoorTwoWays`

**Key Behaviors:** Lane change to bypass obstacles, temporary lane occupancy

#### 3 - EMERGENCY_BRAKE

**Description:** Scenarios requiring immediate braking or emergency maneuvers due to sudden obstacles, crossing pedestrians/vehicles, or unexpected cut-ins.

**Original Scenarios:**
- `BlockedIntersection`
- `DynamicObjectCrossing`
- `PedestrianCrossing`
- `ParkingCrossingPedestrian`
- `VehicleTurningRoute`
- `VehicleTurningRoutePedestrian`
- `OppositeVehicleTakingPriority`
- `OppositeVehicleRunningRedLight`
- `HardBreakRoute`
- `ParkingCutIn`
- `StaticCutIn`
- `ControlLoss`

**Key Behaviors:** Emergency braking, collision avoidance, sudden deceleration

#### 4 - GIVEWAY

**Description:** Scenarios requiring yielding to other vehicles, emergency vehicles, or obstacles, including both urban and highway give-way situations.

**Original Scenarios:**
- `InvadingTurn`
- `YieldToEmergencyVehicle`
- `HighwayCutIn`
- `InterurbanAdvancedActorFlow`

**Key Behaviors:** Yielding, lane deviation for obstacle avoidance, giving priority to emergency vehicles

#### 5 - TRAFFIC_SIGN

**Description:** Scenarios where traffic signs (primarily stop signs) affect the ego-vehicle's behavior.

**Original Scenarios:**
- All scenarios with stop sign influence

**Key Behaviors:** Stopping at stop signs, right-of-way adherence

#### 6 - NORMAL

**Description:** Normal driving scenarios including regular navigation and traffic light interactions. Traffic light is included in normal as it represents standard driving behavior rather than special maneuvers.

**Original Scenarios:**
- All scenarios with traffic light influence
- Default normal driving scenarios

**Key Behaviors:** Regular driving, traffic light compliance, lane keeping

## Expected Directory Structure

After downloading all components, the directory structure should look like this:
```
├── data
│   ├── Bench2Drive-Base
│   ├── camera_labels
│   ├── scenario_labels
│   └── data.md
```

