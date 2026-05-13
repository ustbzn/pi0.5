# Evaluation

## Open-loop Evaluation
Open-loop evaluation validates the model's predictive capabilities under fixed historical conditions.

**Configuration**
1. Update the [drivepi config file](/config/eval/DrivePi0/open_loop.yaml) and [drivemoe config file](/config/eval/DriveMoE/open_loop.yaml)
2. **Important**: The model must be trained with **horizon=20** (predicting the next 20 trajectory points)

**Run Evaluation**
```bash
bash script/evaluation/open_loop_drivepi0.sh
bash script/evaluation/open_loop_drivemoe.sh
```

<table> <caption><strong>Table 1: Statistics of Dataset Distribution and Camera Router Performance.</strong></caption> <thead> <tr> <th rowspan="2">Camera Position</th> <th colspan="2">Training Set</th> <th colspan="4">Evaluation Results (Test Set)</th> </tr> <tr> <th>Samples</th> <th>Ratio (%)</th> <th>Precision</th> <th>Recall</th> <th>F1-score</th> <th>Support</th> </tr> </thead> <tbody> <tr><td>Front Left</td><td>35,105</td><td>15.58</td><td>0.83</td><td>0.89</td><td>0.86</td><td>1,300</td></tr> <tr><td>Front Right</td><td>8,865</td><td>3.93</td><td>0.79</td><td>0.81</td><td>0.80</td><td>666</td></tr> <tr><td>Back</td><td>161,717</td><td>71.76</td><td>0.95</td><td>0.92</td><td>0.93</td><td>8,996</td></tr> <tr><td>Back Left</td><td>13,671</td><td>6.07</td><td>0.77</td><td>0.72</td><td>0.75</td><td>1,120</td></tr> <tr><td>Back Right</td><td>5,990</td><td>2.66</td><td>0.39</td><td>0.90</td><td>0.54</td><td>226</td></tr> <tr style="border-top:2px solid #aaa;"><td><strong>Total / Macro Avg</strong></td><td><strong>225,348</strong></td><td><strong>100.00</strong></td><td><strong>0.75</strong></td><td><strong>0.85</strong></td><td><strong>0.78</strong></td><td><strong>12,308</strong></td></tr> <tr><td><strong>Overall Accuracy</strong></td><td>—</td><td>—</td><td colspan="3"><strong>0.89</strong></td><td>—</td></tr> </tbody> </table>

<table>
<caption><strong>Table 2: Statistics of Dataset Distribution and Action Router Performance.</strong></caption>
<thead>
  <tr>
    <th rowspan="2">Scenario</th>
    <th colspan="2">Training Set</th>
    <th colspan="4">Evaluation Results (Test Set)</th>
   </tr>
  <tr>
    <th>Samples</th>
    <th>Ratio (%)</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Support</th>
   </tr>
</thead>
<tbody>
  <tr><td>Merging</td><td>13,304</td><td>5.02</td><td>0.80</td><td>0.76</td><td>0.78</td><td>670</td></tr>
  <tr><td>Parking Exit</td><td>1,036</td><td>0.46</td><td>1.00</td><td>1.00</td><td>1.00</td><td>40</td></tr>
  <tr><td>Overtaking</td><td>29,921</td><td>13.28</td><td>0.87</td><td>0.91</td><td>0.89</td><td>1,377</td></tr>
  <tr><td>Emergency Brake</td><td>12,064</td><td>5.35</td><td>0.76</td><td>0.67</td><td>0.71</td><td>690</td></tr>
  <tr><td>Giveway</td><td>5,200</td><td>2.31</td><td>0.96</td><td>0.38</td><td>0.54</td><td>482</td></tr>
  <tr><td>Traffic Sign</td><td>45,332</td><td>20.12</td><td>0.75</td><td>0.96</td><td>0.84</td><td>2,045</td></tr>
  <tr><td>Normal</td><td>118,491</td><td>52.58</td><td>0.88</td><td>0.85</td><td>0.86</td><td>7,004</td></tr>
  <tr style="border-top:2px solid #aaa;">
    <td><strong>Total / Macro Avg</strong></td>
    <td><strong>225,348</strong></td>
    <td><strong>100.00</strong></td>
    <td><strong>0.86</strong></td>
    <td><strong>0.79</strong></td>
    <td><strong>0.80</strong></td>
    <td><strong>12,308</strong></td>
  </tr>
  <tr>
    <td><strong>Overall Accuracy</strong></td>
    <td>—</td>
    <td>—</td>
    <td colspan="3"><strong>0.84</strong></td>
    <td>—</td>
  </tr>
</tbody>
</table>

## Closed-loop evaluation
For closed-loop testing, configure the [drivepi config file](/config/eval/DrivePi0/closed_loop.yaml) and [drivemoe config file](/config/eval/DriveMoE/closed_loop.yaml) (update paths, etc.). You can download our DrivePi0 and DriveMoE checkpoint [here](https://huggingface.co/rethinklab/DriveMoE).

The following are our evaluation results on 8*H200 GPUs.

| Model | DS | SR | Json |
|:-------:|:-------:|:-------:|:-------:|
| [DrivePi0-Base-bf16](https://huggingface.co/rethinklab/DriveMoE/blob/main/DrivePi0_Base_bf16.pt) | 55.85 | 30.00 | [DrivePi0-Base-bf16](/docs/drivepi_base_bf16.json) 
| [DrivePi0-Base-fp32](https://huggingface.co/rethinklab/DriveMoE/blob/main/DrivePi0_Base_fp32.pt) | 65.85 | 42.27 |[DrivePi0-Base-fp32](/docs/drivepi_base_fp32.json)
| [DrivePi0-Full-fp32](https://huggingface.co/rethinklab/DriveMoE/blob/main/DrivePi0_Full_fp32.pt) | 67.41 | 44.09 |[DrivePi0-Full-fp32](/docs/drivepi_full_fp32.json)
| [DriveMoE-Base-bf16](https://huggingface.co/rethinklab/DriveMoE/blob/main/DriveMoE_Base_bf16.pt) | 74.22 | 48.64 | [DriveMoE-Base-bf16](/docs/drivemoe_base_bf16_seed2.json)

To properly set up the evaluation environment, please clone the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) repository at the same directory level as your DriveMoE repository
```console
cd ..
git clone https://github.com/Thinklab-SJTU/Bench2Drive.git
```
This will result in the following directory structure:
```
/parent_directory/
├── DriveMoE/
│   ├── ... (DriveMoE contents)
└── Bench2Drive/
    ├── ... (Bench2Drive contents)
```

Download and setup CARLA 0.9.15, you can skip if you have already downloaded CARLA on your device.
```console
mkdir carla
cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
```

Then link team_code like this:
```console
cp DriveMoE/script/evaluation/closed_loop_drivepi0.sh Bench2Drive/leaderboard/scripts
cp DriveMoE/script/evaluation/closed_loop_drivemoe.sh Bench2Drive/leaderboard/scripts
cp DriveMoE/script/evaluation/requirements.txt Bench2Drive
mkdir Bench2Drive/leaderboard/team_code
cd Bench2Drive/leaderboard/team_code
ln -s ../../../DriveMoE/src/agent/team_code/*  ./
cd ../../   # Now you are in the Bench2Drive directory
```

Then set up eval environment
```console
conda create -n DriveMoE_eval python=3.8
conda activate DriveMoE_eval
export CARLA_ROOT=YOUR_CARLA_PATH
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/DriveMoE_eval/lib/python3.8/site-packages/carla.pth
pip install -r requirements.txt
```
Set closed_loop_drivepi0.sh basing on your device and then run:

```bash
bash leaderboard/scripts/closed_loop_drivepi0.sh
bash leaderboard/scripts/closed_loop_drivemoe.sh
```

Follow [this](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools) to use evaluation tools of Bench2Drive.

### Further Assistance
For questions regarding:
- Benchmark implementation details
- Dataset specifications
- Evaluation metrics

Please refer to the [Bench2Drive Documentation](https://github.com/Thinklab-SJTU/Bench2Drive) or open an issue in the [Bench2Drive Issues](https://github.com/Thinklab-SJTU/Bench2Drive/issues) tracker.