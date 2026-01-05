## Evaluation
### Open-loop evaluation
For open-loop testing, configure the [config file](/config/eval/DrivePi0/base.yaml) (update paths, etc.), then run the script to get the accuracy.
```bash
bash script/evaluation/eval_drivepi0.sh
```

### Closed-loop evaluation
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
Then link team_code like this:
```console
mkdir Bench2Drive/leaderboard/team_code
ln -s DriveMoE/src/agent/team_code/*  Bench2Drive/leaderboard/team_code
```
Follow [this](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools) to use evaluation tools of Bench2Drive.

### Further Assistance
For questions regarding:
- Benchmark implementation details
- Dataset specifications
- Evaluation metrics

Please refer to the [Bench2Drive Documentation](https://github.com/Thinklab-SJTU/Bench2Drive) or open an issue in the [Bench2Drive Issues](https://github.com/Thinklab-SJTU/Bench2Drive/issues) tracker.