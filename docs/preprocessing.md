## Data processing
To perform model training or open-loop evaluation, you'll need to partition and preprocess the __Bench2Drive__ dataset first.
Run these two scripts to preprocess the training data.
```console
bash script/data_preprocessing/generate_data.sh
bash script/data_preprocessing/window.sh
```
To normalize data during training, we provide dataset statistics. You may also run `bash script/data_preprocessing/get_statistics.sh` to generate them.