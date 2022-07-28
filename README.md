# MSCnn-pytorch

## Datasets
ShanghaiTech Dataset: [Google Drive](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)
UCF_QNRF Dataset: [Official Source](https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip)
## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.9

PyTorch: 1.12.0

CUDA: 9.2
## Ground Truth

To generate Ground Truth density map, set the path_train and path_test variabel to specific location as per dataset directory structure.

Then, to start generating density map ground truth, run the `python optimize_make_dataset.py `

## Training Process

Try `python train.py "train path" "test path" "result path" --pre "path to pretrained model" gpu task batch_size worker_count epoch_count print_count` to start training process.

Example `python train.py "/home/mujir/train" "/home/mujir/test" "/home/mujir/result" --pre "/home/mujir/result/checkpoint.pth.tar" 0 0 1 4 200 50`

## Validation

Try `python val.py "testing_image_path" --pre "path to pretrained model" gpu best_result_count`

Example `python val.py "/home/mujir/test/" --pre "/home/mujir/result/best_model.pth.tar" 0 3`

## Results

Shanghai Part A MAE: 82.89

Shanghai Part B MAE: 10.56

UCF_QNRF MAE: 210,26

Pretrained Model [Google Drive](https://drive.google.com/file/d/1C_Wsag3C8c2d2Pmg3hehdZ2HRfYXwhi1/view?usp=sharing)
