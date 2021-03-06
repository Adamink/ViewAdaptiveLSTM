<p>
  <img src="https://www.icst.pku.edu.cn/struct/Projects/PKU-MMD_files/teaser.png" width="400">
</p>

This project implements View-Adaptive Neural Network on PKUMMD dataset. 

The codes involve implementation of the following paper: 

- [View Adaptive Recurrent Neural Networks for High Performance
Human Action Recognition from Skeleton Data](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_View_Adaptive_Recurrent_ICCV_2017_paper.pdf)


# Dataset:
Change paths in `scripts/global_variable.py` to path to normalized PKUMMD-v1 data, and path to raw PKUMMD-v2 data and label.
then run
```txt
cd scripts
python gen_v1.py
python gen_v1_strans.py
python gen_v2.py
python gen_v2_strans.py
```

# Run
Change `--dataset_dir` in `scripts/v?_?.sh` to path to corresponding destination in `scripts/global_variable.py`, 
then run
```txt
cd $PATH_TO_FOLDER$
sh ./scripts/v1_CS.sh
sh ./scripts/v1_L.sh
sh ./scripts/v1_M.sh
sh ./scripts/v1_R.sh
sh ./scripts/v2_CS.sh
sh ./scripts/v2_L.sh
sh ./scripts/v2_M.sh
sh ./scripts/v2_R.sh
```

# Results
```txt
Part1 norm 
L 85.4
R 84.1
M 93.2
CS 86.7

Part2 no norm
MR/L 33.3
LR/M 40.3
LM/R 29.7
CS 49.0
```