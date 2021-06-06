# Multi-Task Parameter Passing Networks

### Data Preparation

##### Taskonomy

Our experiments include evaluaions on the Taskonomy and CityScapes datasets. 

The Taskonomy dataset can be downloaded following the instructions of this repository: https://github.com/StanfordVL/taskonomy. 

Note: Our multi-output case is based on the Cauthron building, which can be accessed from https://github.com/alexsax/taskonomy-sample-model-1. However, to experience the distributed and general cases, please download from the main Taskonomy repository mentioned above.

Our splits of the Taskonomy dataset are included in the `splits/`  folder.

##### CityScapes

The CityScapes dataset and splits can be directly downloaded from AdaShare:  https://github.com/sunxm2357/AdaShare.



### Environment

Please refer to the `requirements.txt` in this directory. We recommend using `Python >= 3.7`.



### Running

The running scripts are included in `scripts/` folder, which involves the running of PPNet in different scenarios and datasets. Remember to change the string `YOUR_DATA_DIR` and `YOUR_SAVE_DIR` into your local data directory and the output directory (for saving checkpoints), respectively.

Running example,

```
sh scripts/ppnet_onebuilding.sh
```




