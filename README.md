# ConstraintGBM
## Introduction
ConstraintGBM is a machine learning project that focus on the fairness and Neyman-Pearson tasks. The project is based on two popular gradient boosting libraries, [XGBoost](https://github.com/dmlc/xgboost) and [LightGBM](https://github.com/microsoft/LightGBM). Our project customizes the objective function of XGBoost and LightGBM to include additional information like the last predicted score. The customization enables us to address fairness and Neyman-Pearson tasks in our predictions.

## Features
- Customized objective function in C++.
- The main algorithm logic is implemented in Python, hence only need to call xgboost and lightgbm's `training` api.
    - Only support Python interface

## Installation
### Requirements
All of our installed dependencies are consistent with xgboost-2.0.0-dev and lightgbm-3.3.5
### Easy installation
To install ConstrainGBM, simply clone this repository and run the following command:
```bash
bash install.sh
```


## Usage
Some examples are put under `LightGBM-3.3.5/np_fair_test` and `xgboost/np_fair_test` folder.

## Future work



## citing this package

## Disclaimer
- ConstraintGBM is a research software, therefore it should not be used in production.
- Please open an issue if you find any problems, developers will try to fix and find alternatives.

## Acknowledgements
Many thanks to the very active [XGBoost](https://github.com/dmlc/xgboost) and [LightGBM](https://github.com/microsoft/LightGBM) community for their enthusiastic answers to our questions. Many thanks to [chatgpt](https://openai.com/blog/chatgpt/) for assisting us in writing the code.