# Quantum Chemistry-Driven Machine Learning Approach for the Prediction of the Surface Tension and Speed of Sound in Ionic Liquids
# How to run the code
reproducing the result for predicting IL properties
  1. SurfaceTension-ILs.csv (database of IL surface tension in comma separated values format)
  2. SpeedOfSound_ILs.csv (database of speed of sound in comma separated values format)
  3. Run python MultilinearRegression_IL-SurfaceTension.py to reproduce IL surface tension for multilinear regression (MLR) model
  4. Run python MultilinearRegression_IL-SpeedOfSound.py to reproduce IL speed of sound for multilinear regression (MLR) model
  5. Run GradientBoostingTree_Hyperparameters-IL.py to optimize the hyperparameters for surface tension and speed of sound
  6. Run GradientBoostingTree_IL-SurfaceTension.py to reproduce IL surface tension for Gradient Boosting Tree (GBT) model
  7. Run GradientBoostingTree_IL-SpeedOfSound.py to reproduce IL speed of sound for Gradient Boosting Tree (GBT) model

# Reference
If you find the code useful for your research, please consider citing
@inproceedings{
  Yue2022predictCO2,
  title={Predicting CO2 Absorption in Ionic Liquid with Molecular Descriptors and Explainable Graph Neural Networks},
  author={Yue Jian and Yuyang Wang and Amir Barati Farimani},
  booktitle={},
  year={2022},
  url={}
}
