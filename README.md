# RNNs-Assignment3
# Stock Time Series Prediction Model Document

## Overview
This document introduces a composite deep learning model designed for time series prediction, integrating multiple neural network structures, including Time2Vector (T2V), Long Short-Term Memory Networks (LSTM), Temporal Convolutional Networks (TCN), and attention mechanisms.

## Data Preprocessing Process

### Data Exploration
Initially, `df.head()` displayed basic data features, including date, opening price, highest price, lowest price, closing price, adjusted closing price, and trading volume. `df.columns` further confirmed the names of columns in the dataset.

### Feature Selection
Multiple features were chosen from the data, including opening price, price movement, highest price, lowest price, closing price, adjusted closing price, and trading volume, to serve as model inputs.

### Data Visualization
`features.plot(subplots=True)` was used to visualize the selected features, aiding in understanding the trends of each time series.

### Data Standardization
Selected features were standardized by computing the mean and standard deviation before the training set split point and applying these statistics to transform the entire dataset for better model processing.

### Preparation of Training and Validation Sets
The `multivariate_data` function was defined to construct time series datasets, creating training and validation sets based on a specified historical window size.

### Baseline Model Construction
A simple baseline function was defined that returns the average of input historical data as the next prediction, providing a straightforward yet effective comparison benchmark.

### PyTorch Dataset and DataLoader Preparation
`TensorDataset` and `DataLoader` were used to prepare datasets in PyTorch format, facilitating batch processing and random sampling for the model.

### Principle and Function of the Baseline Model
The baseline model typically serves as a performance benchmark to evaluate more complex models. In this case, the baseline model predicts future values by calculating the average of past values, a simple yet effective method for providing a performance baseline.

### Relationship Between Subsequent Models and Baseline Model
Each subsequent model (such as LSTM, TCN, and MixedModel) used parameters of the baseline model (i.e., statistical characteristics of historical data) as part of their input features or as reference points for performance. The MAE and MSE of the baseline model provided standards for measuring the predictive performance of subsequent models.

## Model Experiments

### LSTM Model
- Captures long-term dependencies in time series to predict future data points.
- Uses the SmoothL1Loss loss function.
- Determined through grid search, the optimizer is Adam with a learning rate of lr=1e-3. After 100 epochs of training, recording training and validation losses, the final results on the dataset show: MAE: 3.11, MSE: 15.22.

### TCN Model
- Utilizes dilated convolution to capture local features in time series data.
- Uses the SmoothL1Loss loss function.
- Determined through grid search, the optimizer is SGD with a learning rate of lr=1e-4. After 100 epochs of training, recording training and validation losses, the final results on the dataset show: MAE: 1.75, MSE: 5.22.

### Mixed Model (MixedModel)
- Combines the optimized models LSTM and TCN to simultaneously capture long and short-term features of time series, using different optimizers for targeted optimization. This combined method could more comprehensively capture features of time series data.
- After 100 epochs of training, recording training and validation losses, the final results on the dataset show MAE: 1.50, MSE: 3.76.

### Deep Mixed Model (BigModel)
- The deep mixed model further extends the concept of the mixed model, integrating T2V layer (Time2Vector instantiated as a single layer), LSTM, TCN, and attention mechanism nhead=5, forming a complex network structure for comprehensive analysis and prediction of time series data.

- Using the Time2Vector layer through linear and periodic encoding of time helps the model better understand and utilize time information. The addition of this time encoding typically allows the model to capture time dependencies and periodic features in time series data, potentially improving prediction accuracy.

- Uses the SmoothL1Loss loss function. The experiment also included using Bayesian optimization technology to choose the optimal optimizer Adam and learning rate 0.00110. After 100 epochs of training, recording training and validation losses, the final results on the dataset show MAE: 1.63, MSE: 4.6.

## Performance Indicators
- Model performance is evaluated by calculating MAE and MSE. Comparing these two indicators across models determines their predictive ability.

## Conclusion
- Experimental results:
  - The LSTM model achieved a MAE of 3.11 and an MSE of 15.22.
  - The TCN model achieved a MAE of 1.75 and an MSE of 5.22.
  - The mixed model (MixedModel) showed better performance with MAE of 1.50 and MSE of 3.76.
  - BigModel MAE dropped to 1.63 and MSE was 4.60.
- The mixed and deep mixed models outperformed the baseline and individual models in terms of MAE and MSE, demonstrating the effectiveness of combining different network models. However, due to the complexities in tuning the MixedModel, which contains different optimizers, the BigModel with an attention mechanism offers more flexibility in parameter adjustment.
- Future work:
  - For the deep mixed model (BigModel), expand the T2V layer and utilize the multi-layer structure of T2V to more deeply extract time features in time series data, further improving the model's adaptability and robustness.
  - Additionally, to address potential overfitting issues, further tuning and regularization techniques are recommended to enhance the model's generalization capabilities. These optimizations are expected to further enhance the performance and practicality of the model in stock time series prediction.
