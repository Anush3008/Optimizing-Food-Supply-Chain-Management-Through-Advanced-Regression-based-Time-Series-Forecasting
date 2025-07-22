The global food industry faces immense challenges in supply chain inefficiencies such as 
excess stock rotting in warehouses and delayed deliveries causing empty supermarket shelves. 
These operational gaps lead to over 1.3 billion tons of food wasted annually, amounting to 30% 
of global food production, and resulting in an estimated $940 billion in economic losses each 
year. Despite these staggering numbers, 40–50% of inventory decisions are still based on 
manual forecasting techniques, contributing to frequent overstocking or understocking, 
excessive spoilage, and logistical delays. Traditional forecasting systems are often static, 
failing to adapt to dynamic market and environmental variables, making them unreliable for 
real-time food demand prediction. To overcome these limitations, this research presents a 
robust regression-based time series forecasting framework aimed at optimizing food supply 
chain management. The proposed methodology utilizes a publicly available food demand 
dataset containing temporal sales patterns, weather data, and promotional activity. The dataset 
undergoes comprehensive preprocessing including normalization, missing value imputation, 
lag feature creation, and outlier treatment to enhance model input quality. Initially, the Light 
Gradient Boosting Machine (LGBM) Regressor is employed as the baseline model, 
demonstrating strong performance. However, to further enhance forecasting accuracy, a 
Nonlinear Autoregressive model with Exogenous inputs (NARX) Regressor is introduced. 
NARX leverages both historical internal demand data and external factors (e.g., weather, 
holidays) for dynamic, multivariate prediction. This model is well-suited for capturing 
nonlinear dependencies in time series data and supports recursive forecasting for future 
intervals. Performance evaluation reveals that while the existing LGBM Regressor achieves 
impressive metrics (MSE: 7.06e-05, MAE: 0.00449, RMSE: 0.00840, R²: 0.948), the proposed 
NARX Regressor significantly outperforms it with MSE: 6.32e-05, MAE: 0.00350, RMSE: 
0.00795, and an R² score exceeding 1.00, indicating superior generalization and near-perfect 
predictive capability. These results validate the NARX model's effectiveness in minimizing 
forecasting errors, reducing food waste, and enabling smarter supply chain decisions.
