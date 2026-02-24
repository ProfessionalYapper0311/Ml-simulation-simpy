# Bank Teller Queue Simulation & Wait Time Prediction

**Student:** Rupam
**Roll No:** 102317201
**Institution:** Thapar Institute of Engineering and Technology

## Assignment Objective
The goal of this project is to use modeling and simulation to generate synthetic data for a queueing system, and then apply Machine Learning models to predict the target variable based on the generated simulation parameters.

## Methodology

### 1. Data Generation (Simulation Setup)
The simulation was built using the `simpy` library to model a bank teller queue. The objective was to observe how different parameters affect a customer's `mean_wait` time. 
To generate a diverse dataset, 1000 independent simulations were run using randomized parameters within the following bounds:
* **Arrival Rate (`arr_rate`):** 0.5 to 5.0
* **Service Rate (`srv_rate`):** 0.5 to 5.0
* **Number of Tellers (`tellers`):** 1 to 5
* **Total Duration (`duration`):** 60 to 300 minutes

For each of the 1000 simulation runs, the generated parameters and the resulting `mean_wait` time were recorded and compiled into a pandas DataFrame (`simul_data.csv`).

### 2. Machine Learning Evaluation
The generated dataset was treated as a regression task, where the goal was to predict the `mean_wait` time using `arr_rate`, `srv_rate`, `tellers`, and `duration` as input features. 
* **Preprocessing:** The data was split into an 80/20 train-test set. Features were normalized using `StandardScaler` to ensure algorithms sensitive to scale (like KNN and SVM) performed optimally.
* **Model Training:** 10 different scikit-learn regression models were evaluated (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, SVM, KNN, and MLP).
* **Evaluation Metrics:** The models were compared using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$) scores.

## Result Table
Below are the evaluation results of the 10 tested models, sorted by highest $R^2$ score. Ensemble tree-based models significantly outperformed standard linear models.

| Model | RMSE | MAE | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **ExtraTrees** | 3.884 | 1.393 | 0.955 |
| **Random Forest (RF)** | 5.010 | 1.824 | 0.926 |
| **Gradient Boosting (GBM)** | 5.820 | 2.757 | 0.900 |
| **MLP (Neural Net)** | 6.044 | 2.841 | 0.892 |
| **KNN** | 6.059 | 2.205 | 0.892 |
| **Decision Tree (DT)** | 7.130 | 2.354 | 0.850 |
| **Linear Regression** | 14.146 | 8.835 | 0.413 |
| **Ridge** | 14.147 | 8.831 | 0.412 |
| **Lasso** | 14.493 | 8.251 | 0.383 |
| **SVM** | 14.643 | 4.666 | 0.371 |

## Result Graphs Explanation

Three primary visualizations were generated to analyze both the raw simulation data and the model performance:

1.  **Exploratory Data Analysis (`eda_plots.png`):** * *Wait Times Distribution:* Shows a heavy right-skew, indicating that in most simulated scenarios, the wait time is very low, but certain combinations (like high arrival rate + low tellers) cause exponential wait time spikes.
    * *Wait Time by Tellers:* A scatter plot demonstrating the inverse relationship between the number of tellers and the mean wait time. Wait times drop significantly as teller capacity increases.
2.  **Model Comparison (`model_comparison.png`):** A horizontal bar chart visually comparing the $R^2$ scores of all 10 models. It clearly highlights that ExtraTrees and Random Forest captured the non-linear dynamics of the queueing system far better than linear approaches.
3.  **Actual vs. Predicted - ExtraTrees (`actual_vs_predicted.png`):** A scatter plot comparing the actual test set wait times against the predictions made by the best-performing model (ExtraTrees). The predictions map closely to the red dashed ideal-fit line, proving the model successfully learned the underlying queueing mechanics.

## Conclusion
The simulation successfully generated non-linear queueing data. Among the evaluated algorithms, the **ExtraTreesRegressor** was the best-performing model, achieving an $R^2$ score of **0.955**, proving highly capable of predicting customer wait times based on simple queue parameters.
