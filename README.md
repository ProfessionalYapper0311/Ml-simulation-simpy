# Synthetic Data Generation and ML Prediction for Queueing Systems

## Project Synopsis
This repository contains a complete pipeline for generating synthetic data via Modeling and Simulation (M&S), followed by the application of Machine Learning algorithms for predictive analysis. The project simulates a multi-server bank queue environment to analyze how fluctuating arrival and service rates impact customer wait times, which is then used to train regression models.

## Architecture & Workflow

### Phase 1: Discrete Event Simulation

To generate realistic synthetic data, a multi-server queue (M/M/c model) was developed using the Python framework **SimPy**. The simulation processes dynamic customer arrivals and server (teller) availability over a defined duration. 

To build a robust dataset consisting of 1,000 unique scenarios, the simulation parameters were randomly sampled from the following continuous and discrete uniform distributions:

| Variable Name | Description | Lower Bound | Upper Bound |
| :--- | :--- | :--- | :--- |
| `arr_rate` | Customer arrival rate (Exponential distribution parameter) | 0.5 | 5.0 |
| `srv_rate` | Teller service rate (Exponential distribution parameter) | 0.5 | 5.0 |
| `tellers` | Count of active bank tellers | 1 | 5 |
| `duration` | Total simulation runtime (minutes) | 60 | 300 |

**Target Variable:** The dependent variable extracted from each run is `mean_wait`, representing the average time customers spent waiting in the queue.

### Phase 2: Predictive Modeling
The generated dataset (`simul_data.csv`) was preprocessed using standard scaling and split into an 80/20 train-test distribution. The objective was to predict the non-linear `mean_wait` metric using 10 different regression algorithms from `scikit-learn`:

* **Ensemble Methods:** Random Forest, Gradient Boosting, Extra Trees
* **Linear Models:** Linear Regression, Ridge, Lasso
* **Other Estimators:** Support Vector Regressor (SVM), K-Nearest Neighbors (KNN), Decision Tree, Multi-Layer Perceptron (MLP)

## Experimental Results

The models were evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ scores. Below is a summary of the top-performing models compared against standard linear baselines:

| Algorithm | $R^2$ Score | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| **ExtraTrees Regressor** | 0.955 | 3.884 | 1.393 |
| **Random Forest** | 0.926 | 5.010 | 1.824 |
| **Gradient Boosting** | 0.900 | 5.820 | 2.757 |
| **MLP Neural Network** | 0.892 | 6.044 | 2.841 |
| **Linear Regression** | 0.413 | 14.146 | 8.835 |

### Graphical Analysis
*(Note: Images are generated and saved automatically upon running the notebook)*
*`eda_plots.png`: Displays the heavy right-skewed distribution of wait times and the inverse correlation between teller count and wait duration.
 
 <img width="1256" height="518" alt="image" src="https://github.com/user-attachments/assets/0564991d-af45-46c4-924a-aac41dc87fc5" />
 
*`model_comparison.png`: A bar chart ranking algorithm performance.

 <img width="589" height="318" alt="image" src="https://github.com/user-attachments/assets/7d6dd103-168a-491e-ae86-9edd9b5e2c70" />
 
*`actual_vs_predicted.png`: Maps the ExtraTrees model's predicted wait times against the true simulated values.

 <img width="494" height="344" alt="image" src="https://github.com/user-attachments/assets/2485b982-5f85-47f2-af7f-04e56616246d" />


## Key Takeaways
Standard linear models performed poorly ($R^2$ ~ 0.41) because queueing systems exhibit exponential, non-linear wait time spikes when arrival rates exceed service capacities. The **ExtraTrees Regressor** successfully captured these complex interactions, achieving the highest accuracy with an $R^2$ score of 0.955.

## Execution Guide
Unlike multi-script architectures, this pipeline is contained entirely within a single Jupyter environment for ease of execution.

1.  Clone the repository.
2.  Open `ml-simul.ipynb` in Google Colab or a local Jupyter Notebook environment.
3.  Ensure the required libraries are installed: `pip install simpy pandas numpy scikit-learn matplotlib seaborn`
4.  Run all cells sequentially. The notebook will automatically execute the 1,000 simulations, train the models, and export the `.csv` results and `.png` plots to your working directory.
