# Urban Traffic Demand Forecasting

Machine learning project for forecasting location-level traffic demand from geospatial, road, and time-based signals.
This is my solution to the Flipkart Gridlock hackathon 2.0 which secured r square of 0.98 and was ranked 1445 among 10000 participants.

## Result

- Public leaderboard R2: **0.918**
- Rank: **1445 / 10000 participants**
- Percentile: **Top 14.5%**

## Problem

The task was to predict traffic demand for road segments represented by geohashes and 15-minute time slots. The train-test split followed a time-based structure: the training data contained a complete previous-day traffic curve and an early calibration window for the target day, while the test data covered later target-day timestamps.

This made the problem closer to spatiotemporal forecasting than ordinary tabular regression.

## Approach

The final solution combined EDA-driven feature engineering with CatBoost regression.

Key ideas:

- Parsed timestamps into hour, minute, quarter-hour, and cyclic time features.
- Created geohash-time and road-time interaction features such as `geo_hour`, `geo_timestamp`, and `RoadType_hour`.
- Used rule-based imputation for missing `RoadType` values using `NumberofLanes`, `LargeVehicles`, and `Landmarks`.
- Used previous-day same-slot demand as a historical reference while masking training rows where that feature would leak the target.
- Built early-window residual calibration features by comparing known target-day demand against previous-day same-slot demand.
- Trained CatBoost with categorical support for high-cardinality location and road features.
- Generated residual-correction variants from the same model to control calibration strength.

## Why CatBoost

CatBoost was a strong fit because the dataset contained many categorical and high-cardinality fields:

- `geohash`
- `RoadType`
- `geo_hour`
- `geo_timestamp`
- road interaction categories

CatBoost handled these features without requiring one-hot encoding or heavy preprocessing.

## Validation Lessons

Random KFold validation was useful for debugging but overly optimistic because many rows shared the same geohashes and time slots. The final approach treated local CV as a sanity check and relied more on split-aware reasoning:

- avoid direct target leakage from previous-day reference features;
- use the known early target-day rows only as calibration signals;
- compare model variants through leaderboard feedback and prediction-distribution checks.

## Repository Files

- `competition_data_eda.ipynb` - full EDA, experiments, and final modeling workflow.
- `approach_document.md` - detailed modeling explanation.
- `FINAL_EMERGENCY_SUBMISSION.csv` - final prediction submitted before completion of 5 fold cv due to lack of time.

## Tech Stack

Python, pandas, NumPy, scikit-learn, CatBoost, Jupyter Notebook

