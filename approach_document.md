# Approach Document: Traffic Demand Prediction

## 1. Problem Understanding

The task is to predict traffic demand for the rows in `test.csv`. Each row represents a location and time slot, identified mainly by:

- `geohash`
- `day`
- `timestamp`
- road-related fields such as `RoadType`, `NumberofLanes`, `LargeVehicles`, and `Landmarks`

From the first inspection, this is not a normal independent-row regression problem. The rows follow a location-time structure. Demand changes by geohash, by road type, and by time of day. Because of this, I treated the problem as a traffic-curve prediction task rather than only a tabular regression task.

## 2. Initial Data Checks

I first loaded `train.csv`, `test.csv`, and `sample_submission.csv`, then checked:

- file shapes
- column names and data types
- missing values
- train-test column consistency
- number of unique geohashes
- day and timestamp coverage

The target column `demand` is available only in train. The test file has the same feature structure but without the target.

The important observation from the split is that train contains day 48 and the early part of day 49, while test contains the later part of day 49. This made the known early day-49 rows very useful for calibration.

## 3. Missing Value Handling

The main missing fields were:

- `RoadType`
- `Temperature`
- `Weather`

`RoadType` was handled carefully because it was one of the strongest predictors of demand.

During EDA, I found that `RoadType` is strongly determined by:

- `NumberofLanes`
- `LargeVehicles`
- `Landmarks`

So instead of filling missing `RoadType` with a generic missing label, I used a rule-based imputation function. This made the missing road type values consistent with the observed road structure.

For example:

- 1 lane, large vehicles not allowed, no landmark -> Residential
- 1 lane, large vehicles not allowed, landmark present -> Street
- 2 lanes, large vehicles allowed, no landmark -> Highway
- 4 or 5 lanes -> Highway

I kept `RoadType_was_missing` as a feature so the model could still know which rows were originally imputed.

For `Temperature`, I used median-style filling based on the available train data. Weather was filled as a categorical value. Earlier feature-importance checks showed that weather and temperature were weaker than road, location, and time features.

## 4. Target and EDA Insights

The target `demand` is highly skewed. Most rows have low demand, but a smaller number of road-location-time combinations have much higher demand.

The strongest patterns found during EDA were:

1. **Road type matters a lot.**  
   Highways, streets, and residential roads have very different demand levels.

2. **Time matters.**  
   Demand changes across the day, so hour and timestamp features are important.

3. **Location matters.**  
   Different geohashes have different baseline traffic patterns.

4. **Road type and time interact.**  
   The same hour can behave differently for highways, streets, and residential roads.

5. **Location and time interact.**  
   A geohash can have its own time-of-day pattern, so local time features are useful.

These observations guided the final feature engineering.

## 5. Feature Engineering

The final feature set was built around four groups of information.

### Time Features

From `timestamp`, I created:

- `hour`
- `minute`
- `quarter_hour`
- `hour_sin`
- `hour_cos`
- `minute_sin`
- `minute_cos`

Cyclic features help the model understand that time is periodic.

### Location Features

From `geohash`, I created:

- `geo4`
- `geo5`
- `geo_hour`
- `geo_timestamp`
- `geo_freq`
- `geo_hour_freq`

These features capture both broad and local traffic behavior.

### Road Features

I used:

- `RoadType`
- `RoadType_hour`
- `road_lanes`
- `road_large`
- `geo5_road`
- `is_highway`
- `is_street`
- `is_residential`
- `large_allowed`

These features describe how road structure changes demand.

### Day-48 Historical Reference

The full day-48 curve is the strongest historical reference available in the competition train file.

For each geohash and timestamp, I created:

- `d48_same_slot_demand`

This feature tells the model what demand looked like at the same location and same time on day 48.

To avoid identity leakage, I masked `d48_same_slot_demand` for day-48 training rows. This prevents the model from simply copying the target for rows where the same day-48 value is already the label.

## 6. Early Day-49 Calibration Features

Train also contains the early part of day 49. I used this as a calibration window.

For each geohash, I created early day-49 summary features:

- `early49_geo_mean`
- `early49_geo_max`
- `early49_geo_min`
- `early49_geo_std`
- `early49_geo_count`

Then I compared early day-49 demand with the matching day-48 same-slot demand. This gave residual-style features that describe how day 49 started relative to day 48.

The residual summaries were created at different levels:

- geohash level
- geo5 level
- RoadType level

Final residual features included:

- `early49_residual_mean`
- `early49_residual_max`
- `early49_residual_min`
- `early49_residual_std`
- `early49_residual_geo5_mean`
- `early49_residual_geo5_std`
- `early49_residual_road_mean`
- `early49_residual_road_std`

These features let the model learn whether a location or road type was starting day 49 above or below its day-48 reference.

## 7. Features Not Used in Final Model

I tested high-cardinality target encoding features such as geohash-hour target encodings. These looked very strong in random KFold validation, but they were risky because the same geohashes and time slots repeat many times.

The final model avoids those target encodings and keeps more stable features:

- raw categorical interactions
- day-48 same-slot reference
- early day-49 summary and residual features

This made the final approach more explainable and less dependent on random-fold memorization.

## 8. Modeling

The final model uses CatBoostRegressor because:

- it handles categorical features directly
- it works well with mixed numerical and categorical traffic features
- it performed strongly in earlier experiments

The final training setup:

- 5-fold shuffled KFold
- CatBoostRegressor
- RMSE objective
- categorical features passed directly to CatBoost
- predictions clipped to `[0, 1]`

The final model is trained once, and predictions are averaged across folds.

## 9. Final Prediction Correction

After the model prediction, I applied a small residual correction based on the early day-49 residual signal.

The correction uses:

- `early49_residual_mean`
- road-type indicators

The formula gives different weights to different road types:

- Highway receives a stronger residual correction
- Street receives a smaller correction
- Residential is handled conservatively

Multiple correction scales are saved:

- `0.00`
- `0.80`
- `1.00`
- `1.20`
- `1.40`

The main submission uses scale `1.00`, because it applies the correction at the natural strength used in the final feature design.

## 10. Model Saving

The final notebook also saves the trained CatBoost fold models.

Saved artifacts include:

- each CatBoost fold model as `.cbm`
- feature list
- categorical feature list
- residual-correction settings
- metadata needed to reproduce predictions

The model artifacts are saved under:

`saved_models/final_residual_calibration`

## 11. Final Submission Files

The final training cell saves scale-based submission files:

- `final_residual_calibration_scale_000.csv`
- `final_residual_calibration_scale_080.csv`
- `final_residual_calibration_scale_100.csv`
- `final_residual_calibration_scale_120.csv`
- `final_residual_calibration_scale_140.csv`

The main candidate is:

`final_residual_calibration_submission.csv`

This file is copied from:

`final_residual_calibration_scale_100.csv`

## 12. Summary

The final approach is based on a simple traffic story:

1. Demand depends heavily on road type, location, and time.
2. Day 48 gives the best full historical reference curve.
3. The early part of day 49 tells us how day 49 starts relative to day 48.
4. CatBoost learns the interaction between these signals.
5. A small residual correction is applied at the end, with different behavior by road type.

This gives a model that is still tabular and competition-friendly, but uses the time structure of the data instead of treating each row as fully independent.
