# Heart Disease Competition - Exploration Notes

## Current Baseline
- **Model**: XGBoost with default parameters + predict_proba
- **CV ROC AUC**: 0.95786
- **LB Score**: 0.95274
- **LB Position**: 254
- **Top Score**: 0.95391 (gap: 0.00117)

## Strategy
- **First half of month**: Feature engineering with single model (XGBoost)
- **Second half**: Ensembles / meta-models / blending

## Key Observations So Far
1. Using `predict_proba` instead of `predict` was a huge jump (0.88 -> 0.95)
2. Feature importance shows **thallium** dominates (63% importance)
3. Top 5 features: thallium, chest_pain_type, number_of_vessels_fluro, exercise_angina, slope_of_st
4. Bottom features (low importance): fbs_over_120, bp, cholesterol, age

## Data Characteristics
- 630,000 training samples
- 13 features (all numeric after cleaning)
- Binary classification target: heart_disease

## Exploration Areas to Try

### 1. Model Comparison
- [ ] LightGBM
- [ ] CatBoost
- [ ] Random Forest with predict_proba
- [ ] Logistic Regression

### 2. Hyperparameter Tuning
- [ ] learning_rate
- [ ] max_depth
- [ ] n_estimators
- [ ] subsample
- [ ] colsample_bytree

### 3. Ensembling
- [ ] Simple average of multiple models
- [ ] Weighted average based on CV scores
- [ ] Stacking

### 4. Feature Engineering (CURRENT FOCUS)
**A) Interaction features with top predictors:**
- [ ] thallium × chest_pain_type
- [ ] thallium × number_of_vessels_fluro
- [ ] thallium × exercise_angina
- [ ] chest_pain_type × exercise_angina
- [ ] Pairwise interactions among top 5 features

**C) Rescue low-importance features:**
- [ ] Age binning (decades, quartiles)
- [ ] Log/sqrt transforms on continuous features (bp, cholesterol, max_hr)
- [ ] Interactions with low-importance features (age × bp, age × cholesterol)
- [ ] Polynomial features for continuous vars

**Note:** Domain knowledge not helpful with synthetic data - staying purely data-driven

---

## Experiment Log

### Experiment 1: [TODO]
- Description:
- Result:
- Learnings:

