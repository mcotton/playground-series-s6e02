# Heart Disease Competition - Exploration Notes

## Current Best
- **Model**: XGBoost with early stopping + predict_proba
- **Features**: 13 original + 3 interactions (max_hr/thallium, sex×chest_pain_type, chest_pain_type×slope_of_st)
- **LB Score**: 0.95291 (0.95292 with polynomial features - likely noise)
- **Top Score**: 0.95391 (gap: ~0.00100)

## Original Baseline
- **Features**: 13 original only
- **CV ROC AUC**: 0.95786
- **LB Score**: 0.95274

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
- [x] Age binning (decades, quartiles) - **didn't help**
- [x] Log/sqrt transforms on continuous features - **didn't help**
- [x] Interactions with low-importance features (age × bp, age × cholesterol) - **made it worse**
- [x] Polynomial features for continuous vars - **likely noise**

**Note:** Domain knowledge not helpful with synthetic data - staying purely data-driven

**TODO - Revisit later:**
- Thallium dropped from #1 (63%) to #14 after adding max_hr/thallium interaction
- Bottom features (thallium, bp, fbs_over_120) may be candidates for removal or rescue via interactions
- Test if removing redundant raw features helps reduce noise

---

## Experiment Log

### Experiment 1: Pairwise Interaction Features (multiply + divide)
- **Description**: Added ALL pairwise interaction features using multiplication and division
- **Result**:
  - CV ROC AUC: 0.95786 → 0.95943 (+0.00157)
  - LB: 0.95274 → 0.95228 (-0.00046) **OVERFIT**
- **Features**: 326 total (13 original + 313 interactions)
- **Learning**: Too many features. Model found noise in training that didn't generalize. Need to be selective.

### Experiment 2: Top 3 Interaction Features Only
- **Description**: Kept only top 3 pairwise features from Experiment 1
- **Features**: 13 original + 3 interactions (max_hr/thallium, chest_pain_type×slope_of_st, sex×chest_pain_type)
- **Result**: LB 0.95274 → 0.95289 (+0.00015) **IMPROVEMENT**
- **Learning**: Fewer, targeted features > many noisy features

### Experiment 3: n_estimators Tuning with Early Stopping
- **Description**: Tested increasing n_estimators to 2000, then used early stopping to find optimal value
- **Results**:
  - n_estimators=2000: CV 0.97715, LB 0.94660 **SEVERE OVERFIT**
  - n_estimators=67 (via early_stopping_rounds=50): CV 0.95722, LB 0.95291 **NEW BEST**
- **Learning**: Default n_estimators (100) was already close to optimal. Early stopping is essential for preventing overfitting. High CV score doesn't mean good generalization.
- **Method**: Used train/val split with early_stopping_rounds=50, then applied best_iteration to final model

### Experiment 4: 4th Interaction Feature
- **Description**: Added `chest_pain_type × thallium` as 4th interaction feature
- **Result**: LB 0.95291 → 0.95285 (-0.00006) **NO IMPROVEMENT**
- **Learning**: 3 interactions is the sweet spot; adding more hurts

### Experiment 5: Remove Low-Importance Features
- **Description**: Removed bottom 3 features (fbs_over_120, bp, thallium)
- **Result**: LB worse **NO IMPROVEMENT**
- **Learning**: Even low-importance features contribute; keep all original features

### Experiment 6: Binned Features
- **Description**: Added 4 binned features for continuous variables
- **Result**: All 4 features landed at bottom of feature importance **NO IMPROVEMENT**
- **Learning**: Binning doesn't help; XGBoost already handles continuous features well via splits

### Experiment 7: Polynomial Features (square/sqrt)
- **Description**: Added sq and sqrt transforms for bp, max_hr, cholesterol (6 features)
- **Result**: LB 0.95291 → 0.95292 (+0.00001) - likely noise
- **Learning**: Gain too small to be confident it's real signal vs LB variance

### Experiment 8: Low-Importance Feature Interactions
- **Description**: Added age×bp, age×cholesterol, bp×cholesterol
- **Result**: Hurt both CV and LB **WORSE**
- **Learning**: Interactions between weak features just add noise

### Experiment 9: Stacking (LR prediction as feature)
- **Description**: Added LogisticRegression prediction as feature to XGBoost
- **Result**: CV 0.95691, LB 0.95289 **WORSE**
- **Learning**: LR prediction became #1 feature, XGBoost over-relied on it. Stacking works better as weighted average, not as input feature.

### Experiment 10: Categorical Encoding for thallium & chest_pain_type
- **Description**: Treated thallium and chest_pain_type as categorical dtype with enable_categorical=True
- **Result**: LB 0.95291 → 0.95276 (-0.00015) **WORSE**
- **Note**: Interaction features (max_hr/thallium, sex×chest_pain_type) were still computed from integers, creating a conflict

### Experiment 10b: Categorical Encoding WITHOUT interaction features
- **Description**: Removed interaction features that leaked ordinal info, kept categorical encoding
- **Result**: LB 0.95288 - better than 10a but still below best (0.95291)
- **Learning**: Integer interactions + leaking ordinal info into categoricals hurts. Clean categorical is better than mixed, but integers + interactions still wins overall.

### Experiment 11: [TODO] - Suggested Next Steps
**Options to try:**
1. Try different models (LightGBM, CatBoost) standalone
2. Simple weighted average ensemble (not stacking)
3. Ablation study - test each of the 3 current interactions individually

