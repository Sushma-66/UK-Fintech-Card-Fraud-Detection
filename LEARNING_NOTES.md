# Learning Notes

Study notes for this project — read these alongside the notebooks so you can explain every step.

## 1. Core ideas in one line each

| Term | Plain English |
|---|---|
| **Class imbalance** | One class is much rarer than the other (here 1 fraud per ~400 payments). |
| **Accuracy** | % of predictions that are right. Useless here: "always genuine" is 99.75% accurate. |
| **Recall** | Of all the real fraud, what % did we catch? (missing fraud = losing money) |
| **Precision** | Of everything we flagged, what % was really fraud? (false alarms = annoyed customers) |
| **False negative** | A fraud we missed. **False positive** = a genuine payment we blocked. |
| **F1** | One number that balances precision and recall. |
| **PR-AUC** | How good the model is across *all* thresholds, focused on the rare class. Random guessing ≈ the fraud rate (0.0025). |
| **ROC-AUC** | Similar idea but looks great even for weak models when data is imbalanced — prefer PR-AUC. |
| **Threshold** | The probability above which we say "fraud". 0.5 is just a default. |
| **Train / validation / test** | Learn / tune settings / final exam (used once). |
| **Data leakage** | Information from the test data sneaking into training, making scores look better than reality. |
| **Pipeline** | Bundles preprocessing + model so the same steps run on new data, fitted on training data only. |
| **One-hot encoding** | Turns `channel = online` into columns like `channel_online = 1`, `channel_atm = 0`. |
| **Scaling (StandardScaler)** | Puts numbers on the same scale (mean 0, std 1) so £ amounts don't overpower counts. |
| **Class weights** | Tell the model that mistakes on fraud matter more. No new data is created. |
| **SMOTE** | Creates synthetic fraud rows by blending real fraud rows with their neighbours. |
| **Regularisation (C / alpha)** | Stops a model memorising the training data. In LR small C = simpler; in MLP big alpha = simpler. |
| **Feature engineering** | Building new useful columns, e.g. `amount_to_avg_ratio`, `is_night`. |
| **Permutation importance** | Shuffle one column; if the score drops a lot, the model relies on it. |

## 2. What the four models do

- **Logistic regression** — adds up weighted features and squashes the total into a 0–1 probability. Each weight (coefficient) is explainable.
- **K-means** — unsupervised. Groups similar rows into k clusters without looking at labels. We used *distance to the nearest centre* as an "unusualness" score.
- **Neural network (MLP)** — layers of neurons that can learn combinations of features (e.g. online + night + new merchant). Powerful but harder to explain.
- **Random forest** — hundreds of decision trees trained on random samples, voting together. Strong on tabular data, handles non-linear patterns, no scaling needed.

## 3. Mistakes in the original project that this version fixes

These make excellent interview talking points ("what would you do differently?"):

1. **Judging by accuracy.** The first neural network reported 99.8% accuracy but predicted *no fraud at all*.
2. **`model.predict(X).astype(int)`** — this chops 0.83 down to 0, so every prediction became "genuine". Fix: `(probs >= threshold).astype(int)`.
3. **PCA re-fitted on the test set** (`pca.fit_transform(X_test)`) — the test data ended up on different axes from the training data, which is why validation accuracy collapsed to 47%. Fix: fit on train, then only `transform` the test set.
4. **Assuming K-means cluster 1 = fraud.** Cluster numbers are arbitrary. Fix: map clusters using the training labels.
5. **Tuning on the test set** (picking C by test FNR). That makes the test score optimistic. Fix: tune on a separate validation set.
6. **Random split on time-ordered data.** Fix: train on earlier dates, test on later ones.
7. **Scaling/SMOTE outside a pipeline** risks leakage. Fix: `imblearn.pipeline.Pipeline`.

## 4. Interview questions to practise

1. Why didn't you use accuracy? *(imbalance; "always genuine" gets 99.75%)*
2. What's the difference between precision and recall, and which matters more for fraud? *(both cost money — that's why I used a £ cost to balance them)*
3. How did you deal with the imbalance? *(class weights, SMOTE, random oversampling — compared on validation PR-AUC)*
4. Why apply SMOTE only to the training data?
5. Why split by date instead of randomly?
6. How did you choose the threshold? *(minimised £ cost on the validation week: missed fraud = amount, false alarm = £5)*
7. Why did Random Forest beat the neural network? *(tabular data, few hundred fraud examples, trees capture interactions easily)*
8. If a bank regulator asked how the model makes decisions, which model would you pick and why?
9. Why did K-means do badly?
10. Your data is synthetic — what changes with real data? *(messier, drifting patterns, delayed labels, need monitoring and retraining)*
11. What would you try next? *(gradient boosting, rolling per-customer features, calibration, drift monitoring)*

## 5. Exercises to make it your own

- Change `FALSE_ALARM_COST_GBP` in `src/utils.py` to £1 and £20 — how do the thresholds and results change?
- Remove `txn_count_last_24h` from `NUMERIC_FEATURES`. How much worse does each model get?
- Add `HistGradientBoostingClassifier` to notebook 05.
- Change `FRAUD_RATE` in the generator to 1% and rerun everything.
- Run the same notebooks on the real Kaggle `creditcard.csv` (you'd need to change the feature lists).
