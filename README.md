# Employee Attrition Prediction

This is a supervised machine learning project based on building a predictive model focused in understanding whether an employee will be churned or not based off several variables.
Click link [here](https://github.com/chrisaliyuda/6-EmployeeAttrition-PredictingLevelofCustomerChurn/blob/main/Employee_Attrition_Prediction%20(2).ipynb) for more project details.

## Data Analysis Process 
### Model used 
1. random forest classifier
2. XGBoost
3. LightGB model

### Data Assessment and Cleaning
```num_var = list(Employee.select_dtypes(include='int'))
num_var = num_var[1:]

# Calculate the number of rows and columns needed based on the number of numerical variables
num_rows = len(num_var) // 3 + (len(num_var) % 3 > 0)
num_cols = min(len(num_var), 3)

plt.figure(figsize=(12, 3 * num_rows))
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

for i, var in enumerate(num_var):
    row, col = i // num_cols, i % num_cols
    sns.distplot(Employee[var], hist=False, ax=axes[row, col])
    axes[row, col].set_title(f'Histogram of {var}')

# Remove empty subplots if there are more than needed
for i in range(len(num_var), num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show()
```
The code checks for distribution of different variables as it might have direct biasness effect to the model if not taken care of. 

### Feature Selection
This was done using the mutual info regression model to get variables relating to our independent variable 
```
for col in x.select_dtypes('O'):
    x[col],_ = x[col].factorize()
mi_scores = mutual_info_regression(x,y, random_state = 0)
mi_scores = pd.Series(mi_scores, name = 'Scores', index = x.columns)
best_features = mi_scores.sort_values(ascending = False).head(17)
```
### Encoding Technique 
```
# Encoding JobRole using target encoding (regularized encoding)
JobRole_impact = X_train.groupby('JobRole')['Attrition'].mean()
X_train['JobRole_encoded'] = X_train['JobRole'].map(JobRole_impact)

# Encoding marital status
Marital_impact = X_train.groupby('MaritalStatus')['Attrition'].mean()
X_train['MaritalStatus_encoded'] =  X_train['MaritalStatus'].map(Marital_impact)

# Encoding Department
Department_impact = X_train.groupby('Department')['Attrition'].mean()
X_train['Department_encoded'] = X_train['Department'].map(Department_impact)
# Encoding BusinessTravel using target encoding (regularized encoding)
BusinessTravel_impact = X_train.groupby('BusinessTravel')['Attrition'].mean()
X_train['BusinessTravel_encoded'] = X_train['BusinessTravel'].map(BusinessTravel_impact)

# Encoding OverTime
OverTime_impact = X_train.groupby('OverTime')['Attrition'].mean()
X_train['OverTime_encoded'] =  X_train['OverTime'].map(OverTime_impact)

# Encoding EducationField
EducationField_impact = X_train.groupby('EducationField')['Attrition'].mean()
X_train['EducationField_encoded'] = X_train['EducationField'].map(EducationField_impact)
```
# Model Training
```
param_grid = {
    'n_estimators': [100, 200, 300],
    'ccp_alpha':[0.0, 0.1, 0.2],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf':[1, 2, 4]}

roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)

rf_classifier = RandomForestClassifier(random_state=0)
grid_search = GridSearchCV(estimator= rf_classifier, param_grid=param_grid, cv=5, scoring=roc_auc_scorer, verbose=2)

grid_search.fit(x_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_roc_auc = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best ROC AUC:", best_roc_auc)
# Splitting data to training and testing set
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.33, random_state = 42)

# setting parameters and training model
class_weights = {0: 1, 1: 2}
rf_classifier = RandomForestClassifier(n_estimators = 100, 
                                       max_depth = 5,
                                       class_weight = class_weights, 
                                       random_state = 42)
model = rf_classifier.fit(x_train, y_train)
```
Using random forest classifier, the model evaluated '83%' accuracy for the test set. 
