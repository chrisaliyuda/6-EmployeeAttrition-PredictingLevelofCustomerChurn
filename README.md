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
