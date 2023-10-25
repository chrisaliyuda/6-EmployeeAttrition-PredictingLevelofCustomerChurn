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
