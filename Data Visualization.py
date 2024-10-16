import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Merging the three datasets
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
df3 = pd.read_csv('dataset3.csv')

df1_sorted = df1.sort_values(by='ID')
df2_sorted = df2.sort_values(by='ID')
df3_sorted = df3.sort_values(by='ID')

merged_df = pd.merge(df1_sorted, df2_sorted, on='ID', how='inner')
merged_df = pd.merge(merged_df, df3_sorted, on='ID', how='inner')

cleaned_df = merged_df.dropna()

cleaned_df.to_csv('cleaned_merged_data.csv', index=False)

# Reading the cleaned data
df = pd.read_csv('cleaned_merged_data.csv')

# Defining screen time variables
screen_time_vars = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']

# Grouping the data by gender, minority, and deprived status
grouped = df.groupby(['gender', 'minority', 'deprived'])[screen_time_vars].mean()

# Resetting the index
grouped = grouped.reset_index()

# Calculating the totals for each group
totals = grouped[screen_time_vars].sum(axis=1)

# Defining the colors (Morandi palette)
morandi_colors = ['#D8C292', '#B5C48F', '#A8C6D3', '#E5E2D4', '#C2B7A3', '#A0B8A6', '#BBC1D0', '#F1E9D2']

# Plotting the stacked bar chart
fig, ax = plt.subplots(figsize=(15, 9))
bottom = [0] * len(grouped['gender'])
bar_width = 0.67

for i, var in enumerate(screen_time_vars):
    bars = ax.bar(grouped.index, grouped[var], bottom=bottom, color=morandi_colors[i], label=var, width=bar_width)

    # Adding percentage labels to each segment
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        percentage = f'{(height / total * 100):.1f}%'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, percentage,
                ha='center', va='center', color='black', fontsize=10)

    bottom += grouped[var]

# Setting X-axis labels without decimals
ax.set_xticks(grouped.index)
ax.set_xticklabels([f"G{int(row['gender'])} M{int(row['minority'])} D{int(row['deprived'])}" for _, row in grouped.iterrows()],
                   rotation=45, ha='right')

# Adding explanations in the legend
ax.legend(
    title='Screen Time Categories\n\nG0: Female\nG1: Male\nM0: Non-Minority\nM1: Minority\nD0: Non-Deprived\nD1: Deprived\n',
    fontsize=9, title_fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

# Setting the axis labels and title
ax.set_ylabel('Average Screen Time (hours)')
ax.set_title('Screen Time by Gender, Minority, and Deprived Status')

# Adjusting the layout
plt.tight_layout()
plt.show()

df = pd.read_csv('cleaned_merged_data.csv')

# Defining wellbeing variables
wellbeing_vars = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme',
                  'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Grouping the data by gender, minority, and deprived status
grouped = df.groupby(['gender', 'minority', 'deprived'])[wellbeing_vars].mean().reset_index()

# Calculating the totals for each group
totals = grouped[wellbeing_vars].sum(axis=1)

# Ensure morandi_colors list has enough colors for each wellbeing variable
morandi_colors = ['#D8C292', '#B5C48F', '#A8C6D3', '#E5E2D4', '#C2B7A3', '#A0B8A6', '#BBC1D0', '#F1E9D2',
                  '#8FB5AB', '#B8D4CD', '#C9A3A1', '#D7CACB', '#9FB7B9', '#E7B8A2', '#A9D4B2']  # Added two more colors

# Check length of colors and wellbeing variables
if len(morandi_colors) < len(wellbeing_vars):
    raise ValueError(f"Color list (length {len(morandi_colors)}) is shorter than the number of wellbeing variables (length {len(wellbeing_vars)})")

# Plotting the stacked bar chart
fig, ax = plt.subplots(figsize=(15, 9))
bottom = [0] * len(grouped['gender'])
bar_width = 0.67

# Plotting each wellbeing variable with a corresponding color
for i, var in enumerate(wellbeing_vars):
    bars = ax.bar(grouped.index, grouped[var], bottom=bottom, color=morandi_colors[i], label=var, width=bar_width)

    # Adding percentage labels to each segment
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        percentage = f'{(height / total * 100):.1f}%'  # Calculate percentage
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, percentage,
                ha='center', va='center', color='black', fontsize=10)

    bottom += grouped[var]

# Setting X-axis labels without decimals
ax.set_xticks(grouped.index)
ax.set_xticklabels([f"G{int(row['gender'])} M{int(row['minority'])} D{int(row['deprived'])}" for _, row in grouped.iterrows()],
                   rotation=45, ha='right')

# Adding legend and explanations
ax.legend(
    title='Wellbeing Categories\n\nG0: Female\nG1: Male\nM0: Non-Minority\nM1: Minority\nD0: Non-Deprived\nD1: Deprived\n',
    fontsize=9, title_fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

# Setting axis labels and title
ax.set_ylabel('Average Wellbeing Scores')
ax.set_title('Wellbeing by Gender, Minority, and Deprived Status')

# Adjusting layout
plt.tight_layout()
plt.show()



# Sum of all time columns
df['total_time'] = df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].sum(axis=1)

# Average of all time columns
df['average_time'] = df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].mean(axis=1)

# Selecting happiness-related columns
happiness_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Create correlation matrix
corr_matrix = df[['total_time', 'average_time'] + happiness_columns].corr()

# Plotting the heatmap
plt.figure(figsize=(15, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between Total/Average Screen Time and Happiness Indicators')
plt.show()

# Calculate total_time and average_time
df['total_time'] = df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].sum(axis=1)
df['average_time'] = df['total_time'] / 8

# Calculate total_wellbeing
df['total_wellbeing'] = df[['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr',
                            'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']].sum(axis=1)

df['gender'] = df['gender'].replace({0: 'Female', 1: 'Male'})

# Draw scatter plots
plt.figure(figsize=(15, 9))

sns.scatterplot(x='average_time', y='total_wellbeing', data=df, hue='gender', palette=['red', 'blue'], alpha=0.1)

# Draw the regression line
for gender, color in zip(['Female', 'Male'], ['blue', 'red']):
    subset = df[df['gender'] == gender]

    # Regression line fitting
    X = subset['average_time'].values.reshape(-1, 1)
    y = subset['total_wellbeing'].values
    model = LinearRegression()
    model.fit(X, y)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))

    plt.plot(x_range, y_pred, color=color, linewidth=2, label=f'{gender} regression line')

plt.title('Relationship between Screen Time and Well-being by Gender')
plt.xlabel('Average Screen Time')
plt.ylabel('Total Well-being')
plt.legend(title='Gender', loc='upper right')

plt.show()

# Calculate total_time and average_time
df['total_time'] = df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].sum(axis=1)
df['average_time'] = df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].mean(axis=1)

# Calculate total_wellbeing
df['total_wellbeing'] = df[['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep',
                            'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']].sum(axis=1)

df['minority'] = df['minority'].replace({0: 'Majority', 1: 'Minority'})

# Draw scatter plots
plt.figure(figsize=(15, 9))

sns.scatterplot(x='average_time', y='total_wellbeing', data=df, hue='minority', palette=['blue', 'red'], alpha=0.1)

# Draw the regression line
for minority, color in zip(['Majority', 'Minority'], ['blue', 'red']):
    subset = df[df['minority'] == minority]

    # Regression line fitting
    X = subset['average_time'].values.reshape(-1, 1)
    y = subset['total_wellbeing'].values
    model = LinearRegression()
    model.fit(X, y)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))

    plt.plot(x_range, y_pred, color=color, linewidth=2, label=f'{minority} regression line')

plt.title('Relationship between Screen Time and Well-being by Minority Status')
plt.xlabel('Average Screen Time')
plt.ylabel('Total Well-being')
plt.legend(title='Minority Status', loc='upper right')

plt.show()

# Calculate total_time and average_time
df['total_time'] = df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].sum(axis=1)
df['average_time'] = df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].mean(axis=1)

# Calculate total_wellbeing
df['total_wellbeing'] = df[['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep',
                            'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']].sum(axis=1)

df['deprived'] = df['deprived'].replace({0: 'Not Deprived', 1: 'Deprived'})

# Draw scatter plots
plt.figure(figsize=(15, 9))

sns.scatterplot(x='average_time', y='total_wellbeing', data=df, hue='deprived', palette=['blue', 'red'], alpha=0.1)

# Draw the regression line
for deprived_status, color in zip(['Not Deprived', 'Deprived'], ['blue', 'red']):
    subset = df[df['deprived'] == deprived_status]

    # Regression line fitting
    X = subset['average_time'].values.reshape(-1, 1)
    y = subset['total_wellbeing'].values
    model = LinearRegression()
    model.fit(X, y)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))

    plt.plot(x_range, y_pred, color=color, linewidth=2, label=f'{deprived_status} regression line')

plt.title('Relationship between Screen Time and Well-being by Deprived Status')
plt.xlabel('Average Screen Time')
plt.ylabel('Total Well-being')
plt.legend(title='Deprived Status', loc='upper right')

plt.show()
