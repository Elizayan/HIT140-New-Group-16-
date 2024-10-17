import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# use boxplot to print outliers of screentime
df2 = pd.read_csv('dataset2.csv')
df1 = pd.read_csv('dataset1.csv')
df_screentime_filtered = df2[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']]

# create boxplot
plt.figure(figsize=(12, 8))
ax = sns.boxplot(data=df_screentime_filtered, orient="h", palette="Set3", whis=1.5)

# Add title and tags
plt.title('Boxplot of Screentime by Category')
plt.xlabel('Duration of use(hours)')
plt.ylabel('Screen time category')

# Display Graphics
plt.show()
plt.close()

# use boxplot to print outliers of wellbeing
df_wellbeing = pd.read_csv('dataset3.csv')
df_wellbeing_filtered = df_wellbeing[['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr','Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']]

# create boxplot
plt.figure(figsize=(12, 8))
ax = sns.boxplot(data=df_wellbeing_filtered, orient="h", palette="Set3", whis=1.5)

# Add title and tags
plt.title('Boxplot of Well-being Scores by Category')
plt.xlabel('Score')
plt.ylabel('Well-being Attributes')

# Display Graphics
plt.show()
plt.close()

# check the outliers with IQR method
df_screentime = pd.read_csv('/Users/yancolor/PycharmProjects/pythonProject3 140/screen_wellbeing.csv')
df_screentime_n = df_screentime['Total_Screentime'].to_numpy()
pct_25 = np.percentile(df_screentime_n, 25)
pct_75 = np.percentile(df_screentime_n, 75)
iqr = pct_75 - pct_25

print("IQR: %.2f 25th percentile: %.2f 75th percentile: %.2f" % (iqr, pct_25, pct_75))

# outlier < Q1 - 1.5 x IQR or  > Q3+1.5 xIQR ， lower limit = -4.25，upper limit = 41.75
lower_limit = pct_25 - 1.5 * iqr
upper_limit = pct_75 + 1.5 * iqr

# separate outliers from non-outliers
outliers = [i for i in df_screentime_n if i > 41.75]
non_outliers = [i for i in df_screentime_n if i <= 41.75]
num_outliers = len(outliers)
print("The number of ourliers: %d" % num_outliers)

# Calculate the number and percentage of outliers
num_outliers = len(outliers)
outlier_ratio = num_outliers / len(df_screentime_n) * 100

print("The number of screentime outliers: {}".format(num_outliers))
print("The percentage of screentime outliers: {:.2f}%".format(outlier_ratio))

# Descriptive statistics with outliers
mean_with_outliers = np.mean(df_screentime_n)
median_with_outliers = np.median(df_screentime_n)

# Descriptive statistics after excluding outliers
mean_without_outliers = np.mean(non_outliers)
median_without_outliers = np.median(non_outliers)

print("The mean of statistics with outliers: {:.2f}, median: {:.2f}".format(mean_with_outliers, median_with_outliers))
print("The mean of statistics  excluding outliers: {:.2f}, median: {:.2f}".format(mean_without_outliers, median_without_outliers))

# check the outliers with IQR method
# Calculate the total well-being index
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
df_screentime['Total_Wellbeing'] = df_screentime[wellbeing_columns].sum(axis=1)
df_wellbeing_n = df_screentime['Total_Wellbeing'].to_numpy()

pct_25_w= np.percentile(df_wellbeing_n, 25)
pct_75_w= np.percentile(df_wellbeing_n, 75)
iqr = pct_75_w - pct_25_w
print("IQR: %.2f 25th percentile: %.2f 75th percentile: %.2f" % (iqr, pct_25_w,pct_75_w ))

# outlier < Q1 - 1.5 x IQR or  > Q3+1.5 xIQR ，
lower_limit_w = pct_25_w - 1.5 * iqr
upper_limit_w = pct_75_w + 1.5 * iqr
print("Lower limit: %.2f  Upper limit: %.2f" % (lower_limit_w, upper_limit_w))


# separate outliers from non-outliers，lower limit = 24，upper limit = 72
outliers_2 = [i for i in df_wellbeing_n if i > 72 or i<24]
non_outliers_2= [i for i in df_wellbeing_n if 24 <=i <= 72]
num_outliers_2= len(outliers_2)
print("The number of ourliers: %d" % num_outliers_2)

# Calculate the number and percentage of outliers
num_outliers_2 = len(outliers_2)
outlier_ratio_2 = num_outliers_2 / len(df_wellbeing_n) * 100

print("The number of wellbeing outliers: {}".format(num_outliers_2))
print("The percentage of wellbeing outliers: {:.2f}%".format(outlier_ratio_2))

# Descriptive statistics with outliers
mean_with_outliers_2 = np.mean(df_wellbeing_n)
median_with_outliers_2= np.median(df_wellbeing_n)

# Descriptive statistics after excluding outliers
mean_without_outliers_2 = np.mean(non_outliers_2)
median_without_outliers_2 = np.median(non_outliers_2)

print("The mean of statistics with outliers: {:.2f}, median: {:.2f}".format(mean_with_outliers_2, median_with_outliers_2))
print("The mean of statistics  excluding outliers: {:.2f}, median: {:.2f}".format(mean_without_outliers_2, median_without_outliers_2))


# read the csv and get the mean screen time of each group：male, female, majority，minority, high deprivation, low deprivation
df_screentime_wellbeing = pd.read_csv('screen_wellbeing.csv')

# Calculate the daily average screen time for each group
df_male_screentime = df_screentime_wellbeing[df_screentime_wellbeing['gender'] == 1]['Total_Screentime'] / 7
mean_male_screentime = df_male_screentime.mean()
print("Average Daily Screentime for males: %.2f" % mean_male_screentime)

df_female_screentime = df_screentime_wellbeing[df_screentime_wellbeing['gender'] == 0]['Total_Screentime'] / 7
mean_female_screentime = df_female_screentime.mean()
print("Average Daily Screentime for females: %.2f" % mean_female_screentime)

df_majority_screentime = df_screentime_wellbeing[df_screentime_wellbeing['minority'] == 0]['Total_Screentime'] / 7
mean_majority_screentime = df_majority_screentime.mean()
print("Average Daily Screentime for majority: %.2f" % mean_majority_screentime)

df_minority_screentime = df_screentime_wellbeing[df_screentime_wellbeing['minority'] == 1]['Total_Screentime'] / 7
mean_minority_screentime = df_minority_screentime.mean()
print("Average Daily Screentime for minority: %.2f" % mean_minority_screentime)

df_hd_screentime = df_screentime_wellbeing[df_screentime_wellbeing['deprived'] == 1]['Total_Screentime'] / 7
mean_hd_screentime = df_hd_screentime.mean()
print("Average Daily Screentime for high deprivation: %.2f" % mean_hd_screentime)

df_ld_screentime = df_screentime_wellbeing[df_screentime_wellbeing['deprived'] == 0]['Total_Screentime'] / 7
mean_ld_screentime = df_ld_screentime.mean()
print("Average Daily Screentime for low deprivation: %.2f" % mean_ld_screentime)

# Prepare data for plotting
labels = ['Male', 'Female', 'Majority', 'Minority', 'High Deprivation', 'Low Deprivation']
means = [mean_male_screentime, mean_female_screentime, mean_majority_screentime, mean_minority_screentime, mean_hd_screentime, mean_ld_screentime]

x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
rects = ax.bar(x, means, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

# Add labels, title, and custom x-axis labels
ax.set_ylabel('Mean Daily Screentime (hours)')
ax.set_xlabel('Group')
ax.set_title('Mean Daily Screentime by Group')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')  # Adjust labels for readability

# Display the exact mean values on each bar
ax.bar_label(rects, padding=3, fmt='%.2f')

# Add gridlines for easier reading
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()

# Show the plot
plt.show()


# Load the data
df_wellbeing = pd.read_csv('/Users/yancolor/PycharmProjects/pythonProject3 140/screen_wellbeing.csv')

# Define wellbeing columns
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Calculate the mean score for each wellbeing item, then average those means for each group

# Male group: Calculate mean of each wellbeing column and then take the average of those means
mean_male_wellbeing = df_wellbeing[df_wellbeing['gender'] == 1][wellbeing_columns].mean().mean()
print("Average wellbeing item mean for males: %.2f" % mean_male_wellbeing)

# Female group
mean_female_wellbeing = df_wellbeing[df_wellbeing['gender'] == 0][wellbeing_columns].mean().mean()
print("Average wellbeing item mean for females: %.2f" % mean_female_wellbeing)

# Majority group
mean_majority_wellbeing = df_wellbeing[df_wellbeing['minority'] == 0][wellbeing_columns].mean().mean()
print("Average wellbeing item mean for majority: %.2f" % mean_majority_wellbeing)

# Minority group
mean_minority_wellbeing = df_wellbeing[df_wellbeing['minority'] == 1][wellbeing_columns].mean().mean()
print("Average wellbeing item mean for minority: %.2f" % mean_minority_wellbeing)

# High deprivation group
mean_hd_wellbeing = df_wellbeing[df_wellbeing['deprived'] == 1][wellbeing_columns].mean().mean()
print("Average wellbeing item mean for high deprivation: %.2f" % mean_hd_wellbeing)

# Low deprivation group
mean_ld_wellbeing = df_wellbeing[df_wellbeing['deprived'] == 0][wellbeing_columns].mean().mean()
print("Average wellbeing item mean for low deprivation: %.2f" % mean_ld_wellbeing)

# Plotting bar chart for average wellbeing item mean by group
labels = ['Male', 'Female', 'Majority', 'Minority', 'High Deprivation', 'Low Deprivation']
means = [mean_male_wellbeing, mean_female_wellbeing, mean_majority_wellbeing, mean_minority_wellbeing, mean_hd_wellbeing, mean_ld_wellbeing]

x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects = ax.bar(x, means, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

# Add labels, title, and custom x-axis labels
ax.set_ylabel('Mean Wellbeing Item Score')
ax.set_xlabel('Group')
ax.set_title('Average Wellbeing Item Mean by Group')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')  # Rotate labels for readability

# Display the exact mean values on each bar
ax.bar_label(rects, padding=3, fmt='%.2f')

# Add gridlines for easier reading
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()

# Show plot
plt.show()

# merge datasets
merged_df = pd.merge(df2, df1, on='ID')
merged_df.to_csv('merged_data.csv', index=False)

# create sum column
merged_df['C_sum'] = merged_df[['C_we', 'C_wk']].sum(axis=1)
merged_df['G_sum'] = merged_df[['G_we', 'G_wk']].sum(axis=1)
merged_df['S_sum'] = merged_df[['S_we', 'S_wk']].sum(axis=1)
merged_df['T_sum'] = merged_df[['T_we', 'T_wk']].sum(axis=1)

# 'C', 'G', 'S', 'T' sum
total_sums = merged_df[['C_sum', 'G_sum', 'S_sum', 'T_sum']].sum()

# Calculate the sum of male and female usage time in each column
male_sums = merged_df[merged_df['gender'] == 1][['C_sum', 'G_sum', 'S_sum', 'T_sum']].sum()
female_sums = merged_df[merged_df['gender'] == 0][['C_sum', 'G_sum', 'S_sum', 'T_sum']].sum()

# calculate st
male_std = merged_df[merged_df['gender'] == 1][['C_sum', 'G_sum', 'S_sum', 'T_sum']].std()
female_std = merged_df[merged_df['gender'] == 0][['C_sum', 'G_sum', 'S_sum', 'T_sum']].std()

# print result
print("Male standard of time spent by men in each column:")
print(male_std.round(2))
print("\nFemale standard of time spent by female in each column:")
print(female_std.round(2))


# Calculate the percentage of time spent by men and women
male_percentage = (male_sums / total_sums) * 100
female_percentage = (female_sums / total_sums) * 100

# print result
print("Percentage of time spent by men in each column:")
print(male_percentage.round(2))
print("\nPercentage of time spent by female in each column:")
print(female_percentage.round(2))

# Labels for each category
labels = ['C_sum', 'G_sum', 'S_sum', 'T_sum']

# Provided percentage data for men and women
men_percentage = [44.93, 81.49, 37.89, 46.29]
women_percentage = [55.07, 18.51, 62.11, 53.71]

# Provided standard deviations for men and women
men_std = [3.53, 3.80, 4.46, 3.37]
women_std = [3.76, 2.13, 4.64, 3.43]

# Set the bar width
width = 0.5

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the stacked bars for men and women
men_bars = ax.bar(labels, men_percentage, width, yerr=men_std, label='Men', color='skyblue')
women_bars = ax.bar(labels, women_percentage, width, yerr=women_std, bottom=men_percentage, label='Women', color='lightcoral')

# Add labels, title, and legend
ax.set_ylabel('Percentage of Time Spent (%)')
ax.set_title('Percentage of Time Spent by Men and Women in Each Category')
ax.legend()

# Annotate each bar with percentage values
for i, (men, women) in enumerate(zip(men_percentage, women_percentage)):
    ax.text(i, men / 2, f'{men:.2f}%', ha='center', va='center', color='black')  # Men label in the center of men bar
    ax.text(i, men + (women / 2), f'{women:.2f}%', ha='center', va='center', color='black')  # Women label in the center of women bar

# Display the chart
plt.show()

# monority
# Calculate the sum of minority and majority usage time in each column
minority_sums = merged_df[merged_df['minority'] == 1][['C_sum', 'G_sum', 'S_sum', 'T_sum']].sum()
majority_sums = merged_df[merged_df['minority'] == 0][['C_sum', 'G_sum', 'S_sum', 'T_sum']].sum()

# Calculate standard deviation for minority and majority usage times
minority_std = merged_df[merged_df['minority'] == 1][['C_sum', 'G_sum', 'S_sum', 'T_sum']].std()
majority_std = merged_df[merged_df['minority'] == 0][['C_sum', 'G_sum', 'S_sum', 'T_sum']].std()

# Print standard deviations
print("Standard deviation of time spent by minority in each column:")
print(minority_std.round(2))
print("\nStandard deviation of time spent by majority in each column:")
print(majority_std.round(2))

# Calculate the percentage of time spent by minority and majority
minority_percentage = (minority_sums / total_sums) * 100
majority_percentage = (majority_sums / total_sums) * 100

# Print percentage results
print("Percentage of time spent by minority in each column:")
print(minority_percentage.round(2))
print("\nPercentage of time spent by majority in each column:")
print(majority_percentage.round(2))

# Labels for each category
labels = ['C_sum', 'G_sum', 'S_sum', 'T_sum']

# Using the calculated percentage and standard deviation data
minority_percentage_values = minority_percentage.tolist()
majority_percentage_values = majority_percentage.tolist()
minority_std_values = minority_std.tolist()
majority_std_values = majority_std.tolist()

# Set the bar width
width = 0.5

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the stacked bars for minority and majority
minority_bars = ax.bar(labels, minority_percentage_values, width, yerr=minority_std_values, label='Minority', color='skyblue')
majority_bars = ax.bar(labels, majority_percentage_values, width, yerr=majority_std_values, bottom=minority_percentage_values, label='Majority', color='lightcoral')

# Add labels, title, and legend
ax.set_ylabel('Percentage of Time Spent (%)')
ax.set_title('Percentage of Time Spent by Minority and Majority in Each Category')
ax.legend()

# Annotate each bar with percentage values
for i, (minority, majority) in enumerate(zip(minority_percentage_values, majority_percentage_values)):
    ax.text(i, minority / 2, f'{minority:.2f}%', ha='center', va='center', color='black')  # Minority label in the center of minority bar
    ax.text(i, minority + (majority / 2), f'{majority:.2f}%', ha='center', va='center', color='black')  # Majority label in the center of majority bar

# Display the chart
plt.show()

# Calculate the sum of high deprived and low deprived usage time in each column
high_deprived_sums = merged_df[merged_df['deprived'] == 1][['C_sum', 'G_sum', 'S_sum', 'T_sum']].sum()
low_deprived_sums = merged_df[merged_df['deprived'] == 0][['C_sum', 'G_sum', 'S_sum', 'T_sum']].sum()

# Calculate standard deviation for high deprived and low deprived usage times
high_deprived_std = merged_df[merged_df['deprived'] == 1][['C_sum', 'G_sum', 'S_sum', 'T_sum']].std()
low_deprived_std = merged_df[merged_df['deprived'] == 0][['C_sum', 'G_sum', 'S_sum', 'T_sum']].std()

# Print standard deviations
print("Standard deviation of time spent by high deprived in each column:")
print(high_deprived_std.round(2))
print("\nStandard deviation of time spent by low deprived in each column:")
print(low_deprived_std.round(2))

# Calculate the percentage of time spent by high deprived and low deprived
high_deprived_percentage = (high_deprived_sums / total_sums) * 100
low_deprived_percentage = (low_deprived_sums / total_sums) * 100

# Print percentage results
print("Percentage of time spent by high deprived in each column:")
print(high_deprived_percentage.round(2))
print("\nPercentage of time spent by low deprived in each column:")
print(low_deprived_percentage.round(2))

# Labels for each category
labels = ['C_sum', 'G_sum', 'S_sum', 'T_sum']

# Using the calculated percentage and standard deviation data
high_deprived_percentage_values = high_deprived_percentage.tolist()
low_deprived_percentage_values = low_deprived_percentage.tolist()
high_deprived_std_values = high_deprived_std.tolist()
low_deprived_std_values = low_deprived_std.tolist()

# Set the bar width
width = 0.5

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the stacked bars for high deprived and low deprived
high_deprived_bars = ax.bar(labels, high_deprived_percentage_values, width, yerr=high_deprived_std_values, label='High Deprived', color='skyblue')
low_deprived_bars = ax.bar(labels, low_deprived_percentage_values, width, yerr=low_deprived_std_values, bottom=high_deprived_percentage_values, label='Low Deprived', color='lightcoral')

# Add labels, title, and legend
ax.set_ylabel('Percentage of Time Spent (%)')
ax.set_title('Percentage of Time Spent by High Deprived and Low Deprived in Each Category')
ax.legend()

# Annotate each bar with percentage values
for i, (high, low) in enumerate(zip(high_deprived_percentage_values, low_deprived_percentage_values)):
    ax.text(i, high / 2, f'{high:.2f}%', ha='center', va='center', color='black')  # High Deprived label in the center of the high bar
    ax.text(i, high + (low / 2), f'{low:.2f}%', ha='center', va='center', color='black')  # Low Deprived label in the center of the low bar

# Display the chart
plt.show()

# Load datasets
df3 = pd.read_csv('/Users/yancolor/PycharmProjects/pythonProject3 140/dataset3.csv')
df1 = pd.read_csv('/Users/yancolor/PycharmProjects/pythonProject3 140/dataset1.csv')

# Merge datasets
merged_df = pd.merge(df3, df1, on='ID')
merged_df.to_csv('merged_data.csv', index=False)

# Define the 14 wellbeing items
wellbeing_items = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Calculate the sum of well-being scores for each category based on gender
male_sums = merged_df[merged_df['gender'] == 1][wellbeing_items].sum()
female_sums = merged_df[merged_df['gender'] == 0][wellbeing_items].sum()

# Calculate the standard deviation for each gender category
male_std = merged_df[merged_df['gender'] == 1][wellbeing_items].std()
female_std = merged_df[merged_df['gender'] == 0][wellbeing_items].std()

# Calculate the percentage of well-being scores for male and female groups
total_sums = male_sums + female_sums
male_percentage = (male_sums / total_sums) * 100
female_percentage = (female_sums / total_sums) * 100

# Calculate the sum of well-being scores for each category based on majority/minority
majority_sums = merged_df[merged_df['minority'] == 0][wellbeing_items].sum()
minority_sums = merged_df[merged_df['minority'] == 1][wellbeing_items].sum()

# Calculate the standard deviation for each majority/minority category
majority_std = merged_df[merged_df['minority'] == 0][wellbeing_items].std()
minority_std = merged_df[merged_df['minority'] == 1][wellbeing_items].std()

# Calculate the percentage of well-being scores for majority and minority groups
total_sums_minority_majority = majority_sums + minority_sums
majority_percentage = (majority_sums / total_sums_minority_majority) * 100
minority_percentage = (minority_sums / total_sums_minority_majority) * 100

# Calculate the sum of well-being scores for high and low deprivation
high_deprived_sums = merged_df[merged_df['deprived'] == 1][wellbeing_items].sum()
low_deprived_sums = merged_df[merged_df['deprived'] == 0][wellbeing_items].sum()

# Calculate the standard deviation for each deprivation category
high_deprived_std = merged_df[merged_df['deprived'] == 1][wellbeing_items].std()
low_deprived_std = merged_df[merged_df['deprived'] == 0][wellbeing_items].std()

# Calculate the percentage of well-being scores for high and low deprivation groups
total_sums_deprivation = high_deprived_sums + low_deprived_sums
high_deprived_percentage = (high_deprived_sums / total_sums_deprivation) * 100
low_deprived_percentage = (low_deprived_sums / total_sums_deprivation) * 100

# Labels for each well-being item
labels = wellbeing_items

# Convert percentage and standard deviation to lists
male_percentage_values = male_percentage.tolist()
female_percentage_values = female_percentage.tolist()
male_std_values = male_std.tolist()
female_std_values = female_std.tolist()

majority_percentage_values = majority_percentage.tolist()
minority_percentage_values = minority_percentage.tolist()
majority_std_values = majority_std.tolist()
minority_std_values = minority_std.tolist()

high_deprived_percentage_values = high_deprived_percentage.tolist()
low_deprived_percentage_values = low_deprived_percentage.tolist()
high_deprived_std_values = high_deprived_std.tolist()
low_deprived_std_values = low_deprived_std.tolist()

# Set the bar width
width = 0.5

# Create the figure and axis for Gender
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the stacked bars for male and female
male_bars = ax.bar(labels, male_percentage_values, width, yerr=male_std_values, label='Male', color='#457b9d')  # Soft Blue
female_bars = ax.bar(labels, female_percentage_values, width, yerr=female_std_values, bottom=male_percentage_values, label='Female', color='#e63946')  # Soft Red

# Add labels, title, and legend
ax.set_ylabel('Percentage of Well-being Score (%)')
ax.set_title('Percentage of Well-being Scores by Gender for Each Well-being Item')
ax.legend()

# Annotate each bar with percentage values for gender
for i, (male, female) in enumerate(zip(male_percentage_values, female_percentage_values)):
    ax.text(i, male / 2, f'{male:.2f}%', ha='center', va='center', color='black')
    ax.text(i, male + (female / 2), f'{female:.2f}%', ha='center', va='center', color='black')

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Display the chart
plt.show()

# Create the figure and axis for Minority/Majority
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the stacked bars for majority and minority
majority_bars = ax.bar(labels, majority_percentage_values, width, yerr=majority_std_values, label='Majority', color='#2a9d8f')  # Soft Green
minority_bars = ax.bar(labels, minority_percentage_values, width, yerr=minority_std_values, bottom=majority_percentage_values, label='Minority', color='#f4a261')  # Soft Orange

# Add labels, title, and legend
ax.set_ylabel('Percentage of Well-being Score (%)')
ax.set_title('Percentage of Well-being Scores by Minority/Majority for Each Well-being Item')
ax.legend()

# Annotate each bar with percentage values for minority/majority
for i, (majority, minority) in enumerate(zip(majority_percentage_values, minority_percentage_values)):
    ax.text(i, majority / 2, f'{majority:.2f}%', ha='center', va='center', color='black')
    ax.text(i, majority + (minority / 2), f'{minority:.2f}%', ha='center', va='center', color='black')

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Display the chart
plt.show()

# Create the figure and axis for High/Low Deprivation
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the stacked bars for high deprivation and low deprivation
high_deprived_bars = ax.bar(labels, high_deprived_percentage_values, width, yerr=high_deprived_std_values, label='High Deprivation', color='#264653')  # Dark Blue-Grey
low_deprived_bars = ax.bar(labels, low_deprived_percentage_values, width, yerr=low_deprived_std_values, bottom=high_deprived_percentage_values, label='Low Deprivation', color='#a8dadc')  # Light Aqua

# Add labels, title, and legend
ax.set_ylabel('Percentage of Well-being Score (%)')
ax.set_title('Percentage of Well-being Scores by High/Low Deprivation for Each Well-being Item')
ax.legend()

# Annotate each bar with percentage values for high/low deprivation
for i, (high, low) in enumerate(zip(high_deprived_percentage_values, low_deprived_percentage_values)):
    ax.text(i, high / 2, f'{high:.2f}%', ha='center', va='center', color='black')
    ax.text(i, high + (low / 2), f'{low:.2f}%', ha='center', va='center', color='black')

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Display the chart
plt.show()
