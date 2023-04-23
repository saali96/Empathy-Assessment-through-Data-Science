#!/usr/bin/env python
# coding: utf-8

# In[113]:


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


# # Loading and Preparing the Data

# In[267]:


import os
import pandas as pd
import re

def loadData(directory):
    # create an empty list to store the loaded dataframes
    dataframes = []
    # Load the QuestionnaireIB.csv file
    questionnaire_ib = pd.read_csv('QuestionnaireIB.csv', encoding='ISO-8859-1')
    
    # Specify the columns for interpolation
    columns_to_interpolate = ['Gaze point X', 'Gaze point Y', 'Gaze point left X', 'Gaze point left Y', 
                              'Gaze point right X', 'Gaze point right Y', 'Gaze direction left X', 
                              'Gaze direction left Y', 'Gaze direction left Z', 'Gaze direction right X', 
                              'Gaze direction right Y', 'Gaze direction right Z']
    
    # Columns to check duplicates
    columns_to_check = ['Participant name', 'Eye movement type', 'Recording duration', 'Gaze event duration', 'Pupil diameter left', 'Pupil diameter right']

    # loop through all files in the directory and load csv files with "dataset_III" in their name
    for filename in os.listdir(directory):
        if "dataset_II_" in filename and filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            df.drop('Participant name', axis=1, inplace=True)
            # Extract trial number from the filename
            trial = int(filename.split('_trial_')[1].split('.')[0])
            pattern = r"participant_(\d+)_trial"
            match = re.search(pattern, filename)

            if match:
                participant = match.group(1)
            else:
                print("Participant number not found in filename.")
            # Add 'Trial' column to the dataframe and fill it with trial number
            df['Trial'] = trial
            df['Participant name'] = 'Participant'+participant
            df = pd.merge(df, questionnaire_ib[['Participant name','Total Score extended']], on='Participant name', how='left')
           
            for col in columns_to_interpolate:
                # Use interpolate() method with cubic interpolation for NaN values
                df[col] = df[col].interpolate(method='cubic')
                # Use fillna() method to fill remaining NaN values with original values

            df.fillna(method='backfill', inplace=True)
            df.fillna(method='ffill', inplace=True)

            # Replace , with . only for object type columns
            obj_cols = df.select_dtypes(include=['object']).columns

            # Loop over the columns and try to convert to numeric data type
            for col in obj_cols:
                try:
                    df[col] = pd.to_numeric(df[col].str.replace(',', '.'))
                except ValueError:
                    # If the conversion fails, do nothing
                    pass
                if df[col].apply(lambda x: str(x).isnumeric()).all():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Drop exact duplicate rows based on specified columns
            df.drop_duplicates(subset=columns_to_check, keep='first', inplace=True)

            # Reset index
            df.reset_index(drop=True, inplace=True)
            df = df.drop(columns=['Unnamed: 0'])
            dataframes.append(df)

    # concatenate all dataframes into one dataframe
    all_data = pd.concat(dataframes, ignore_index=True)

    # print the shape of the resulting dataframe
    print("Shape of the loaded data:", all_data.shape)
    return all_data

all_data = loadData("EyeT/EyeT")
df = all_data.copy()


# Merging and using dataset II(explaination and evaluation for this selection at the end). Also, we are checking for duplicates on 'Participant name', 'Recording duration', 'Gaze event duration', 'Pupil diameter left', 'Pupil diameter right' column and dropping duplicate rows and the column **'Unnamed: 0'** as it is unecessary. Furthermore, we are replacing the ',' with '.' to make the columns numerical. Also, changing the dataype of columns to **'float'** that are actually numbers or scalar type but the dataype is currently 'object'. We are doing this so we can use these for correlation and PCA etc and we won't loose meaningful information.
# 
# Moreover, merging the **Total Score extended** of **Questionnaire IB**. We will be using the Total Score Extended as our output variable and we are using this as it includes the score of all the questions from the original questionnaire, plus additional questions.

# # Exploring the data

# In[268]:


all_data = df.copy()
all_data


# In[269]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(all_data.dtypes)


# Checking for data types after the data has been loaded, doing this to ensure no meaningful column is left with unexpected datatype.

# In[270]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Create a copy of the original dataset
all_data_copy = all_data.copy()

# Convert 'Eyetracker timestamp' column to datetime
all_data_copy['Eyetracker timestamp'] = pd.to_datetime(all_data_copy['Eyetracker timestamp'], unit='ms')

# Convert 'Recording duration' column to datetime in milliseconds
all_data_copy['Recording duration'] = pd.to_datetime(all_data_copy['Recording duration'], unit='ms')

# Set figure size
plt.figure(figsize=(12, 6)) # Width = 12 inches, Height = 6 inches

# Create scatter plot
plt.scatter(all_data_copy['Eyetracker timestamp'], all_data_copy['Recording duration'])

# Set plot labels and title
plt.xlabel('Eyetracker Timestamp')
plt.ylabel('Recording Duration')
plt.title('Eyetracker Timestamp vs Recording Duration')

# Show the plot
plt.show()


# This plot shows the Recording duration was majorly between 30 sec to 1 minute 30 swecond.

# In[271]:


pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')


# In[272]:


# Import libraries
import matplotlib.pyplot as plt

# Specify the columns for interpolation
columns_to_interpolate = ['Gaze point X', 'Gaze point Y', 'Gaze point left X', 'Gaze point left Y', 
                          'Gaze point right X', 'Gaze point right Y', 'Gaze direction left X', 
                          'Gaze direction left Y', 'Gaze direction left Z', 'Gaze direction right X', 
                          'Gaze direction right Y', 'Gaze direction right Z']

# Specify the gaze duration column
gaze_duration_column = 'Gaze event duration'

# Loop through each column and create scatter plots and line plots
for col in columns_to_interpolate:
    # Filter rows with missing values in the column
    missing_mask = all_data[col].isna()
    
    # Extract gaze duration for missing and non-missing values
    gaze_duration_missing = all_data.loc[missing_mask, gaze_duration_column]
    gaze_duration_non_missing = all_data.loc[~missing_mask, gaze_duration_column]
    
    # Extract column values for missing and non-missing values
    col_values_missing = all_data.loc[missing_mask, col]
    col_values_non_missing = all_data.loc[~missing_mask, col]
    
    # Create scatter plot of gaze duration vs column values
    plt.figure()
    plt.scatter(gaze_duration_non_missing, col_values_non_missing, label='Non-missing')
    plt.scatter(gaze_duration_missing, col_values_missing, label='Missing', color='red', marker='x')
    plt.xlabel('Gaze Duration')
    plt.ylabel(col)
    plt.title(f'Gaze Duration vs {col}')
    plt.legend()
    plt.show()


# The above relationship shows the relationship of gaze direction and gaze point with gaze duration. We can observe that data is mostly skewed and in some cases where it is asymmetrical or triangle like shape, the data is decreasing after some duration. This shows that data is actually non linear

# In[273]:


all_data.hist(bins=50, figsize=(20,15))


# This histogram plot shows that in some cases data is dtricktly skewed and in some cases it is syymetrical. We can also see some irregular spike, this shows us that data needs further preprocessing.

# In[274]:


import pandas as pd

# Group by 'participant_name' and calculate the sum of 'recording_duration'
grouped = all_data.groupby('Participant name').agg({'Recording duration': 'sum'})

# Define a custom aggregation function to select the first value from an array
select_first = lambda x: x.iloc[0]

grouped['Gaze event duration'] = all_data.groupby('Participant name').agg({'Gaze event duration': 'sum'})

# Group by 'Participant name' and apply the custom aggregation function to 'Total Score Extended' column
grouped['Total Score Extended'] = all_data.groupby('Participant name')['Total Score extended'].agg(select_first)

# Reset the index to make 'Participant name' a regular column
grouped = grouped.reset_index()

# Drop duplicate rows of 'Participant name'
grouped = grouped.drop_duplicates(subset='Participant name', keep='first')

# Sort the DataFrame based on 'Total Score Extended' column in descending order
grouped = grouped.sort_values(by='Total Score Extended', ascending=False)

# Optional: Reset the index to have a consecutive integer index
grouped = grouped.reset_index(drop=True)

# 'grouped' DataFrame now contains unique rows with sum of recording_duration for each participant
grouped


# We have created a new df that has sum of Recording duration and	Gaze event duration for each participant as well as their extended score.

# In[275]:


import matplotlib.pyplot as plt

# Plot 'Recording duration' vs 'Total Score Extended'
plt.scatter(grouped.index, grouped['Recording duration'])
plt.xlabel('Participant')
plt.ylabel('Total Recording Duration (miliseconds)')
plt.title('Total Recording duration vs Participant')
plt.show()


# In this plot we can observe that majority of the participant took almost similar total duration to perform the activity, some outliers can also be observed.

# In[276]:


import matplotlib.pyplot as plt

# Plot 'Recording duration' vs 'Total Score Extended'
plt.scatter(grouped['Gaze event duration'], grouped['Total Score Extended'])
plt.xlabel('Total Gaze Event Duration (miliseconds)')
plt.ylabel('Total Score Extended')
plt.title('Total Gaze Event Duration vs Total Score Extended')
plt.show()


# We can observe a ceiling effect on Total Gaze Event Duration vs Total Score Extended. the score is increasing as total gaze duraion is increasing and after duration reaches 3e^6 it started decreasing, this suggest that relationship between them is non linear.

# In[277]:


import matplotlib.pyplot as plt

# Plot 'Recording duration' vs 'Total Score Extended'
plt.scatter(grouped['Recording duration'], grouped['Total Score Extended'])
plt.xlabel('Recording duration (miliseconds)')
plt.ylabel('Total Score Extended')
plt.title('Recording duration vs Total Score Extended')
plt.show()


# We can observe an irregular behaviour of score with duration, it is dropping and rising within the first quartile of duration. Again this is non linear.

# In[278]:


pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')

print("nulls:",all_data.isnull().sum())


# In[279]:


# perform one-hot encoding
one_hot = pd.get_dummies(all_data['Eye movement type'])

# concatenate the one-hot encoded columns to the original dataframe
all_data = pd.concat([all_data, one_hot], axis=1)

# drop the original categorical column
all_data.drop('Eye movement type', axis=1, inplace=True)

all_data_copy = all_data


# Performing one hot encoding on **Eye movement type** because based it might have some important information and we need to perform correalation on these as well.

# In[280]:


# Compute correlation matrix
corr_matrix = all_data.corr()

# Extract correlation coefficients for the score column
score_corr = corr_matrix['Total Score extended']
print(score_corr)


# We can observe that corelation matrix is majorly negative and not showing any strong correlation of differnt columns with Total Score Extended.

# In[281]:


numeric_cols = all_data.select_dtypes(include=['float64', 'int64','uint8']).columns.tolist()
df_numeric = all_data[numeric_cols].copy()
df_numeric = df_numeric.fillna(0)


# Selecting all the numeric columns so we can normalize it by filling nans with 0 to perform regressor feature extraction and PCA to extract some features.

# In[282]:


# Create the scaler object
scaler = MinMaxScaler()

# Fit and transform the data
df_scaled = scaler.fit_transform(df_numeric)

# Convert the scaled data back to a DataFrame
df_normalized = pd.DataFrame(df_scaled, columns=df_numeric.columns)


# Normalizing the data so we have everything scaled within a range.

# In[283]:


y = df_normalized['Total Score extended']
X = df_normalized.drop(['Total Score extended'], axis=1)


# In[284]:


# Fit the decision tree regressor to the data
clf = DecisionTreeRegressor()
clf.fit(X, y)

# Calculate feature importances
importances = clf.feature_importances_
importances


# In[285]:


# Calculate feature importances
importances = clf.feature_importances_

# Convert the values to regular numbers
importances = np.round(importances * 100, decimals=2)

# Create a dictionary to map attribute names with importance values
importances_dict = {attr: imp for attr, imp in zip(X.columns, importances)}

# Print the dictionary
importances_dict


# Here we can see that eye position right X and Y are showing some significant numbers, among the columns that are useful according to the domain knowledge.

# # PCA

# In[286]:


# One-hot encode categorical variables
data_encoded = df_normalized

# Separate target variable
y = df_normalized['Total Score extended']
X = df_normalized.drop(['Total Score extended'], axis=1)

# Perform PCA
pca = PCA()
pca.fit(df_scaled)

# Choose the number of components to keep
variance_ratio = pca.explained_variance_ratio_
n_components = len(variance_ratio[variance_ratio > 0.01])

# Transform the data to reduced dimension space
df_pca = pca.transform(df_scaled)[:, :n_components]

# Visualize the data
plt.scatter(df_pca[:, 0], df_pca[:, 1])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


# In[287]:


#Extract the most important features
components = pd.DataFrame(pca.components_)
most_important = components.abs().max(axis=0).sort_values(ascending=False)

# Create a dictionary to map feature indices to column names
feature_map = dict(zip(range(len(X.columns)), X.columns))

# Get the column names and loading scores of the top 10 most important features
top_features = [(feature_map[i], score) for i, score in enumerate(most_important[:-1])]

# Sort the top features by loading score in descending order
top_features = sorted(top_features, key=lambda x: x[1], reverse=True)

# Print the top 10 most important features with their corresponding column names and loading scores
for i, (feature, score) in enumerate(top_features):
    print(f"{i+1}. {feature}: {score:.3f}")


# Here we have list of all the useful features and their contributions identified by PCA. At this point we have regressor identified columns as well as PCA ones, now we will apply our domain knowledge to filter and extract the useful features.

# # Preparing the data

# In[288]:


import pandas as pd

# Group the data by Participant and Trial, and find the mode of each group
# df_mode = all_data.groupby(['Participant name', 'Trial']).apply(lambda x: x.mode().iloc[0])

def aggregate_func(group):
    return group.mode().iloc[0]

# Groupby 'Participant' and 'Trial', and apply the custom aggregation function
result = all_data.groupby(['Participant name', 'Trial']).apply(aggregate_func).reset_index(drop=True)

result


# In[289]:


count = result.groupby('Participant name').size().reset_index(name='Count')

print(count)


# Since we have multiple trials for each participant and rows are almost identical, we are extracting mode row, i.e. the one mostly repeated. Extracting one row for each trial and for each participant.

# In[290]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assume 'Trial' column represents time periods

# Sort the dataframe by 'Trial'
result = result.sort_values('Trial')
result


# Here, we are sorting the rows on the basis of trial id so we can treat this problem as time series. As this was highlighted in the feedback that this should be treated as time series problem.

# In[291]:


# Define the useful columns and dropping the rest
cols_to_keep = ['Recording duration', 'Pupil diameter left', 'Pupil diameter right', 'Gaze event duration', 'EyesNotFound','Fixation','Saccade','Unclassified','Gaze event duration', 'Gaze point X', 'Gaze point Y','Gaze point left X','Gaze point left Y','Gaze point right X','Gaze point right Y','Gaze direction left X','Gaze direction left Y','Gaze direction left Z','Gaze direction right X','Gaze direction right Y','Gaze direction right Z','Total Score extended','Participant name']
all_data = result.drop(columns=[col for col in result.columns if col not in cols_to_keep])


# Here, we have idenfitifed and filtered out the useful features. The selection is based on their contribution as well as domain knowledge.

# In[292]:


all_data


# In[293]:


all_data['Participant name'] = all_data['Participant name'].str.extract(r'(\d+)')


# In[294]:


pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')

print("nulls:",all_data.isnull().sum())

pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')


# In[295]:


# Create the scaler object
scalerSelectedCols = MinMaxScaler()

# Fit and transform the data
df_scaled = scalerSelectedCols.fit_transform(all_data)

# Convert the scaled data back to a DataFrame
df_normalized = pd.DataFrame(df_scaled, columns=all_data.columns)
df_normalized


# # Training and Evaluating the Model

# In[296]:


from sklearn.model_selection import GroupShuffleSplit

# Assuming your DataFrame is called 'df_normalized' and the target variable is called 'y'
X = df_normalized.drop(['Total Score extended','Participant name'], axis=1)
y = df_normalized['Total Score extended']
groups = df_normalized['Participant name']  # Specify the group variable

# Create an instance of GroupShuffleSplit with the desired number of splits and test set size
n_splits = 1 
test_size = 0.3  # You can adjust this value as needed
group_shuffle_split = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

# Iterate over the group-wise splits
for train_index, test_index in group_shuffle_split.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Use X_train, y_train for training and X_test, y_test for testing
    # You can also print the shapes of the resulting arrays for verification
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)


# Here, we have normalized and splitted the data into train test. We will be using this split for our model training.

# In[297]:


from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# create an instance of the Random Forest Regressor class
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# define the number of folds for cross-validation
num_folds = 10

# create an instance of GroupKFold with the number of folds
gkf = GroupKFold(n_splits=num_folds)

# perform group k-fold cross-validation on the model
scores = -cross_val_score(rf, X, y, cv=gkf, groups=df_normalized['Participant name'], scoring='neg_mean_squared_error', n_jobs=-1)

variance = np.var(y)

# Create a DataFrame to store the MSE and R2 scores
df_scores_D2 = pd.DataFrame({'R2 score': 1 - (scores/variance),'MSE':scores})
# compute the mean and standard deviation of the scores
mean_score = scores.mean()
std_score = scores.std()

print("Mean Squared Error (MSE) on group k-fold cross-validation: {:.3f} +/- {:.3f}".format(mean_score, std_score))
print("Mean R2 on group k-fold cross-validation:{:.3f}".format(df_scores_D2['MSE'].mean()))
print("Variance",variance)
df_scores_D2


# Now, we are cross validating the rando forest regressor on 10 folds with group k fold method so we can have different participant each time when training and testing and also grouping them together, this will show us the actual performance of the model and we can handle the data leakage. Please note that we I have again changed the evaluation type and now this will be treated like a normal regression problem instead of a timeseries regression problem as it does not make sense.

# Here, we have validated the randomforest regressor with 10 folds. The mean of 10 fold MSE is **0.054** which is very low and we also have standard deviation of MSE on cross-validated folds which is **0.031**. But, we can see that mean R2 score is very low this suggest that model is **0.054** which is not good enough and is explaining less variance in the data than the baseline model, which is not desirable.

# In[298]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create an instance of the Random Forest Regressor class
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test)

# Compute the mean squared error on the test set
mse = mean_squared_error(y_test, y_pred)

# Compute the R2 score on the test set
r2 = r2_score(y_test, y_pred)

# Display the mean squared error (MSE) and R2 score
print("Mean Squared Error (MSE) on test set: {:.3f}".format(mse))
print("R2 Score on test set: {:.3f}".format(r2))


# To our surprise we can see the model is giving the MSE of 0.007 and R2 score of 0.832 which suggest the model's accuracy is 83.2%. Please note that we have already dropped the duplicates and filled the nans so this possibly is not the case of over fitting.

# # Predicting

# In[299]:


def predict(test):
    testDf = test.copy()
    y_pred = rf.predict(testDf)
    testDf['Total Score extended'] = y_pred
    testDf['Participant name'] = 0
    y_pred_denormalized = scalerSelectedCols.inverse_transform(testDf)
    pred = pd.DataFrame(y_pred_denormalized)
    print("Empathy Score",pred.iloc[0][16])
    # Calculate feature importances
    importances = rf.feature_importances_

    # Convert the values to regular numbers
    importances = np.round(importances * 100, decimals=3)

    # Create a dictionary to map attribute names with importance values
    importances_dict = {attr: imp for attr, imp in zip(X.columns, importances)}

    # Print the dictionary
    print("\nThe Explaination for the Score: below we have a dictionary with contribution score where, 100% means mostly contributed:")
    return importances_dict
predict(X_test)


# # Dataset Comparison

# In[300]:


import matplotlib.pyplot as plt
import pandas as pd

# Create a figure for R2 score plot
fig1, ax1 = plt.subplots()
# Plot R2 score for D3
df_scores_D3.plot(y='R2 score', ax=ax1, label='D3 R2 Score', kind='line')
# Plot R2 score for D2
df_scores_D2.plot(y='R2 score', ax=ax1, label='D2 R2 Score', kind='line')
# Set the title, labels, and legend for the R2 score plot
ax1.set_title('R2 Score for Participants in D3 and D2')
ax1.set_xlabel('Fold')
ax1.set_ylabel('R2 Score')
ax1.legend()

# Create a figure for MSE plot
fig2, ax2 = plt.subplots()
# Plot MSE for D3
df_scores_D3.plot(y='MSE', ax=ax2, label='D3 MSE', kind='line', color='red')
# Plot MSE for D2
df_scores_D2.plot(y='MSE', ax=ax2, label='D2 MSE', kind='line', color='green')
# Set the title, labels, and legend for the MSE plot
ax2.set_title('MSE for Participants in D3 and D2')
ax2.set_xlabel('Fold')
ax2.set_ylabel('MSE')
ax2.legend()

# Show the plots
plt.show()


# Here we have plotted the MSE and R2 score on each fold and we can clearly see that dataset 3 has much more variance in R2 and MSE than dataset II but the mean MSE and R2 is almost similar. This suggest that changing the dataset is not actually helping to train a better model.

# In[ ]:





# In[ ]:




