# Empathy Assessment through Data Science


# Introduction  
In this project, we will be utilizing the T4 empathy dataset as our data source. After loading the dataset, we perform exploratory data analysis (EDA) to gain insights into the data, including visualizing the data and conducting statistical analysis. Based on the findings from EDA, we select the most relevant features for empathy assessment using techniques such as correlation analysis, feature importance ranking, and PCA dimensionality reduction.

Next, we develop predictive model for empathy assessment using various random tree regressor. We use cross-validation to evaluate the performance of this model based on appropriate evaluation metrics that was MSE.

# Usage 
1. Download the T4 empathy dataset from https://figshare.com/articles/dataset/Eye_Tracker_Data/19729636/2.  

2. Create a new folder named "eyeT" in the same directory where the jupyter file is saved.  

3. Extract the downloaded data into the "eyeT" folder to ensure that the dataset is in the correct file path for the Jupyter file to access.  

4. Run the Empathy Assignment.ipynb or empathy.py, which will load the dataset, perform exploratory data analysis (EDA), select features, preprocess the data and train the machine learning model for empathy assessment. The file contain code snippets for data preprocessing, EDA, feature selection, model development, and model evaluation. 

5. Once the model is trained, you can use it to predict the empathy score for new data or test data. The file include code snippets for making predictions using the trained model, along with generating the reasons for the predicted score and the contribution of each column towards the score. 
