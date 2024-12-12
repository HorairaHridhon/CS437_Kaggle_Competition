import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif

# Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# # Basic Dataset Information
# print('Train set Shape:', train_data.shape)
# print('\nMissing Values in train data:\n', train_data.isnull().sum())
# print('\nData Types in train data :\n', train_data.dtypes)

# Save the 'id' column before dropping it
test_ids = test_data["id"]
train_data = train_data.drop(columns=['id', 'Name'])
test_data = test_data.drop(columns=['id', 'Name'])

# train_data['Sleep Duration'].unique()

# Mapping sleep duration values into broader categories
def map_sleep_duration(value):
    # Define the mapping logic
    if value in ['1-2 hours', '2-3 hours', '3-4 hours', '1-3 hours',
                 '4-5 hours', 'Less than 5 hours', 
                 '1-6 hours', '5-6 hours', '4-6 hours', '3-6 hours',
                 ]:
        return 'Less than 6 hours'
        
    elif value in ['6-7 hours', '7-8 hours', '6-8 hours',
                   'Moderate', '8 hours', 
                  ]:
        return '6-8 hours'
    # elif value in ['7-8 hours']:
    #     return '7-8 hours'
    elif value in ['8-9 hours', '9-11 hours', '10-11 hours', 'More than 8 hours']:
        return 'More than 8 hours'
    else:
        # Treat other invalid values as NaN
        return None

# Apply the mapping to the "Sleep Duration" column
train_data['Sleep Duration'] = train_data['Sleep Duration'].apply(map_sleep_duration)

# Apply sleep duration mapping
test_data['Sleep Duration'] = test_data['Sleep Duration'].apply(map_sleep_duration)

# # Display unique values after cleaning
# print(train_data['Sleep Duration'].unique())
# print(test_data['Sleep Duration'].unique())

# categorical_columns = train_data.select_dtypes(include=['object', 'category']).columns
# numerical_columns = train_data.select_dtypes(include=["float64", "int64"]).columns

# categorical_columns_test = test_data.select_dtypes(include=['object', 'category']).columns
# numerical_columns_test = test_data.select_dtypes(include=["float64", "int64"]).columns

# train_data['Dietary Habits'].unique()

 

# Mapping dietary habits values into broader categories
def map_dietary_habits(value):
    # Define the mapping logic
    if value in ['Healthy', 'More Healthy']:
        return 'Healthy'
    elif value in ['Unhealthy', 'Less than Healthy', 'No Healthy', 'Less Healthy']:
        return 'Unhealthy'
    elif value in ['Moderate']:
        return 'Moderate'
    else:
        # Treat other invalid values as NaN
        return None

# Apply the mapping to the "Sleep Duration" column
train_data['Dietary Habits'] = train_data['Dietary Habits'].apply(map_dietary_habits)

# Apply dietary habits mapping
test_data['Dietary Habits'] = test_data['Dietary Habits'].apply(map_dietary_habits)

# # Display unique values after cleaning
# print(train_data['Dietary Habits'].unique())
# print(test_data['Dietary Habits'].unique())

# train_data['Profession'].unique()

# Mapping profession values into broader categories
def map_profession(value):
    # Define the mapping logic
    if value in ['Teacher', 'Researcher', 'Research Analyst', 'Student',
                 'Academic',
                ]:
        return 'Academic'
        
    elif value in ['Software Engineer', 'Civil Engineer', 'Mechanical Engineer', 
                   'Data Scientist', 'Architect']:
        return 'Engineering'
        
    elif value in ['Business Analyst', 'Finanancial Analyst', 'Financial Analyst', 'Analyst',
                   'Accountant',  'Sales Executive', 'Digital Marketer', 'Customer Support',
                   'Manager', 'Marketing Manager', 'HR Manager', 'City Manager',
                   'Investment Banker', 'Educational Consultant', 'Consultant', 'Travel Consultant', 
                   'Family Consultant', 'Entrepreneur',]:
        return 'Corporate'

        
    elif value in ['Doctor', 'Medical Doctor', 'Pharmacist']:
        return 'Medicine'
        
    elif value in ['Chef', 'Chemist', 'Electrician', 'Plumber', 'Lawyer', 
                   'Judge', 'Pilot', 'Working Professional']:
        return 'Misc Professions'
    
    # elif value in ['Educational Consultant', 'Consultant', 'Travel Consultant', 
    #                'Family Consultant']:
    #     return 'Consultant'

    elif value in ['UX/UI Designer', 'Content Writer', 'Graphic Designer',]:
        return 'Freelancers'

    elif value in ['Unemployed']:
        return 'Unemployed'
    
    else:
        # Treat other invalid values as NaN
        return None

# Apply the mapping to the "Sleep Duration" column
train_data['Profession'] = train_data['Profession'].apply(map_profession)

# Apply profession mapping
test_data['Profession'] = test_data['Profession'].apply(map_profession)

# # Display unique values after cleaning
# print(train_data['Profession'].unique())
# print(test_data['Profession'].unique())

 

# train_data['Degree'].unique()

# Mapping degrees into broader categories
def map_degrees(value):
    # Define the mapping logic
    if value in ['BHM', 'LLB', 'B.Pharm', 'BBA', 'BSc', 'B.Sc', 'BE', 'BCA', 
                 'BA', 'B.Arch', 'B.Com', 'B.Ed', 'Class 12', 'B.Tech',
                 'BH', 'BEd', 'S.Teech', 'Class 11', 'P.Com', 'LL.Com', 
                 'L.Ed', 'P.Pharm', 'BArch', 'S.Pharm', 'LLBA','LLCom', 
                 'B BA', 'B.B.Arch', 'BB', 'LLTech', 'S.Arch', 'B.Student', 'LL B.Ed',
                 'LLS', 'LLEd', 'E.Tech', 'N.Pharm', 'LCA', 'B B.Com', 'HCA', 'LHM',
                 'BPharm', 
                 ]:
        return 'Undergraduate'
        
    elif value in ['MBA', 'M.Tech', 'MSc', 'M.Com', 'MCA', 'ME', 'MTech', 'MEd',
                   'M.Ed', 'MHM', 'MPA', 'MA', 'M.Arch', 'M. Business Analyst',
                   'M_Tech', 'M.S', 'MPharm', 'M.Pharm', 
                   ]:
        return 'Postgraduate'
        
    elif value == 'PhD':
        return 'Doctorate'
        
    elif value in ['MBBS', 'MD', 'LLM', 'H_Pharm', ]:
        return 'Professional Degree'

    
    else:
        # Treat other invalid values as NaN
        return None

# Apply the mapping to the "Degree" column
train_data['Degree'] = train_data['Degree'].apply(map_degrees)

# Apply degree mapping
test_data['Degree'] = test_data['Degree'].apply(map_degrees)

# # Display unique values after cleaning
# print(train_data['Degree'].unique())
# print(test_data['Degree'].unique())

valid_cities = [
       'Ludhiana', 'Varanasi', 'Visakhapatnam', 'Mumbai', 'Kanpur',
       'Ahmedabad', 'Thane', 'Nashik', 'Bangalore', 'Patna', 'Rajkot',
       'Jaipur', 'Pune', 'Lucknow', 'Meerut', 'Agra', 'Surat',
       'Faridabad', 'Hyderabad', 'Srinagar', 'Ghaziabad', 'Kolkata',
       'Chennai', 'Kalyan', 'Nagpur', 'Vadodara', 'Vasai-Virar', 'Delhi',
       'Bhopal', 'Indore', 'Ishanabad', 'Gurgaon',
]

# # valid_professions = [
       
# # ]

# Step 2: Replace invalid entries with NaN
def clean_column(column, valid_values):
    return column.apply(lambda x: x if x in valid_values else None)

train_data['City'] = clean_column(train_data['City'], valid_cities)
test_data['City'] = clean_column(test_data['City'], valid_cities)

# print(train_data['City'].unique())
# print(test_data['City'].unique())

# Convert ordinal columns to categorical
ordinal_columns = [
    'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 
    'Job Satisfaction', 'Financial Stress'
]

for col in ordinal_columns:
    train_data[col] = train_data[col].astype('category')
    test_data[col] = test_data[col].astype('category')

categorical_columns = train_data.select_dtypes(include=['object', 'category']).columns
numerical_columns = train_data.select_dtypes(include=["float64", "int64"]).columns.difference(['Depression'])

 

X = train_data.drop(columns=['Depression'])
y = train_data['Depression']

# Pipeline for processing and modeling
num_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
cat_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_preprocessor, numerical_columns),
        ('cat', cat_preprocessor, categorical_columns)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('select', SelectKBest(score_func=f_classif)), # feature selection
    ('classifier', GradientBoostingClassifier(random_state=1))
])

# Hyperparameter search
param_grid = [
    {
        'select__k': ['all', 10, 20],
        'classifier': [GradientBoostingClassifier(random_state=1)],
        'classifier__n_estimators': [100, 200, 500],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1],
    },
    
    {
        'select__k': ['all', 10, 20],
        'classifier': [RandomForestClassifier(class_weight='balanced', random_state=1)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20],
        'classifier__max_features': ['sqrt', 'log2', None],
    },
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(pipeline, param_grid, scoring='accuracy', 
                                   cv=cv, n_iter=10, n_jobs=-1, random_state=42,
                                   error_score='raise', verbose=1
                                  )
random_search.fit(X, y)

# Evaluate the model
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy Score:", random_search.best_score_)

best_model = random_search.best_estimator_

 

# Fit the best model on the entire training data
best_model.fit(X, y)

# Make predictions (if needed)
predictions = best_model.predict(X)

# Print confirmation
print("Best model fitted on the entire training set.")

 

print(classification_report(y, predictions))

 

# Define class labels explicitly
class_labels = ['No Depression', 'Depression']

# Compute confusion matrix
conf_matrix = confusion_matrix(y, predictions)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

 

# Predict on test data
test_predictions = best_model.predict(test_data)

# Prepare submission
submission = pd.DataFrame({
    "id": test_ids,
    "Depression": test_predictions
})
submission.to_csv("submission_project.csv", index=False)
print("Submission saved as 'submission_project.csv'.")

 

# def predict_interactively(best_model, pipeline, feature_names):
#     # Example user input
#     user_input = {
#         "Age": 30,
#         "City": "Mumbai",
#         "Profession": "Teacher",
#         "Sleep Duration": "6-8 hours",
#         "Dietary Habits": "Healthy",
#         "Degree": "Undergraduate",
#         # Add other features here...
#     }

#     # Convert user input to a DataFrame
#     user_df = pd.DataFrame([user_input])

#     # Ensure input DataFrame has all required columns
#     user_df = user_df.reindex(columns=feature_names, fill_value=0)  # Replace 0 with other defaults if needed

#     # Make prediction using the fitted pipeline
#     prediction = pipeline.predict(user_df)

#     return f"Predicted Depression Status: {'Yes' if prediction[0] == 1 else 'No'}"

# # Define feature names based on the training data
# feature_names = X.columns.tolist()

# # Ensure the pipeline is fitted
# pipeline.fit(X, y)

# # Use the function
# result = predict_interactively(best_model, pipeline, feature_names)
# print(result)