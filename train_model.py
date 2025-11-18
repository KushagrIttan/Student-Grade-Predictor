import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

def train_advanced_model():
    """
    Trains an advanced RandomForestRegressor model on the student performance dataset,
    including feature engineering, and saves the entire pipeline.
    """
    try:
        data = pd.read_csv('student-mat.csv', sep=';')
    except FileNotFoundError:
        print("Error: 'student-mat.csv' not found.")
        return

    # Define features to use
    numerical_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                          'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 
                          'absences', 'G1', 'G2']
    
    categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                            'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 
                            'activities', 'nursery', 'higher', 'internet', 'romantic']

    features = numerical_features + categorical_features
    target = 'G3'

    X = data[features]
    y = data[target]

    # Create a preprocessor which one-hot encodes categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='passthrough')

    # Create the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split data and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model (optional, but good practice)
    score = model_pipeline.score(X_test, y_test)
    print(f"Model R^2 score on test set: {score:.4f}")

    # Save the entire pipeline to a single file
    with open('student_model.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)

    print("Advanced model pipeline trained and saved as 'student_model.pkl'")

if __name__ == '__main__':
    train_advanced_model()