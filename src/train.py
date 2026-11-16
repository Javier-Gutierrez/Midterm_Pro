#Libraries
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


#Paths
DATA_PATH = Path("data/application_train.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "model_pipeline.joblib"


def train():
    df = pd.read_csv(DATA_PATH)

    #Features

    numerical = ["AMT_INCOME_TOTAL",
                 "AMT_CREDIT",
                 "AMT_ANNUITY",
                 "DAYS_BIRTH",
                 "DAYS_EMPLOYED",
                 "EXT_SOURCE_1",
                 "EXT_SOURCE_2",
                 "EXT_SOURCE_3"
                 ]
    categorical = ["NAME_CONTRACT_TYPE",
                   "CODE_GENDER",
                   "FLAG_OWN_CAR"]

    target = "TARGET"

    X = df[numerical+categorical]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

    #Cleaning and scaling
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
            ]
    )

    cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical),
            ("cat", cat_transformer, categorical)
            ]
    )

    #Merging preprocess and Logistic Regression
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]
    )

    model_pipeline.fit(X_train, y_train)

    #Predictions
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred = model_pipeline.predict(X_test)

    #Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\n Clasification Report")
    print(classification_report(y_test, y_pred))

    #Saving model
    joblib.dump(model_pipeline, MODEL_PATH)

if __name__ == "__main__":
    train()

