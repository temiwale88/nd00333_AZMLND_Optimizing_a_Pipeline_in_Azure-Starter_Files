from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"


def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    return x_df, y_df 

def main():     
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--solver', type=str, default='lbfgs', help="Algorithm to use in the optimization problem")
    parser.add_argument('--penalty', type=str, default='l2', help="Used to specify the norm used in the penalization")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Penalty:", args.penalty)
    run.log("Solver for optimization:", args.solver)

    from azureml.core import Dataset
    dataUrl = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'
    
    ds = TabularDatasetFactory.from_delimited_files(path = dataUrl)

    x, y = clean_data(ds)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter, penalty=args.penalty, solver=args.solver).fit(x_train, y_train)

    # save model
    os.makedirs('./outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(model, './outputs/model.joblib')
      
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy)) #source: https://bit.ly/3mTxEWR && https://bit.ly/3hgonXx
  
    y_pred = model.predict(x_test)
    auc_weighted = roc_auc_score(y_pred, y_test, average='weighted')
    run.log("AUC_weighted", np.float(auc_weighted)) #source: https://bit.ly/3mTxEWR && https://bit.ly/3hgonXx
    
    # creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

# Run main here not in any other module (e.g. ipynb) it's called
if __name__ == '__main__':
    main()