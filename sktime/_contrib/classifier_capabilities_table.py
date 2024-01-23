"""Auto-generate a classifier capabilities summary."""
import pandas as pd

from sktime.registry import all_estimators

# List of columns in the table
df_columns = [
    "Classifier Category",
    "Classifier Name",
    "multivariate",
    "unequal_length",
    "missing_values",
    "train_estimate",
    "contractable",
]
# creates dataframe as df
df = pd.DataFrame([], columns=df_columns)
# Loop through all the classifiers
for classiName, classiClass in all_estimators(estimator_types="classifier"):
    category = str(classiClass).split(".")[2]
    try:
        # capabilities of each of the classifier classifier
        cap_dict = classiClass.capabilities
        multivariate = str(cap_dict["multivariate"])
        unequal_length = str(cap_dict["unequal_length"])
        missing_values = str(cap_dict["missing_values"])
        train_estimate = str(cap_dict["train_estimate"])
        contractable = str(cap_dict["contractable"])
        # Adding capabilities for each classifier in the table
        record = {
            "Classifier Category": category,
            "Classifier Name": classiName,
            "multivariate": multivariate,
            "unequal_length": unequal_length,
            "missing_values": missing_values,
            "train_estimate": train_estimate,
            "contractable": contractable,
        }
    except AttributeError:
        record = {
            "Classifier Category": category,
            "Classifier Name": classiName,
            "multivariate": "N/A",
            "unequal_length": "N/A",
            "missing_values": "N/A",
            "train_estimate": "N/A",
            "contractable": "N/A",
        }
    df = pd.concat([df, pd.DataFrame(record, index=[0])], ignore_index=True)
df.to_html("Classifier_Capabilities.html", index=False, escape=False)
