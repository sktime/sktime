import pandas as pd
from sktime.utils import all_estimators
#List of columns in the table
df_columns=['Classifier Category','Classifier Name','multivariate','unequal_length','missing_values','train_estimate','contractable']
#creates dataframe as df
df = pd.DataFrame([], columns=df_columns)
#Loop through all the classifier 
for classiName, classiClass in all_estimators(estimator_types="classifier"):
    # print("Categories")
    # print(str(classiClass).split('.')[2])
    
    category=str(classiClass).split(".")[2]
    try:
        # capabilites of each classifier
        cap_dict=classiClass.capabilities
        #print(cap_dict["multivariate"])
        multivariate=str(cap_dict["multivariate"])
        unequal_length=str(cap_dict["unequal_length"])
        missing_values=str(cap_dict["missing_values"])
        train_estimate=str(cap_dict["train_estimate"])
        contractable=str(cap_dict["contractable"])
        df = df.append({'Classifier Category': category, 'Classifier Name': classiName, 'multivariate': multivariate,'unequal_length':unequal_length,'missing_values':missing_values,'train_estimate':train_estimate,'contractable':contractable}, ignore_index=True)
    except:
         print(classiName + " - No Capabilites Given")
    
df.to_html("Classifier_Capabilities.html", index=False, escape=False)   

