##encoding=utf-8
##author=Samson Su
##date=2015-04-15

"""
Category type columns, exclude the class column (17 columns)
------------------------------------------------------------
    ["Revenue.Code", "Service.Code", "Procedure.Code", "Diagnosis.Code", 
    "Price.Index", "In.Out.Of.Network", "Reference.Index",
    "Capitation.Index", "Group.Index", "Subscriber.Index", "Subgroup.Index",
    "Claim.Type", "Claim.Subscriber.Type", "Claim.Pre.Prince.Index",
    "Claim.Current.Status", "Network.ID", "Agreement.ID"]
    
    ["Revenue.Code", "Service.Code", "Procedure.Code", "Diagnosis.Code", "Subscriber.Index"] are
    statistical significant
"""

from dev01_understand_the_data import read_data
from angora.DATA.js import load_js, safe_dump_js
import numpy as np, pandas as pd

df = read_data()
class_label = load_js("class_label.json", enable_verbose=False)
df["_class"] = class_label

def category_type_column_significant_analysis():
    """see result in significant.xlsx
    """
    writer = pd.ExcelWriter("significant.xlsx")
    for column_name in ["Revenue.Code", "Service.Code", "Procedure.Code", "Diagnosis.Code", 
                        "Price.Index", "In.Out.Of.Network", "Reference.Index",
                        "Capitation.Index", "Group.Index", "Subscriber.Index", "Subgroup.Index",
                        "Claim.Type", "Claim.Subscriber.Type", "Claim.Pre.Prince.Index",
                        "Claim.Current.Status", "Network.ID", "Agreement.ID"]:
        print("analyzing %s" % column_name)
        
        result = list()
        
        for choice in df[column_name].unique():
            sub_df = df[df[column_name] == choice]["_class"]
            total = len(sub_df)
            num_of_yes = sum(sub_df)
            probability = num_of_yes * 1.0 / total
            result.append([choice, probability, num_of_yes, total])
            
        result = sorted(result, key=lambda x: x[1], reverse=True) # sort by probability
        result = pd.DataFrame(result, columns=["key", "probability", "num_of_yes", "total"])
        result.to_excel(writer, column_name, index=False)
    
    writer.save()

if __name__ == "__main__":
    category_type_column_significant_analysis()