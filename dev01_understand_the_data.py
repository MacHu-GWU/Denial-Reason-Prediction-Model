##encoding=utf-8
##author=Samson Su
##date=2015-04-15

"""
All columns (28 columns)
------------------------
    ["Claim.Number", "Claim.Line.Number", "Member.ID", "Provider.ID", 
     "Line.Of.Business.ID", "Revenue.Code", "Service.Code", "Place.Of.Service.Code", 
     "Procedure.Code", "Diagnosis.Code", "Claim.Charge.Amount", "Denial.Reason.Code", 
     "Price.Index", "In.Out.Of.Network", "Reference.Index", "Pricing.Index", 
     "Capitation.Index", "Subscriber.Payment.Amount", "Provider.Payment.Amount", 
     "Group.Index", "Subscriber.Index", "Subgroup.Index", "Claim.Type", 
     "Claim.Subscriber.Type", "Claim.Pre.Prince.Index", "Claim.Current.Status", 
     "Network.ID", "Agreement.ID"]
     
Numeric type columns (3 columns)
--------------------------------
    ["Claim.Charge.Amount", "Subscriber.Payment.Amount", "Provider.Payment.Amount"]
    
Category type columns, exclude the class column (17 columns)
------------------------------------------------------------
    ["Revenue.Code", "Service.Code", "Procedure.Code", "Diagnosis.Code", 
    "Price.Index", "In.Out.Of.Network", "Reference.Index",
    "Capitation.Index", "Group.Index", "Subscriber.Index", "Subgroup.Index",
    "Claim.Type", "Claim.Subscriber.Type", "Claim.Pre.Prince.Index",
    "Claim.Current.Status", "Network.ID", "Agreement.ID"]

Useless columns
---------------
    ["Claim.Number", "Claim.Line.Number", "Member.ID", "Provider.ID", ... etc]

- 472559 rows, 28 columns
- 1971 entries: class = 1, 0.42%
- 259 distinct Denial.Reason.Code
"""

from angora.DATA.js import load_js, safe_dump_js
import numpy as np, pandas as pd

def read_data():
    df = pd.read_csv("claim.sample.csv", index_col=0,
                     dtype={
                        "Revenue.Code": str,
                        "Revenue.Code": str, 
                        "Service.Code": str, 
                        "Procedure.Code": str, 
                        "Diagnosis.Code": str, 
                        "Price.Index": str, 
                        "In.Out.Of.Network": str, 
                        "In.Out.Of.Network": str, 
                        "Reference.Index": str,
                        "Capitation.Index": str, 
                        "Group.Index": str, 
                        "Subscriber.Index": str, 
                        "Subgroup.Index": str,
                        "Claim.Type": str, 
                        "Claim.Subscriber.Type": str, 
                        "Claim.Pre.Prince.Index": str,
                        "Claim.Current.Status": str, 
                        "Network.ID": str, 
                        "Agreement.ID": str,
                        })
    return df

def understand_the_data_and_generate_the_class_label():
    df = read_data()
    code_we_care = set(["F13", "J8G", "JO5", "JB8", "JE1", "JC9", "JF1", "JF9", "JG1", "JPA", "JES"])
    class_label = list()
    for code in df["Denial.Reason.Code"]:
        if code in code_we_care:
            class_label.append(1)
        else:
            class_label.append(0)
            
    print("%s rows, %s columns." % df.shape)
    print("%s rows class = 1" % sum(class_label))
    print("%s distinct Denial.Reason.Code" % len(df["Denial.Reason.Code"].unique()))
    
    safe_dump_js(class_label, "class_label.json", enable_verbose=False)
    
if __name__ == "__main__":
    understand_the_data_and_generate_the_class_label()