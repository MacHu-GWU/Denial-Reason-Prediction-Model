##encoding=utf-8
##author=Samson Su
##date=2015-04-15

from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
import numpy as np, pandas as pd
import random

df = pd.read_csv("train.txt")

def classify():
    """algorithm evaluation, parameter optimization
    """
    yes_dataset = df[df["_class"] == 1] # 470588
    no_dataset = df[df["_class"] == 0] # 1971

    parameter_analysis = list()
    for criterion in np.arange(0.05, 0.91, 0.05):
        print("doing experiment at criterion = %s ..." % criterion)
        rate_list = list()
        for i in range(10):
            # shuffle yes_dataset and no_dataset, so we can randomly choose 90% yes_dataset
            # + 90% no_dataset as train dataset, 10% yes_dataset + 10% no_dataset as test dataset
            yes_index = yes_dataset.index.tolist()
            random.shuffle(yes_index)
            no_index = no_dataset.index.tolist()
            random.shuffle(no_index)
            
            # concatenate 90%yes + 90%no, 10%yes + 10%no
            train = pd.concat([
                        yes_dataset.loc[yes_index[:1774], :],
                        no_dataset.loc[no_index[:423530], :]
                        ])
            test = pd.concat([
                        yes_dataset.loc[yes_index[1774:], :],
                        no_dataset.loc[no_index[423530:], :]
                        ]) 
            
            # split data and label
            train_data, train_label = (train[["Revenue.Code", 
                                              "Service.Code", 
                                              "Procedure.Code", 
                                              "Diagnosis.Code", 
                                              "Subscriber.Index"]], 
                                       train["_class"])
            test_data, test_label = (test[["Revenue.Code", 
                                           "Service.Code", 
                                           "Procedure.Code", 
                                           "Diagnosis.Code", 
                                           "Subscriber.Index"]], 
                                     test["_class"])
            
            # apply classifier
            clf = GaussianNB()
            clf.fit(train_data, train_label)
            probability = clf.predict_proba(test_data).T[1]
        
            result = pd.DataFrame()
            result["_class"] = test_label
            result["_predict"] = probability >= criterion
            
            result_yes = result[result["_class"] == 1]
            yes_yes_rate = sum(result_yes["_class"] == result_yes["_predict"])/len(result_yes["_predict"])
            
            result_no = result[result["_class"] == 0]
            no_no_rate = sum(result_no["_class"] == result_no["_predict"])/len(result_no["_predict"])
            
            rate_list.append((yes_yes_rate, no_no_rate))

        rate_list = pd.DataFrame(rate_list)
        yes_yes_rate, no_no_rate = rate_list.mean()[0], rate_list.mean()[1]
        parameter_analysis.append((criterion, yes_yes_rate, no_no_rate))
    
    # save data to excel spreadsheet
    parameter_analysis = pd.DataFrame(parameter_analysis, columns=["criterion", "yes_yes_rate", "no_no_rate"])
    writer = pd.ExcelWriter("parameter_analysis.xlsx")
    parameter_analysis.to_excel(writer, "parameter_analysis", index=False)
    writer.save()

if __name__ == "__main__":
    classify()
