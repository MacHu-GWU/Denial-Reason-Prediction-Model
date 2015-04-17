##encoding=utf-8
##author=Samson Su
##date=2015-04-15

"""
Numeric type columns (3 columns)
--------------------------------
    ["Claim.Charge.Amount", "Subscriber.Payment.Amount", "Provider.Payment.Amount"]
    
    
Prior probability and its distribution
--------------------------------------
    Claim.Charge.Amount (0, 105530)
    Subscriber.Payment.Amount (0, 0)
    Provider.Payment.Amount (0, 880)
"""

from dev01_understand_the_data import read_data
from angora.DATA.js import load_js, safe_dump_js
from matplotlib import pyplot as plt
import numpy as np, pandas as pd

df = read_data()
class_label = load_js("class_label.json", enable_verbose=False)
df["_class"] = class_label

def numeric_column_correlation_analysis():
    """plot trends
    """
    for column_name in ["Claim.Charge.Amount", "Subscriber.Payment.Amount", "Provider.Payment.Amount"]:
        index0 = df["_class"] == 0 # 470588 entries
        index1 = df["_class"] == 1 # 1971 entries
    
        fig = plt.figure(figsize=[20, 10])
        ax = fig.add_subplot(1,1,1)
        
        line1, = ax.plot(df[column_name][index0], df["_class"][index0], "b.") # 470588 entries
        line2, = ax.plot(df[column_name][index1], df["_class"][index1], "r.") # 1971 entries
        
        ax.set_xlabel(column_name)
        ax.set_ylabel("Yes/No? 1/0")
        ax.set_title("%s vs _class" % column_name)
        
        _max, _min = df[column_name].max(), df[column_name].min()
        ax.set_xlim([_min - 0.05 * (_max - _min),
                     _max + 0.05 * (_max - _min),])
        
        ax.set_ylim([-0.5, 1.5])
        ax.grid()
        ax.legend([line1, line2], ["No", "Yes"])
        plt.savefig(r"image\numeric_column_vs_Class\%s_vs_class.png" % column_name)
    
    plt.close("all")

def plot_histgram():
    """find out the distribution
    Claim.Charge.Amount (0, 105530)
    Subscriber.Payment.Amount (0, 0)
    Provider.Payment.Amount (0, 880)
    """
    index0 = df["_class"] == 0 # 470588 entries
    index1 = df["_class"] == 1 # 1971 entries
    
    fig = plt.figure(figsize=[20, 10])
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    
    n1, bins1, patches1 = ax1.hist(df["Claim.Charge.Amount"][index1].values, bins=20)
    ax1.set_title("Claim.Charge.Amount value histgram when class = 1")
    n2, bins2, patches2 = ax2.hist(df["Subscriber.Payment.Amount"][index1].values, bins=20)
    ax2.set_title("Subscriber.Payment.Amount value histgram when class = 1")
    n3, bins3, patches3 = ax3.hist(df["Provider.Payment.Amount"][index1].values, bins=20)
    ax3.set_title("Provider.Payment.Amount value histgram when class = 1")
    
    plt.savefig(r"image\numeric_column_vs_Class\numeric_column_histgram.png")
    plt.close("all")
    
    print(bins1)
    print(bins2)
    print(bins3)

def find_posterier_probability():
    """see if we can use numeric columns to build a probability classifier
    Claim.Charge.Amount (0, 105530)
    Subscriber.Payment.Amount (0, 0)
    Provider.Payment.Amount (0, 880)
    """
    global df
    df = df[df["Claim.Charge.Amount"] >= 0]
    df = df[df["Claim.Charge.Amount"] <= 105530]

    df = df[df["Subscriber.Payment.Amount"] == 0]

    df = df[df["Provider.Payment.Amount"] >= 0]
    df = df[df["Provider.Payment.Amount"] <= 880]
    print(sum(df["_class"]) * 1.0 / len(df["_class"])) # 0.004368, poor result
    
if __name__ == "__main__":
    numeric_column_correlation_analysis()
    plot_histgram()
    find_posterier_probability()
    
#     index1 = df["_class"] == 1 # 1971 entries
#     c1 = df["Claim.Charge.Amount"][index1]
#     c2 = df["Subscriber.Payment.Amount"][index1]
#     c3 = df["Provider.Payment.Amount"][index1]
#     print(c1.min(), c1.max())
#     print(c2.min(), c2.max())
#     print(c3.min(), c3.max())
#     
#     c3.sort()
#     for i in c3:
#         print(i)