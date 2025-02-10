import json 
import pandas as pd

data = pd.read_csv("out.csv", index_col=0)

"""
Data structure: 
{
    p1: {
        "Geom": {
            "E": {
                "gtype": {
                    "e": {
                        "task": {
                            "T1": {
                                "accuracy": list[int], 
                                "time": list[float], 
                                "effort": list[float]
                            }
                        }
                    }
                    "h": {
                        
                    }
                    "s": {
                        
                    }                                        
                }
            }
            "H": {
                
            }
            "S": {
                
            }
        }
    }
}
"""

def createObjectEntry():
    d = dict()
    for geom in "EHS":
        d[geom] = dict()
        for gtype in "ehs":
            d[geom][gtype] = dict()
            for task in range(1,7):
                d[geom][gtype][f"T{task}"] = {
                    "accuracy": list(),
                    "time": list(),
                    "effort": list()
                }
    return d


DS = list()
for pname, row in data.iterrows():
    pentry = createObjectEntry()
    for geom in "EHS":
        for qnum in range(18):
            stem = f"{geom}-q_{qnum}"
            gname = row[f"graph_{stem}"]
            gtype = gname[0]
            ttype = row[f"type_{stem}"]

            pentry[geom][gtype][ttype]["accuracy"].append(row[f"correct_{stem}"])
            pentry[geom][gtype][ttype]["time"].append(row[f"time_{stem}"] / 1000)            

            effort = row[f"panCount_{stem}"] + row[f"zoomCount_{stem}"] + row[f"dblclickCount_{stem}"]
            pentry[geom][gtype][ttype]["effort"].append(effort)
    DS.append(pentry)

with open("CHI-format.json", 'w') as fdata:
    json.dump(DS,fdata,indent=4)
