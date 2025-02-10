import json

with open("out.json", "r") as fdata:
    dbdata = json.load(fdata)

dbdata = [entry for entry in dbdata if entry["completed_test"]]

with open("out.json", "w") as fdata:
    json.dump(dbdata, fdata, indent=4)