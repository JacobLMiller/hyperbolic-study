import json

def count_correct(data, q_lookup, id_lookup):

    response = [d for d in data if int(d["id"]) == id_lookup][0]

    correct = {"E": 0, "H": 0, "S": 0}

    for geom in ["E", "H", "S"]:
        for key, item in response["results"][geom].items():
            if item == q_lookup[key]["q_answer"]:
                correct[geom] += 1
    
    print(correct)


if __name__ == "__main__":
    with open("out.json", "r") as fdata:
        data = json.load(fdata)

    with open("../src/application/static/data/test-questions.json", 'r') as fdata:
        questions = json.load(fdata)["questions"]

    q_lookup = { d["question_id"]: d for d in questions }
    
    entry = data[-1]

    print(entry.keys())
    print(type(entry["results"]))
    print(entry["results"])

    import pandas as pd 

    df = pd.DataFrame(entry["results"])
