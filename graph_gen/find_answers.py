import graph_tool.all as gt 
import numpy as np
import random
from collections import Counter

def apsp(G,weights=None):
    d = np.array( [v for v in gt.shortest_distance(G,weights=weights)] ,dtype=float)
    return d

def count_degree(G: gt.Graph, name="none", target=3):
    candidates = list()
    for v in G.vertices():
        num = 0        
        deg, num = Counter([u.out_degree() for u in v.all_neighbors()]).most_common()[0]
        if 2 <= deg <= 6 and 2 <= num <= 8:
            candidates.append((int(v), num, deg))

    if len(candidates) < 1: 
        print("Could not find candidates for degree")
        return None, None

    v, num, target = candidates[random.randint(0,len(candidates)-1)]
    
    q_text = f"How many neighbors of the highlighted node have degree {target}?"
    return (v,num), q_text

def common_neighbors(G: gt.Graph, name="none", target=2):
    candidates = list() 
    for v in G.vertices():
        for u in G.vertices():
            if v == u: continue 
            n1 = {int(n) for n in v.all_neighbors()}
            n2 = {int(n) for n in u.all_neighbors()}
            intersect = n1.intersection(n2)
            if len(intersect) == target: 
                candidates.append((int(u), int(v)))
    if len(candidates) < 1: return None, None

    q_text = "How many neighbors do the two nodes have in common?"
    return candidates[random.randint(0,len(candidates)-1)], q_text

def shortest_path(G:gt.Graph, name="none", target=3):
    candidates = list()
    d = apsp(G)
    for u in G.iter_vertices():
        for v in range(u):
            if d[u,v] == target: candidates.append((u,v))

    q_text = "What is the length of the shortest path between the two highlighted nodes?"
    return candidates[random.randint(0,len(candidates)-1)], q_text  

def estimate_size(G:gt.Graph, name):
    width = 100
    if random.random() < 0.5: 
        q_text = "About how many edges are in the graph?"
        answer = G.num_edges() // width
    else: 
        q_text = "About how many nodes are in the graph?"
        answer = G.num_vertices() // width

    correct_response = [max(0, width*answer-(width // 2)), width*(answer+1)-(width // 2)]

    tvar = random.random()
    if tvar < 0.25: 
        all_answers = [[i*width + correct_response[0], i*width + correct_response[1]] for i in range(4)]
        correct = 1
    elif tvar < 0.5:
        all_answers = [[i*width + correct_response[0], i*width + correct_response[1]] for i in range(4)]
        correct = 2
    elif tvar < 0.75: 
        all_answers = [[i*width + correct_response[0], i*width + correct_response[1]] for i in range(4)]
        correct = 3
    else: 
        all_answers = [[i*width + correct_response[0], i*width + correct_response[1]] for i in range(4)]
        correct = 4

    

    obj = {
        "q_type": "T4", 
        "q_text": q_text, 
        "q_options": dict(zip(
            [f"option-{n+1}" for n in range(4)],
            ["-".join(map(str, ans)) for ans in all_answers]
        )), 
        "graph": {
            "graph_type": "eucldiean", 
            "graph_id": name
        }, 
        "node": [],
        "q_answer": f"option-{correct}"
    }

    return obj

def find_mode(G, name):
    q_text = "What is the most frequent degree of the highlighted nodes?"

    while True: 
        v = random.randint(0,G.num_vertices()-1)
        visited = [v]
        next_Q = [u for u in G.iter_all_neighbors(v)]
        for _ in range(4):
            Q = next_Q
            next_Q = list()
            while Q: 
                u = Q.pop() 
                visited.append(u)
                
                for w in G.iter_all_neighbors(u):
                    if w not in visited:
                        next_Q.append(w)
        
        highlighted = [v for v in visited][:random.randint(3,7)]
        if len(highlighted) < 3: continue

        degrees = G.get_total_degrees(highlighted)
        from collections import Counter
        count_degrees = Counter(degrees)
        answers = count_degrees.most_common(4)
        
        same_counts = {ans[0] for ans in answers if ans[1] == answers[0][1]}
        # if len(answers) < 2: continue
        # if answers[0][1] != answers[1][1]: break
        break


    answers = [answers[0][0]]
    correct_ans = answers[0]    
    while len(answers) < 4:
        num = random.randint(max(answers[0]-5, 0),answers[0]+5)
        if num not in answers and num not in same_counts: answers.append(num)

    option_set = sorted([int(a) for a in answers])
    correct = option_set.index(correct_ans) + 1

    obj = {
        "q_type": "T5", 
        "q_text": q_text, 
        "q_options": dict(zip(
            [f"option-{n+1}" for n in range(4)],
            option_set
        )), 
        "graph": {
            "graph_type": "eucldiean", 
            "graph_id": name
        }, 
        "node": [f"node_{v}" for v in highlighted], 
        "q_answer": f"option-{correct}"
    }
    
    return obj
    
def find_bfs(G, name):
    depth = random.randint(2,3)

    while True:

        v = random.randint(0,G.num_vertices()-1)
        visited = {v}
        next_Q = [u for u in G.iter_all_neighbors(v)]

        for _ in range(depth):
            Q = next_Q
            next_Q = list()
            while Q: 
                u = Q.pop() 
                visited.add(u)
                
                for w in G.iter_all_neighbors(u):
                    if w not in visited:
                        next_Q.append(w)
        print("hello")
        if len(visited)-1 < 12: break
        

    q_text = f"How many nodes are reachable in {depth} steps from the highlighted node?"

    answers = [len(visited) - 1]
    while len(answers) < 4:
        num = random.randint(max(answers[0]-5, 0),answers[0]+5)
        if num not in answers: answers.append(num)

    option_set = sorted([int(a) for a in answers])
    correct = option_set.index(len(visited) - 1) + 1

    obj = {
        "q_type": "T6", 
        "q_text": q_text, 
        "q_options": dict(zip(
            [f"option-{n+1}" for n in range(4)],
            option_set
        )), 
        "graph": {
            "graph_type": "eucldiean", 
            "graph_id": name
        }, 
        "node": [f"node_{v}"], 
        "q_answer": f"option-{correct}"
    }

    return obj

def random_option_set(correct):
    if isinstance(correct, bool):
        return ["Yes", "No"], 1 if correct else 2
    tvar = random.random() 
    if tvar < 0.25: 
        return [correct, correct+1, correct+2, correct+3], 1
    if tvar < 0.5:
        return [correct-1, correct, correct+1, correct+2], 2
    elif tvar < 0.75:
        return [correct-2, correct-1, correct, correct+1], 3
    else:
        return [correct-3, correct-2, correct-1, correct], 4

def append_question_node(func, G, target, name, q_type):
    if isinstance(target, bool):
        node, q_text = func(G,target=target)
    else:
        node = None 
        while node is None:     
            node, q_text = func(G,target=target)
            if node is None: 
                target -= 1
            if target < 1: return None
    
    if "degree" in q_text:
        node,target = node
    
    option_set, correct = random_option_set(target)
    if isinstance(option_set[0], int): 
        while any(op < 1 for op in option_set) or \
                (correct == 1 and "length" in q_text) or \
                (correct == 1 and "degree" in q_text):
            print(q_text)
            print(option_set, correct, target)
            option_set, correct = random_option_set(target)

    node_obj = [f"node_{node}"] if isinstance(node, int) \
        else [f"node_{node[0]}", f"node_{node[1]}"]

    obj = {
        "q_type": q_type, 
        "q_text": q_text, 
        "q_options": dict(zip(
            [f"option-{n+1}" for n in range(4)],
            option_set
        )), 
        "graph": {
            "graph_type": "eucldiean", 
            "graph_id": name
        }, 
        "node": node_obj, 
        "q_answer": f"option-{correct}"
    }

    return obj

def find_questions(G, name):
    questions = [
        # append_question_node(count_degree, G, 3, name, "T1"), 
        append_question_node(count_degree, G, random.randint(2,8), name, "T1"),
        append_question_node(common_neighbors, G, random.randint(1,8), name, "T2"), 
        # append_question_node(common_neighbors, G, True, name, "T2"), 
        append_question_node(shortest_path, G, random.randint(3,7), name, "T3"), 
        # append_question_node(shortest_path, G, random.randint(3,7), name, "T3"), 
        estimate_size(G, name), 
        find_mode(G, name), 
        find_bfs(G, name)
    ] 

    print(f"I think this is true: {[1 if q is not None else 0 for q in questions]}")
    questions = [q for q in questions if q is not None]

    return questions


def gen_questions():
    import json
    
    js = {'questions': list()}
    for c in ["e", "h", "s"]:
        for n in range(3):
            with open(f"../src/application/static/data/{c}_group_{n}.json", "r") as fdata:
                gdata = json.load(fdata)
            
            name = f"{c}_group_{n}"
            G = gt.Graph(directed=False)
            G.add_vertex(len(gdata["nodes"]))
            G.add_edge_list([(e["source"], e["target"]) for e in gdata["links"]])
            
            js["questions"].extend(
                find_questions(G, name)
            )

    for i, question in enumerate(js["questions"]):
        question["question_id"] = f"q_{i}"

    with open("../src/application/static/data/test-questions2.json", 'w') as fdata:
        json.dump(js, fdata, indent=4)

def replace_questions():
    task_dict = {
        "T1": lambda G, name : append_question_node(count_degree, G, random.randint(2,8), name, "T1"), 
        "T2": lambda G,name : append_question_node(common_neighbors, G, random.randint(1,8), name, "T2"),
        "T3": lambda G,name : append_question_node(shortest_path, G, random.randint(3,7), name, "T3"), 
        "T4": lambda G,name : estimate_size(G,name),
        "T5": lambda G,name : find_mode(G,name), 
        "T6": lambda G,name : find_bfs(G,name)
    }

    import json 
    with open("../src/application/static/data/test-questions-2.json", "r") as fdata:
        questions = json.load(fdata)

    for i in range(len(questions)):
        for j in range(3):
            old_question = questions[i][j]
            if old_question["graph"]["graph_id"] == "e_group_2": 

                with open(f"../src/application/static/data/{old_question['graph']['graph_id']}.json", 'r') as fdata:
                    gdata = json.load(fdata)
                G = gt.Graph(directed=False)
                G.add_vertex( len(gdata["nodes"]) )
                G.add_edge_list( [(e["source"], e["target"]) for e in gdata["links"]] )

                new_question = task_dict[ old_question["q_type"] ](G, "e_group_2")
                
                new_question["question_id"] = old_question["question_id"]

                questions[i][j] = new_question

    with open("../src/application/static/data/test-questions-2.json", "w") as fdata:
        json.dump(questions, fdata, indent=4)

if __name__ == "__main__":
    # import json 
    # with open("../src/application/static/data/s_group_2.json", 'r') as fdata:
    #     gdata = json.load(fdata)
    # print(len( gdata['links'] ))
    replace_questions()