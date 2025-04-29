from flask import jsonify, render_template, request, redirect, url_for, session
from urllib.parse import unquote as urllib_unquote
from application import app
from werkzeug.utils import secure_filename
import json
import random
from datetime import datetime
from pymongo import MongoClient
import hsne_wrapper
import numpy as np
from flask import g 
import gc

from scipy.sparse import coo_matrix
# from mongopass import mongopass

TITLE = "Hyperbolic test site"

from enum import Enum
SURVEY = Enum("SURVEY", ["T1", "T2", "T3"])

HSNE = None
nlevels = 3
curlevel = 3

ffile = "blobs"



with open(f"application/static/data/hyperbolic/{ffile}.json", 'r') as fdata:
    CLASSES = [d['class'] for d in json.load(fdata)]

def getEuclideanData():
    if not hasattr(g, 'hsne'):
        print("Creating new HSNE instance for this request....")
        g.hsne = create_hsne_instance()

    X = g.hsne.getEmbedding().reshape((-1,2))
    labels = g.hsne.getIdxAtScale(nlevels-1)

    curlevel = nlevels

    jsdata = {
        "nodes": [{
            'id': int(lab), 
            'class': CLASSES[int(lab)],
            'index': i,
            "euclidean": {
                    "x": float(X[i,0]), 
                    "y": float(X[i,1])
                }
            } for i, lab in enumerate(labels)]
    }

    return jsdata

def create_hsne_instance(ffile='blobs'):
    data = np.load(f"application/static/data/raw_data/{ffile}.npy")
    npnts, ndims = data.shape
    print(npnts, ndims)
    hsne_instance = hsne_wrapper.PyHierarchicalSNE().initialize(npnts, ndims, nlevels, data.flatten())

    return hsne_instance

@app.route('/drilldown', methods=["POST"])
def drill_down():
    if not hasattr(g, 'hsne'):
        print("Creating new HSNE instance for this request....")
        g.hsne = create_hsne_instance()
    
    data = request.get_json()
    print(data)
    landmarks = np.array([int(d) for d in data['landmarks']], dtype=np.int32)
    # data = np.arange(15,dtype=np.uint32)
    
    print("Landmarks array dtype:", landmarks.dtype)
    print("Landmarks array shape:", landmarks.shape)
    print("Landmarks array contents:", landmarks)


    print(g.hsne)
    rows, columns, values, idxes = g.hsne.drillDownMatrix(int(data['scale'])-1, landmarks)

    n = max(max(rows), max(columns)) + 1
    mat = coo_matrix((values, (rows,columns)), shape=(n,n)).tocsr()
    print(mat)
    # init = initialization.pca(mat)

    # aff = affinity.PrecomputedAffinities(mat)

    # emb = TSNEEmbedding(
    #     init,
    #     aff, 
    #     # negative_gradient_method='fft',
    #     # n_jobs=8,
    #     verbose=True
    # )

    # print(emb)

    # X1 = emb.optimize(250, exaggeration=12)
    # X2 = X1.optimize(750,)

    from sklearn.manifold import TSNE
    P = mat.toarray()
    tsne = TSNE(metric='precomputed',n_components=2,perplexity=30,init='random')
    # tsne.learning_rate_ = 200.0
    X2 = tsne.fit_transform(1-P)

    print(X2)

    labels = g.hsne.getIdxAtScale(int(data['scale']-2))

    print(labels)

    jsdata = {
        "nodes": [
            {
                "id": int(idxes[i]), 
                "index": int(idxes[i]), 
                "class": CLASSES[int(labels[int(idxes[i])])],
                "euclidean": {
                    "x": float(X2[i,0]),
                    "y": float(X2[i,1])
                }
            }
        for i in range(X2.shape[0])]
    }

    # print(jsdata)
    # curlevel = curlevel - 1

    return jsonify({"message": "Data recieved", "status": "success", "data": jsdata}), 200

def getHyperbolicData():
    with open(f"application/static/data/hyperbolic/{ffile}.json", 'r') as fdata:
        data = json.load(fdata)

    jsdata = {"nodes": data}

    return jsdata




IS_LIVE = app.config.get("LIVE_VERSION")
prefix = "application" if IS_LIVE else "src/application"

with open("application/static/data/test-questions-2.json", "r") as qdata:
    Questions = json.load(qdata)
    # random.shuffle(Questions)
    for q_arr in Questions:
        for q in q_arr:
            del q["q_answer"]

    Q_groups = [
        [q for q in Questions if q[0]["q_type"] in ["T1", "T3"]], 
        [q for q in Questions if q[0]["q_type"] in ["T2", "T5"]], 
        [q for q in Questions if q[0]["q_type"] in ["T4", "T6"]]
    ]

# All Questions
NUM_QUESTIONS = len(Q_groups[0]) 
# NUM_QUESTIONS = len(Questions)   
# print(NUM_QUESTIONS)

# For pilot
# NUM_QUESTIONS = 5



vis_acronym = {
    "E": "euclidean",
    "H": "hyperbolic", 
    "S": "sphere"
}

vis_order = ["E", "H", "S"]

def generate_id(id):
    # id += str(riemannCollection.count_documents({}))
    return random.randint(0,500)
    # return str(riemannCollection.count_documents({}))

def get_graph(id):
    floc = f"application/static/data/{id}.json"
    if IS_LIVE: floc = floc.replace("src/", "")
    with open(floc, 'r') as fdata:
        gdata = json.load(fdata)    

    return gdata

GRAPHS = {f"{geom}_group_{num}": get_graph(f"{geom}_group_{num}") for geom in "ehs" for num in range(3)}

def get_question_full(index, ind):
    question = Questions[int(index)][int(ind)]
    graph_id = question["graph"]["graph_id"]

    return GRAPHS[graph_id], question

def get_question(index, q_group, q_num):
    print( int(q_group), int(index) )
    if int(q_group) == -1:
        question = Questions[int(index)][ q_num ]
        graph_id = question["graph"]["graph_id"]
        return GRAPHS[graph_id], question 

    question = Q_groups[int(q_group)][int(index)][ q_num ]
    print(question)
    graph_id = question["graph"]["graph_id"]
    return GRAPHS[graph_id], question

def balanced_latin_square(n):
    """
    Generates a balanced latin square in n variables. 
    Taken from https://medium.com/@graycoding/balanced-latin-squares-in-python-2c3aa6ec95b9
    """
    l = [[((j//2+1 if j%2 else n-j//2) + i) % n for j in range(n)] for i in range(n)]
    if n % 2:  # Repeat reversed for odd n
        l += [seq[::-1] for seq in l]
    return l    

LATIN_SQUARE = balanced_latin_square( NUM_QUESTIONS )
GEOM_ORDER   = balanced_latin_square( 3 )

@app.template_filter('unquote')
def unquote(url):
    safe = app.jinja_env.filters['safe']
    return safe(urllib_unquote(url))

@app.route('/')
@app.route('/test_type')
def index():
    session["test_type"] = 0
    return render_template('choose.html', title=TITLE, data=None)

@app.route("/test_type<v1>")
def test_type(v1):
    print(v1)
    return render_template('consent.html', title=TITLE, data=None)


@app.route('/choose')
def choose():
    geom = ["E", "H", "S"]
    # n = int(riemannCollection.count_documents({})) % 3
    n = random.randint(0,2)
    print(request.form.get("id"))
    return redirect(url_for("user_index_form", id=geom[n]))
    #return render_template('choose.html', title='non-Euclidean Graph Survey Homepage (pilot!)', data=None)

@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/homepage<data>')
def homepage(data):
    id = request.args.get('id')
    G = get_graph("sample")
    data = {
        "finished": data, 
        "graph": G, 
        "geom": session["current_geom"], 
        "first_time": True if session["geom_num"] == 0 else False
    }
    return render_template("homepage.html", title=TITLE, data=data, id=id)

@app.route('/test<id>_<q>') 
def view(id, q):
    if "current_geom" not in session: pass
    elif id and session["current_geom"] in ["E", "H", "S"]:
        vis_type = vis_acronym[session["current_geom"]]
        gdata, question = get_question(q, session["q_group"], session["geom_num"])
        gdata["geom"] = session["current_geom"]
        print(f"geom number is {session['geom_num']}")
        return render_template("visualization.html", 
            title=TITLE, 
            data=gdata, 
            id=id, 
            q_id=question["question_id"], 
            question=question,
            progress=[session["geom_num"] * NUM_QUESTIONS + session["cur_index"], 3 * NUM_QUESTIONS]
        )
    return redirect(url_for("index"))

@app.route('/hyperbolic')
def hyperbolic():
    gdata, question = get_question(0, 0)
    gdata["geom"] = "H"
    return render_template("visualization.html", 
        title=TITLE,
        data=gdata,
        id="test-hyperbolic",
        q_id=0,
        question=question,
        progress=[0,1]
    )

@app.route('/spherical')
def spherical():
    gdata, question = get_question(0, 0)
    gdata["geom"] = "S"
    return render_template("visualization.html", 
        title=TITLE,
        data=gdata,
        id="test-spherical",
        q_id=0,
        question=question,
        progress=[0,1]
    )    

@app.route('/euclidean<qid>_<ind>')
@app.route('/euclidean', defaults={'qid': None, 'ind': None})
def euclidean(qid, ind):
    if qid: 
        gdata, question = get_question_full(int(qid), int(ind))
    else: 
        qid = 0
        gdata, question = get_question(0, 0)
    gdata["geom"] = "E"
    return render_template("visualization.html", 
        title=TITLE,
        data=gdata,
        id="test-euclidean",
        q_id=qid,
        question=question,
        progress=[0,1]
    )    


@app.route('/<id>')
def user_index(id):
    if "E" in id:
        return redirect(url_for("euc_view_home", data=True, id=id))
    elif "S" in id:
        return redirect(url_for("sph_view_home", data=True, id=id))
    elif "H" in id:
        return redirect(url_for("hyp_view_home", data=True, id=id))
    else:
        return render_template('errors/404.html'), 404

def is_valid(id):
    return True

# @app.route("/full_test")
def full_test():
    user_id = "full_test"
    session["geom_num"] = 0
    session["geom_order"] = GEOM_ORDER[0]
    session["current_geom"] = vis_order[ session["geom_order"][0] ]
    session["user_q_order"] = 0
    session["test_status"] = False 
    session["q_group"] = -1

    session.modified = True

    new_val = { 
        "id": str(user_id), 
        "completed_test": False,
        "geometry":  session["current_geom"],
        "time_start": datetime.isoformat(datetime.now()),
        "time_end": "",
        "question_order": list(range(len(Questions))), 
        "results": {"E": dict(), "S": dict(), "H": dict()}, 
        "interaction_list": {"E": dict(), "S": dict(), "H": dict()}, 
        "likert-feedback": {"E": dict(), "S": dict(), "H": dict()}, 
        "feedback": dict(), 
        "q_group": session["q_group"]
    }
    # riemannCollection.insert_one(new_val)   

    return redirect(url_for("homepage", data=True, id=user_id)) 

@app.route("/index/")
def user_index_form():
    id = request.args.get('id')

    if id == "E": 
        data = getEuclideanData()
    elif id == "H":
        data = getHyperbolicData()

    return render_template("visualization.html", 
        title=TITLE, 
        data={"geom": id} | data, 
        id=id, 
        q_id=None, 
        question=None,
        progress=[1, 3 * NUM_QUESTIONS]
    )    

    user_id = generate_id(id)
    session["user_id"] = user_id
    session["geom_num"] = 0
    session["geom_order"] = GEOM_ORDER[ int(user_id) % len(GEOM_ORDER) ]
    session["current_geom"] = vis_order[ session["geom_order"][0] ]
    session["user_q_order"] = int(user_id) % NUM_QUESTIONS
    session["test_status"] = False
    session["q_group"] = random.randint(0,2)

    print(f"Assigned q_group {session['q_group']}")

    session.modified = True

    new_val = { 
        "id": str(user_id), 
        "completed_test": False,
        "geometry":  [vis_order[ g ] for g in session["geom_order"]],
        "time_start": datetime.isoformat(datetime.now()),
        "time_end": "",
        "question_order": LATIN_SQUARE[session["user_q_order"]], 
        "results": {"E": dict(), "S": dict(), "H": dict()}, 
        "interaction_list": {"E": dict(), "S": dict(), "H": dict()}, 
        "likert-feedback": {"E": dict(), "S": dict(), "H": dict()}, 
        "feedback": dict(), 
        "q_group": session["q_group"]
    }
    # riemannCollection.insert_one(new_val)

    if is_valid(id):
        return redirect(url_for("homepage", data=True, id=user_id))
    else:
        return render_template('errors/404.html'), 404

@app.route('/start_test<id>')
def start_test(id):
    print(id)
    if id == "full_test": 
        start_index = 0
    else: 
        start_index = LATIN_SQUARE[ session["user_q_order"] ][0]
    session["cur_index"] = 0

    session.modified = True
    return redirect(url_for("view", id=id, q=start_index))


@app.route('/next_question', methods=['GET', 'POST'])
def next_question():

    if request.method == 'POST':
        # print(request.json)
        id = request.form.get('id')
        q_id = request.form.get('q')


        # q_id = Questions[int(request.form.get('q'))]["question_id"]
        user_answers = request.form.get('a')
        user_interactions = request.form.get('j')

        update_db_entry(
            id=id, 
            q_id=q_id,
            answers=user_answers, 
            interactions=user_interactions
        )

        session["cur_index"] += 1
        session.modified = True

        if session["cur_index"] >= NUM_QUESTIONS:
            session["geom_num"] += 1
            if session["geom_num"] >= 3:
                session["test_status"] = True
                session.modified = True
                return redirect(url_for("get_likert_feedback", id=id))
           
            session["cur_index"] = 0 
            session["current_geom"] = vis_order[ session["geom_order"][session["geom_num"]] ]
            session["user_q_order"] = random.randint(0, NUM_QUESTIONS-1)
            session.modified = True
            return redirect(url_for("get_likert_feedback", id=id))

        next_question = LATIN_SQUARE[ session["user_q_order"] ][session["cur_index"]]
        if id == "full_test": 
            next_question = int(session["cur_index"])
        return redirect(url_for("view", id=id, q=next_question))
    return redirect(url_for("index"), code=307)

@app.route('/next_question_between', methods=['GET', 'POST'])
def next_question_between():

    if request.method == 'POST':
        id = request.form.get('id')
        q_id = request.form.get('q')

        # q_id = Questions[int(request.form.get('q'))]["question_id"]
        user_answers = request.form.get('a')
        user_interactions = request.form.get('j')

        update_db_entry(
            id=id, 
            q_id=q_id,
            answers=user_answers, 
            interactions=user_interactions
        )

        session["cur_index"] += 1
        session.modified = True
        if session["cur_index"] >= NUM_QUESTIONS:
            session["test_status"] = True
            session.modified = True
            return redirect(url_for("get_likert_feedback", id=id))

        next_question = LATIN_SQUARE[ session["user_q_order"] ][session["cur_index"]]
        return redirect(url_for("view", id=id, q=next_question))
    return redirect(url_for("index"), code=307)


@app.route('/test-likert')
def test_likert():
    id = 'test123'
    return render_template("qual-feedback.html", title='Qualitative Feedback', id=id)

@app.route('/test-feedback')
def test_feedback():
    id = 'test123'
    return render_template("feedback.html", title='Feedback', id=id)

@app.route('/get-likert-feedback<id>')
def get_likert_feedback(id):
    return render_template("qual-feedback.html", title='Qualitative Feedback', id=id)

@app.route('/store-likert-feedback<id>', methods=['GET', 'POST'])
def store_likert_feedback(id):

    if request.method == 'POST':
        feedback = request.form.get("final_answers")
        if (id != "test123" and id != "full_test"):
            pass
            # doc = riemannCollection.find_one({'id': id})
            # doc["likert-feedback"][vis_order[ session["geom_order"][session["geom_num"]-1] ]] = feedback
            # riemannCollection.update_one(
            #     {'id': id},
            #     {"$set": doc}
            # )
        print(session["geom_num"])
        if session["geom_num"] >= 3:
            session["test_status"] = True
            session.modified = True
            return redirect(url_for("get_feedback", id=id))
        return redirect(url_for("homepage", data=True, id=session["user_id"]))
    return redirect(url_for("index"), code=307)

@app.route('/get-feedback<id>')
def get_feedback(id):
    return render_template("feedback.html", title='General Feedback', id=id)

@app.route('/store-feedback<id>', methods=['GET', 'POST'])
def store_feedback(id):

    if request.method == 'POST':
        feedback = request.form.get("responses")
        if (id != "test123" and id != "full_test"):
            pass
            # doc = riemannCollection.find_one({'id': id})
            # doc["feedback"] = feedback
            # riemannCollection.update_one(
            #     {'id': id},
            #     {"$set": doc}
            # )

        return redirect(url_for("end", id=id))
    return redirect(url_for("index"), code=307)

@app.route('/end')
def end():
    print(type(session))
    print(session.keys())
    id = request.args.get('id')
    if (len(session.keys()) != 0) and (session["test_status"]):
        # doc = riemannCollection.find_one({'id': id})
        # doc["completed_test"] = session["test_status"]
        # doc["time_end"] = datetime.isoformat(datetime.now())
        # riemannCollection.update_one(
        #     {'id': id}, 
        #     {"$set": doc}
        # )
        session.clear()
    return render_template("end.html", title='End of Study')

def update_db_entry(id, q_id, answers, interactions):
    import time 
    start = time.perf_counter()

    # doc = riemannCollection.find_one({"id": id})
    
    # doc["results"][ session["current_geom"] ][q_id] = answers
    # doc["interaction_list"][ session["current_geom"] ][q_id] = interactions

    # riemannCollection.update_one(
    #     {"id": id}, 
    #     {"$set": doc}
    # )

    end = time.perf_counter()

    print(f"Fetch and update took {end-start} seconds")

