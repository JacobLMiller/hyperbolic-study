from flask import jsonify, render_template, request, redirect, url_for, session
from urllib.parse import unquote as urllib_unquote
from application import app
from werkzeug.utils import secure_filename
import json
import re
from pymongo import MongoClient
# from mongopass import mongopass

client = MongoClient(app.config.get('MONGOPASS'))

db = client.riemannStudy
riemannCollection = db.riemannCollection

graph_ids = dict(zip(range(9), [f"{gtype}_group_{num}.json" for gtype in ["s","h","e"] for num in range(3)]))

def generate_id(id):
    id += str(riemannCollection.count_documents({}))
    return id

def get_graph(id):
    print(id)
    id_int = int(re.findall(r"\d+", id)[0])
    with open(f"src/application/data/{graph_ids[id_int]}", 'r') as fdata:
        gdata = json.load(fdata)

    return gdata

def get_question(id):
    import os

    with open(f"src/application/data/sample_questions.json", 'r') as fdata:
        qdata = json.load(fdata)
    graph_id = ""
    question = {}
    for q in qdata["questions"]:
        if q["q_id"] == id:
            graph_id = q["graph"]["graph_id"]
            question = q
            break
    return get_graph(graph_id), question

@app.template_filter('unquote')
def unquote(url):
    safe = app.jinja_env.filters['safe']
    return safe(urllib_unquote(url))


@app.route('/')
@app.route('/index')
def index():
    return render_template('choose.html', title='non-Euclidean Graph Survey Homepage (pilot!)', data=None)

@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/euclidean/homepage<data>') 
def euc_view_home(data):
    id = request.args.get('id')
    return render_template("euc-vis-home.html", title='Euclidean Homepage', data=data, id=id)

@app.route('/euclidean/test<id>_<q>') 
def euc_view(id, q):
    if "E" in id:
        gdata, question = get_question(q)
        return render_template("visualization.html", title='Euclidean', data=gdata, id=id, q_id=q, question=question)
    return redirect(url_for("index"))

@app.route('/<id>')
def user_index(id):
    if   "E" in id:
        return redirect(url_for("euc_view_home", data=True, id=id))
    elif "S" in id:
        return redirect(url_for("sph_view_home", data=True, id=id))
    elif "H" in id:
        return redirect(url_for("hyp_view_home", data=True, id=id))
    else:
        return render_template('errors/404.html'), 404

@app.route('/index/')
def user_index_form():
    id = request.args.get('id')
    user_id = generate_id(id)
    session["user_id"] = user_id
    session["user_answers"] = dict()
    session["user_interactions"] = dict()
    session["test_status"] = False
    session["likert_feedback"] = dict()
    session["comments"] = dict()
    
    new_val = { "id": user_id, "completed_test": False }
    riemannCollection.insert_one(new_val)

    if "E" in id:
        return redirect(url_for("euc_view_home", data=True, id=user_id))
    elif "S" in id:
        return redirect(url_for("sph_view_home", data=True, id=user_id))
    elif "H" in id:
        return redirect(url_for("hyp_view_home", data=True, id=user_id))
    else:
        return render_template('errors/404.html'), 404


@app.route('/euclidean/test/next<id>_<q>_<a>_<j>')
def next_question(id, q, a, j):
    
    session["user_answers"][q] = a
    session["user_interactions"][q] = j
    session.modified = True
    print(session)
    # TODO: Fix Euclidean interaction. Current interaction list spams zoom
    next_q_index = question_queue.index(q) + 1
    if next_q_index >= len(question_queue):
        session["test_status"] = True
        session.modified = True
        return redirect(url_for("get_likert_feedback", id=id))
    else:
        next_q = question_queue[next_q_index]
        return redirect(url_for("euc_view", id=id, q=next_q))
    
@app.route('/spherical/test/next<id>_<q>_<a>_<j>')
def sph_next_question(id, q, a, j):
    session["user_answers"][q] = a
    user_int = json.loads(j)
    if q in session["user_interactions"].keys():
        session["user_interactions"][q].append(user_int)
    else:
        session["user_interactions"][q] = user_int
    
    session.modified = True
    print(session)
    # TODO: Fix Spherical interaction. Current interaction list has extra zooms
    next_q_index = question_queue.index(q) + 1
    if next_q_index >= len(question_queue):
        session["test_status"] = True
        session.modified = True
        return redirect(url_for("get_likert_feedback", id=id))
    else:
        next_q = question_queue[next_q_index]
        return redirect(url_for("sph_view", id=id, q=next_q))
    
@app.route('/hyperbolic/test/next<id>_<q>_<a>_<j>')
def hyp_next_question(id, q, a, j):
    session["user_answers"][q] = a
    session["user_interactions"][q] = j
    session.modified = True
    # TODO: Fix same node being recorded for hover.
    next_q_index = question_queue.index(q) + 1
    if next_q_index >= len(question_queue):
        session["test_status"] = True
        session.modified = True
        return redirect(url_for("get_likert_feedback", id=id))
    else:
        next_q = question_queue[next_q_index]
        return redirect(url_for("hyp_view", id=id, q=next_q))

@app.route('/test-likert')
def test_likert():
    id = 'test123'
    return render_template("qual-feedback.html", title='Qualitative Feedback', id=id)

@app.route('/get-likert-feedback<id>')
def get_likert_feedback(id):
    return render_template("qual-feedback.html", title='Qualitative Feedback', id=id)

@app.route('/store-likert-feedback<id>')
def store_likert_feedback(id):
    
    for i in range(1, 13):
        session["likert_feedback"][f'lq{i}'] = request.args.get(f'lq{i}')
    
    session.modified = True
    return redirect(url_for("get_feedback", id=id))

@app.route('/get-feedback<id>')
def get_feedback(id):
    return render_template("feedback.html", title='Feedback', id=id)

@app.route('/store-feedback<id>')
def store_feedback(id):

    for i in range(1, 4):
        session["comments"][f"fq{i}"] = request.args.get(f'fq{i}')

    session.modified = True
    return redirect(url_for("end", id=id))
    

@app.route('/end')
def end():
    id = request.args.get('id')
    if (session["test_status"]):
        query = { "id" : id}
        update = {"$set": { "completed_test": session["test_status"], "results": session["user_answers"], "interaction_list": session["user_interactions"], "feedback": session["comments"], "likert-feedback": session["likert_feedback"] } }
        riemannCollection.update_one(query, update)
    return render_template("end.html", title='End of Study')