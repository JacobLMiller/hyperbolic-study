import graph_tool.all as gt
import numpy as np

def load_graph_from_txt(fname):
    G = gt.load_graph_from_csv(fname,hashed=False)
    return G

def euclidean_random_graph(n=100, m=100):
    assert m <= (n * (n-1)) / 2

    G = gt.Graph(directed=False)
    G.add_vertex(10)

    _,hist = gt.label_components(G)
    print(hist)
    
    while len(hist) > 1:
        
        edist = lambda u,v: np.sum(np.square( u - v ))
        emb = np.random.uniform(0,1,(n,2))
        pairs = [((v,u), edist(emb[u],emb[v])) for u in range(n) for v in range(u)]
        edges = [p[0] for p in sorted(pairs, key = lambda pair: pair[1])[:m]]

        G = gt.Graph(directed=False)
        G.add_vertex(n)
        G.add_edge_list(edges)

        _,hist = gt.label_components(G)        
        print(hist)

    return G

sin, cos, asin, acos, sqrt = np.sin, np.cos, np.arcsin, np.arccos, np.sqrt

def spherical_random_graph(n=100,m=100):
    assert m <= (n * (n-1)) / 2

    G = gt.Graph(directed=False)
    G.add_vertex(10)

    _,hist = gt.label_components(G)
    
    while len(hist) > 1:    
        sdist = lambda x1,x2: acos( sin(x1[0])*sin(x2[0]) + cos(x1[0])*cos(x2[0])*cos(x1[1]-x2[1]) )
        emb = np.random.uniform([0,0], [np.pi, 2*np.pi], (n,2))
        pairs = [((v,u), sdist(emb[u], emb[v])) for u in range(n) for v in range(u)]
        edges = [p[0] for p in sorted(pairs, key=lambda pair: pair[1])[:m]]

        G = gt.Graph(directed=False)
        G.add_vertex(n)
        G.add_edge_list(edges)

        _,hist = gt.label_components(G)        

    return G

cosh, sinh, acosh = np.cosh, np.sinh, np.arccosh

def hyperbolic_random_graph(n=100,m=100):
    assert m <= (n * (n-1)) / 2

    G = gt.Graph(directed=False)
    G.add_vertex(10)

    _,hist = gt.label_components(G)
    c = 0
    while len(hist) > 1:    
        c += 1
        hdist = lambda x1,x2: acosh( cosh(x1[0]) * cosh(x2[0]) - sinh(x1[0]) * sinh(x2[0] * cos(x2[1] - x1[1])) )
        emb = np.random.uniform([0, 0], [1, 2*np.pi], (n,2))
        pairs = [((v,u), hdist(emb[u], emb[v])) for u in range(n) for v in range(u)]
        edges = [p[0] for p in sorted(pairs, key=lambda pair: pair[1])[:m]]

        G = gt.Graph(directed=False)
        G.add_vertex(n)
        G.add_edge_list(edges)

        _,hist = gt.label_components(G)        

    return G

n1 = 50
n2 = 100

E_group = [euclidean_random_graph(n1, 3*n1), euclidean_random_graph(n2, 4*n2), load_graph_from_txt("graphs/dwt_162.txt")]
S_group = [spherical_random_graph(n1, 3*n1), spherical_random_graph(n2, 4*n2), load_graph_from_txt("graphs/dodecahedron_3.txt")]
H_group = [hyperbolic_random_graph(n1, 3*n1), hyperbolic_random_graph(n2, 4*n2), gt.price_network(150,directed=False)]

names = {
    "e_group": ["e_50_150", "e_100_400", "square_lattice"],
    "s_group": ["s_50_150", "s_100_400", "dodecahedron_3"],
    "h_group": ["h_50_150", "h_100_400", "tree_150"]
}

from s_gd2 import layout_convergent
from modules.SGD_MDS_sphere import SMDS
from modules.SGD_hyperbolic import HMDS 
from modules.graph_functions import apsp
import json
import tqdm
from scipy import spatial


def sample_graph():

    G =  gt.collection.data["karate"]
    print(f"Finding graph sample")
    d = apsp(G)
    U = [u for u,v in G.iter_edges()]
    V = [v for u,v in G.iter_edges()]

    EX   = layout_convergent(U,V)
    SX,_ = SMDS(d,scale_heuristic=True).solve(1000,1e-7,schedule="convergent")
    HX   = HMDS(d).solve(1000)

    geom_mean = EX.mean(axis=0)
    distance, index = spatial.KDTree(EX).query(geom_mean)

    nodes = [{
        "id": v,
        "euclidean": {"x": EX[v,0], "y": EX[v,1]},
        "spherical": {"x": SX[v,1], "y": SX[v,0]}, #SMDS returns coordinates reversed from how webtool expects
        "hyperbolic": {"x": HX[v,0], "y": HX[v,1]}
    } for v in G.iter_vertices()]
    links = [{
        "source": u,
        "target": v
    } for u,v in G.iter_edges()]
    graph = {"multigraph": False, "undirected": True, "name": "karate", "central_node": int(index)}

    with open(f"../src/application/static/data/sample.json", "w") as fdata:
        json.dump({"graph": graph, "nodes": nodes, "links": links}, fdata, indent=4)    

def gen_graphs():
    for sname, group in zip(("e_group", "s_group", "h_group"),(E_group, S_group, H_group)):
        for i,G in enumerate(group):
            print(f"Finding graph {names[sname][i]}")
            d = apsp(G)
            U = [u for u,v in G.iter_edges()]
            V = [v for u,v in G.iter_edges()]

            EX   = layout_convergent(U,V)
            SX,_ = SMDS(d,scale_heuristic=True).solve(2000,1e-7,schedule="convergent")
            HX   = HMDS(d).solve(1000)

            geom_mean = EX.mean(axis=0)
            distance, index = spatial.KDTree(EX).query(geom_mean)            

            nodes = [{
                "id": v,
                "euclidean": {"x": EX[v,0], "y": EX[v,1]},
                "spherical": {"x": SX[v,1], "y": SX[v,0]}, #SMDS returns coordinates reversed from how webtool expects
                "hyperbolic": {"x": HX[v,0], "y": HX[v,1]}
            } for v in G.iter_vertices()]
            links = [{
                "source": u,
                "target": v
            } for u,v in G.iter_edges()]
            graph = {"multigraph": False, "undirected": True, "name": names[sname][i], "central_node": int(index)}

            with open(f"../src/application/static/data/{sname}_{i}.json", "w") as fdata:
                json.dump({"graph": graph, "nodes": nodes, "links": links}, fdata, indent=4)
        
def specific_graph(name):
    G = load_graph_from_txt("graphs/dwt_361.txt")    
    print(f"Number of nodes: {G.num_vertices()}")
    print(f"Number of edges: {G.num_edges()}")
    print(0 / 0)
    print(f"Finding graph {name}")
    d = apsp(G)
    U = [u for u,v in G.iter_edges()]
    V = [v for u,v in G.iter_edges()]

    EX   = layout_convergent(U,V)
    SX,_ = SMDS(d,scale_heuristic=True).solve(2000,1e-7,schedule="convergent")
    HX   = HMDS(d).solve(1000)

    geom_mean = EX.mean(axis=0)
    distance, index = spatial.KDTree(EX).query(geom_mean)            

    nodes = [{
        "id": v,
        "euclidean": {"x": EX[v,0], "y": EX[v,1]},
        "spherical": {"x": SX[v,1], "y": SX[v,0]}, #SMDS returns coordinates reversed from how webtool expects
        "hyperbolic": {"x": HX[v,0], "y": HX[v,1]}
    } for v in G.iter_vertices()]
    links = [{
        "source": u,
        "target": v
    } for u,v in G.iter_edges()]
    graph = {"multigraph": False, "undirected": True, "name": name, "central_node": int(index)}

    with open(f"../src/application/static/data/e_group_{2}.json", "w") as fdata:
        json.dump({"graph": graph, "nodes": nodes, "links": links}, fdata, indent=4)            

if __name__ == "__main__":
    specific_graph("lattice")