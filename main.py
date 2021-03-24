import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import itemfreq

CMxyz = np.loadtxt("cpl_matric.txt")  # uvoz matrike
N = int(np.max(CMxyz[:, 0])) + 1  # razberemo število vozlov (+1 ker štejemo z 0)

print("Stevilo vozlov %d" % N)

CM = np.zeros([N, N], int)  # ustvarimo prazno 2D matriko
for c in CMxyz:
    CM[int(c[0])][int(c[1])] = int(c[2])  # pretvorba  xyz->2d matriko

G = nx.Graph()  # ustavrimo prazno mrežo G
for i in range(N):  #
    G.add_node(i)  # dodamo vozel i

for i in range(N):
    for j in range(N):
        if (CM[i, j] > 0):
            G.add_edge(i, j)  # ustvarimo povezavo med i in j vozlom

# lege=nx.circular_layout(G) #določi pozicije za izris, po krogu
# lege=nx.fruchterman_reingold_layout(G,k=1.0/np.sqrt(N))


lege = np.loadtxt("node_coordinates.txt")

nodesize = []
for i in range(N):
    nodesize.append(70 * np.sqrt(G.degree(i) + 1))

plt.figure(figsize= (15, 10))
plt.subplot(1, 2, 1)
nx.draw(G, lege, node_size=nodesize, node_color="blue", alpha=0.9, linewidths=0.2)
plt.title("delta = -10.0, clustering = %.3lf, avg_shorthest_path=%.3lf" %(nx.average_clustering(G), nx.average_shortest_path_length(G)))
plt.axis("off")

plt.subplot(1, 2, 2)
node_degree=[val for (node, val) in G.degree()]
freq = itemfreq(node_degree)
a = freq[:, 0]
b = freq[:, 1]
plt.scatter(np.log10(a), np.log10(b))
plt.xlabel("log(k)")
plt.ylabel("log(p(k))")
plt.savefig("SFN.png", dpi=100, bbox_inches="tight")
plt.show()
plt.close()