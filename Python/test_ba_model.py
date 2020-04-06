import networkx as nx
import pandas as pd
import network_tools as nt
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

df = pd.read_csv(nt.get_path() + '/data/Good_et_al/networks_naive/network_55250.txt', index_col=0, sep = '\t')
df_values = df.values
G = nx.from_numpy_matrix(df_values)
#print(G)
#nx.draw(G)
#plt.savefig(nt.get_path() + '/figs/ntwrk_good.png', format="PNG")
