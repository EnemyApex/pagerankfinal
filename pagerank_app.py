"""
Implémentation et expérimentation de l'algorithme PageRank
avec interface graphique en Python (Tkinter + NetworkX + Matplotlib).

Auteur : Projet d'analyse de données - Antananarivo
Description :
    - Implémentation "from scratch" de PageRank (méthode des puissances / power iteration).
    - Interface graphique permettant de :
        * construire un graphe (manuellement ou aléatoirement),
        * régler le facteur d'amortissement (damping factor) et la tolérance,
        * lancer le calcul et visualiser le graphe avec les scores PageRank,
        * afficher la courbe de convergence,
        * exporter les résultats (CSV).

Dépendances : numpy, networkx, matplotlib, tkinter (inclus dans la distribution standard de Python)
Installation : pip install numpy networkx matplotlib
Exécution   : python pagerank_app.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import csv
import time
import streamlit

# ----------------------------------------------------------------------------
# 1. IMPLEMENTATION DE L'ALGORITHME PAGERANK (méthode des puissances)
# ----------------------------------------------------------------------------
def pagerank_power_iteration(adj: np.ndarray, d: float = 0.85,
                              tol: float = 1e-8, max_iter: int = 200):
    """
    Calcule les scores PageRank d'un graphe dirigé par la méthode des puissances.

    Paramètres
    ----------
    adj : np.ndarray (n, n)
        Matrice d'adjacence du graphe (adj[i, j] = 1 s'il existe un lien i -> j).
    d : float
        Facteur d'amortissement (damping factor), généralement 0.85.
    tol : float
        Seuil de convergence (norme L1 de la différence entre deux itérations).
    max_iter : int
        Nombre maximal d'itérations autorisées.

    Retour
    ------
    PR : np.ndarray (n,)
        Vecteur des scores PageRank (somme = 1).
    n_iter : int
        Nombre d'itérations effectuées jusqu'à convergence.
    history : list[float]
        Historique de l'erreur (norme L1) à chaque itération, utile pour
        tracer la courbe de convergence.
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([]), 0, []

    out_degree = adj.sum(axis=1)
    dangling = (out_degree == 0)          # pages sans lien sortant

    # Matrice de transition stochastique M (M[i, j] = probabilité d'aller de i vers j)
    M = np.zeros((n, n))
    for i in range(n):
        if out_degree[i] > 0:
            M[i] = adj[i] / out_degree[i]

    PR = np.ones(n) / n                    # distribution initiale uniforme
    history = []
    n_iter = 0

    for it in range(1, max_iter + 1):
        dangling_mass = PR[dangling].sum()
        # Redistribution de la masse des noeuds "dangling" + saut aléatoire (1-d)/n
        new_PR = (1 - d) / n + d * (M.T @ PR + dangling_mass / n)
        diff = np.abs(new_PR - PR).sum()
        history.append(diff)
        PR = new_PR
        n_iter = it
        if diff < tol:
            break

    return PR, n_iter, history


# ----------------------------------------------------------------------------
# 2. INTERFACE GRAPHIQUE (Tkinter)
# ----------------------------------------------------------------------------
class PageRankApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Expérimentation de l'algorithme PageRank")
        self.root.geometry("1150x700")

        self.G = nx.DiGraph()
        self.pr_scores = None
        self.history = []

        self._build_layout()

    # ---------------------------- UI LAYOUT --------------------------------
    def _build_layout(self):
        # -- Panneau de contrôle (gauche) --
        control = ttk.Frame(self.root, padding=10)
        control.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control, text="Paramètres du graphe", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        frm_random = ttk.LabelFrame(control, text="Graphe aléatoire", padding=8)
        frm_random.pack(fill=tk.X, pady=6)

        ttk.Label(frm_random, text="Nombre de noeuds :").grid(row=0, column=0, sticky="w")
        self.nb_nodes = tk.IntVar(value=10)
        ttk.Entry(frm_random, textvariable=self.nb_nodes, width=8).grid(row=0, column=1)

        ttk.Label(frm_random, text="Probabilité de lien (p) :").grid(row=1, column=0, sticky="w")
        self.prob_edge = tk.DoubleVar(value=0.2)
        ttk.Entry(frm_random, textvariable=self.prob_edge, width=8).grid(row=1, column=1)

        ttk.Button(frm_random, text="Générer graphe aléatoire",
                   command=self.generate_random_graph).grid(row=2, column=0, columnspan=2, pady=5)

        frm_manual = ttk.LabelFrame(control, text="Graphe manuel", padding=8)
        frm_manual.pack(fill=tk.X, pady=6)

        ttk.Label(frm_manual, text="Lien (ex: A B) :").grid(row=0, column=0, sticky="w")
        self.edge_entry = tk.StringVar()
        ttk.Entry(frm_manual, textvariable=self.edge_entry, width=15).grid(row=0, column=1)
        ttk.Button(frm_manual, text="Ajouter lien", command=self.add_edge).grid(row=1, column=0, columnspan=2, pady=3)
        ttk.Button(frm_manual, text="Réinitialiser graphe", command=self.reset_graph).grid(row=2, column=0, columnspan=2)

        frm_params = ttk.LabelFrame(control, text="Paramètres PageRank", padding=8)
        frm_params.pack(fill=tk.X, pady=6)

        ttk.Label(frm_params, text="Facteur d'amortissement (d) :").grid(row=0, column=0, sticky="w")
        self.damping = tk.DoubleVar(value=0.85)
        ttk.Entry(frm_params, textvariable=self.damping, width=8).grid(row=0, column=1)

        ttk.Label(frm_params, text="Tolérance :").grid(row=1, column=0, sticky="w")
        self.tol = tk.DoubleVar(value=1e-8)
        ttk.Entry(frm_params, textvariable=self.tol, width=8).grid(row=1, column=1)

        ttk.Label(frm_params, text="Max itérations :").grid(row=2, column=0, sticky="w")
        self.max_iter = tk.IntVar(value=200)
        ttk.Entry(frm_params, textvariable=self.max_iter, width=8).grid(row=2, column=1)

        ttk.Button(control, text="Calculer PageRank", command=self.run_pagerank).pack(fill=tk.X, pady=10)
        ttk.Button(control, text="Exporter résultats (CSV)", command=self.export_csv).pack(fill=tk.X)

        self.status = tk.StringVar(value="Aucun graphe chargé.")
        ttk.Label(control, textvariable=self.status, wraplength=250, foreground="gray").pack(pady=10, anchor="w")

        # -- Zone de visualisation (droite, avec onglets) --
        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.tab_graph = ttk.Frame(notebook)
        self.tab_conv = ttk.Frame(notebook)
        self.tab_table = ttk.Frame(notebook)
        notebook.add(self.tab_graph, text="Graphe & scores")
        notebook.add(self.tab_conv, text="Courbe de convergence")
        notebook.add(self.tab_table, text="Tableau des résultats")

        self.fig_graph = Figure(figsize=(6, 5))
        self.ax_graph = self.fig_graph.add_subplot(111)
        self.canvas_graph = FigureCanvasTkAgg(self.fig_graph, master=self.tab_graph)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_conv = Figure(figsize=(6, 5))
        self.ax_conv = self.fig_conv.add_subplot(111)
        self.canvas_conv = FigureCanvasTkAgg(self.fig_conv, master=self.tab_conv)
        self.canvas_conv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        columns = ("Noeud", "Score PageRank", "Rang")
        self.tree = ttk.Treeview(self.tab_table, columns=columns, show="headings")
        for c in columns:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)

    # --------------------------- ACTIONS GUI --------------------------------
    def generate_random_graph(self):
        n = self.nb_nodes.get()
        p = self.prob_edge.get()
        self.G = nx.gnp_random_graph(n, p, directed=True, seed=None)
        self.status.set(f"Graphe aléatoire généré : {n} noeuds, {self.G.number_of_edges()} liens.")
        self._draw_graph()

    def add_edge(self):
        text = self.edge_entry.get().strip()
        if not text:
            return
        parts = text.split()
        if len(parts) != 2:
            messagebox.showwarning("Format invalide", "Utilisez le format : NoeudSource NoeudCible (ex: A B)")
            return
        self.G.add_edge(parts[0], parts[1])
        self.edge_entry.set("")
        self.status.set(f"Lien ajouté : {parts[0]} -> {parts[1]}. Total liens : {self.G.number_of_edges()}")
        self._draw_graph()

    def reset_graph(self):
        self.G = nx.DiGraph()
        self.pr_scores = None
        self.ax_graph.clear()
        self.canvas_graph.draw()
        self.ax_conv.clear()
        self.canvas_conv.draw()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.status.set("Graphe réinitialisé.")

    def run_pagerank(self):
        if self.G.number_of_nodes() == 0:
            messagebox.showwarning("Graphe vide", "Créez ou générez un graphe avant de lancer le calcul.")
            return

        nodes = list(self.G.nodes())
        adj = nx.to_numpy_array(self.G, nodelist=nodes)

        t0 = time.time()
        pr, n_iter, history = pagerank_power_iteration(
            adj, d=self.damping.get(), tol=self.tol.get(), max_iter=self.max_iter.get()
        )
        elapsed = time.time() - t0

        self.pr_scores = dict(zip(nodes, pr))
        self.history = history

        self.status.set(
            f"Calcul terminé en {n_iter} itérations ({elapsed*1000:.2f} ms).\n"
            f"Somme des scores : {pr.sum():.6f}"
        )

        self._draw_graph()
        self._draw_convergence()
        self._fill_table(nodes, pr)

    def _draw_graph(self):
        self.ax_graph.clear()
        if self.G.number_of_nodes() > 0:
            pos = nx.spring_layout(self.G, seed=42)
            if self.pr_scores:
                sizes = [3000 * self.pr_scores[n] + 200 for n in self.G.nodes()]
                colors = [self.pr_scores[n] for n in self.G.nodes()]
            else:
                sizes = 400
                colors = "#4C72B0"
            nx.draw_networkx_nodes(self.G, pos, ax=self.ax_graph, node_size=sizes,
                                    node_color=colors, cmap="viridis", alpha=0.9)
            nx.draw_networkx_edges(self.G, pos, ax=self.ax_graph, arrowstyle="-|>", arrowsize=12)
            nx.draw_networkx_labels(self.G, pos, ax=self.ax_graph, font_size=8)
            self.ax_graph.set_title("Graphe (taille du noeud = score PageRank)")
        self.ax_graph.axis("off")
        self.canvas_graph.draw()

    def _draw_convergence(self):
        self.ax_conv.clear()
        if self.history:
            self.ax_conv.plot(range(1, len(self.history) + 1), self.history, marker="o", markersize=3)
            self.ax_conv.set_yscale("log")
            self.ax_conv.set_xlabel("Itération")
            self.ax_conv.set_ylabel("Erreur L1 (log)")
            self.ax_conv.set_title("Convergence de l'algorithme")
        self.canvas_conv.draw()

    def _fill_table(self, nodes, pr):
        for row in self.tree.get_children():
            self.tree.delete(row)
        order = np.argsort(-pr)
        for rank, idx in enumerate(order, start=1):
            self.tree.insert("", "end", values=(nodes[idx], f"{pr[idx]:.6f}", rank))

    def export_csv(self):
        if self.pr_scores is None:
            messagebox.showwarning("Rien à exporter", "Lancez d'abord le calcul PageRank.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Noeud", "Score PageRank"])
            for node, score in sorted(self.pr_scores.items(), key=lambda x: -x[1]):
                writer.writerow([node, score])
        messagebox.showinfo("Export réussi", f"Résultats exportés vers :\n{path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PageRankApp(root)
    root.mainloop()
