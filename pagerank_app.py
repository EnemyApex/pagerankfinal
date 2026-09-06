"""
============================================================================
 APPLICATION WEB — Expérimentation de l'algorithme PageRank
============================================================================
Auteur      : Projet d'analyse de données — Antananarivo
Framework   : Streamlit (interface web) + NetworkX (graphes) + Matplotlib
              (visualisation) + NumPy (calcul numérique)

Description :
    Cette application permet de :
        1. Construire un graphe orienté (génération aléatoire, saisie
           manuelle de liens, ou jeux d'exemples pédagogiques).
        2. Régler les hyperparamètres de PageRank : facteur d'amortissement
           (damping factor), tolérance de convergence, nombre max
           d'itérations.
        3. Calculer les scores PageRank avec une implémentation "from
           scratch" par la méthode des puissances (power iteration).
        4. Visualiser le graphe (taille/couleur des noeuds = score),
           la courbe de convergence, et un tableau de classement.
        5. Exporter les résultats au format CSV.

Exécution locale :
    pip install -r requirements.txt
    streamlit run pagerank_app.py

Déploiement :
    Ce fichier est prêt à être déployé sur Streamlit Community Cloud
    (streamlit.io/cloud) : il suffit de pousser ce dépôt sur GitHub et
    de pointer Streamlit Cloud vers pagerank_app.py.
============================================================================
"""

import io
import time

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================================
# 1. CONFIGURATION GÉNÉRALE DE LA PAGE
# ============================================================================
# set_page_config DOIT être le premier appel Streamlit du script.
st.set_page_config(
    page_title="PageRank Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Palette et petite feuille de style "maison" pour un rendu plus professionnel
# (Streamlit permet d'injecter du CSS via st.markdown avec unsafe_allow_html)
# ----------------------------------------------------------------------------
PRIMARY_COLOR = "#4C72B0"     # bleu utilisé aussi dans les graphes matplotlib
ACCENT_COLOR = "#DD8452"      # orange d'accent
BG_CARD = "#F7F9FC"

CUSTOM_CSS = f"""
<style>
    /* Titre principal avec dégradé discret */
    .main-title {{
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, {PRIMARY_COLOR}, {ACCENT_COLOR});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }}
    .subtitle {{
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }}
    /* Cartes de métriques */
    div[data-testid="stMetric"] {{
        background-color: {BG_CARD};
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }}
    /* Boutons principaux */
    div.stButton > button {{
        border-radius: 8px;
        font-weight: 600;
    }}
    /* Séparateurs plus discrets */
    hr {{
        margin: 0.6rem 0;
    }}
    footer {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# 2. IMPLÉMENTATION DE L'ALGORITHME PAGERANK (méthode des puissances)
# ============================================================================
def pagerank_power_iteration(adj: np.ndarray, d: float = 0.85,
                              tol: float = 1e-8, max_iter: int = 200):
    """
    Calcule les scores PageRank d'un graphe dirigé par la méthode des
    puissances (power iteration), implémentée "from scratch" (sans
    utiliser nx.pagerank).

    Paramètres
    ----------
    adj : np.ndarray (n, n)
        Matrice d'adjacence du graphe (adj[i, j] = 1 s'il existe un
        lien i -> j).
    d : float
        Facteur d'amortissement (damping factor), généralement 0.85.
        Représente la probabilité qu'un "surfeur aléatoire" suive un
        lien plutôt que de sauter vers une page au hasard.
    tol : float
        Seuil de convergence (norme L1 de la différence entre deux
        itérations successives).
    max_iter : int
        Nombre maximal d'itérations autorisées (sécurité anti-boucle
        infinie si la convergence est lente).

    Retour
    ------
    PR : np.ndarray (n,)
        Vecteur des scores PageRank (la somme des scores vaut 1).
    n_iter : int
        Nombre d'itérations effectuées jusqu'à convergence.
    history : list[float]
        Historique de l'erreur (norme L1) à chaque itération, utile
        pour tracer la courbe de convergence.
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([]), 0, []

    # Degré sortant de chaque noeud (nombre de liens partant du noeud i)
    out_degree = adj.sum(axis=1)
    dangling = (out_degree == 0)  # noeuds "puits" (sans lien sortant)

    # Construction de la matrice de transition stochastique M
    # M[i, j] = probabilité de passer du noeud i au noeud j en suivant un lien
    M = np.zeros((n, n))
    for i in range(n):
        if out_degree[i] > 0:
            M[i] = adj[i] / out_degree[i]

    PR = np.ones(n) / n  # distribution initiale uniforme (1/n pour chaque noeud)
    history = []
    n_iter = 0

    for it in range(1, max_iter + 1):
        # Masse de probabilité "perdue" par les noeuds sans lien sortant :
        # elle est redistribuée uniformément sur tous les noeuds.
        dangling_mass = PR[dangling].sum()

        # Équation de PageRank :
        #   PR = (1-d)/n  +  d * ( M^T . PR  +  masse des noeuds puits / n )
        new_PR = (1 - d) / n + d * (M.T @ PR + dangling_mass / n)

        diff = np.abs(new_PR - PR).sum()  # norme L1 de l'écart
        history.append(diff)
        PR = new_PR
        n_iter = it

        if diff < tol:  # convergence atteinte -> on arrête plus tôt
            break

    return PR, n_iter, history


# ============================================================================
# 3. JEUX DE GRAPHES D'EXEMPLE (pour démarrer rapidement une démonstration)
# ============================================================================
def get_example_graph(name: str) -> nx.DiGraph:
    """Retourne un graphe dirigé prédéfini, utile pour une démo rapide."""
    G = nx.DiGraph()

    if name == "Web simplifié (5 pages)":
        edges = [
            ("A", "B"), ("A", "C"), ("B", "C"),
            ("C", "A"), ("D", "C"), ("D", "B"),
            ("E", "D"), ("B", "E"),
        ]
        G.add_edges_from(edges)

    elif name == "Étoile (hub central)":
        hub = "Hub"
        for i in range(1, 7):
            G.add_edge(f"N{i}", hub)
            G.add_edge(hub, f"N{i}")

    elif name == "Chaîne + boucle":
        chain = [f"P{i}" for i in range(1, 7)]
        for a, b in zip(chain[:-1], chain[1:]):
            G.add_edge(a, b)
        G.add_edge(chain[-1], chain[0])  # boucle refermant la chaîne

    return G


# ============================================================================
# 4. ÉTAT DE SESSION (persistance des données entre les interactions)
# ============================================================================
# Streamlit ré-exécute tout le script à chaque interaction : on utilise
# st.session_state pour conserver le graphe et les résultats entre deux
# clics de l'utilisateur.
if "G" not in st.session_state:
    st.session_state.G = nx.DiGraph()
if "pr_scores" not in st.session_state:
    st.session_state.pr_scores = None
if "history" not in st.session_state:
    st.session_state.history = []
if "n_iter" not in st.session_state:
    st.session_state.n_iter = 0
if "elapsed_ms" not in st.session_state:
    st.session_state.elapsed_ms = 0.0
if "layout_seed" not in st.session_state:
    st.session_state.layout_seed = 42


# ============================================================================
# 5. EN-TÊTE DE L'APPLICATION
# ============================================================================
st.markdown('<div class="main-title">🕸️ PageRank Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Implémentation "from scratch" de l\'algorithme '
    'PageRank (méthode des puissances) — construisez un graphe, réglez les '
    'paramètres, et observez la convergence en temps réel.</div>',
    unsafe_allow_html=True,
)


# ============================================================================
# 6. BARRE LATÉRALE — CONSTRUCTION DU GRAPHE ET PARAMÈTRES
# ============================================================================
with st.sidebar:
    st.header("⚙️ Construction du graphe")

    mode = st.radio(
        "Mode de construction",
        ["Exemple prédéfini", "Graphe aléatoire", "Saisie manuelle"],
        help="Choisissez comment construire le graphe orienté à analyser.",
    )

    # ---- Mode 1 : exemples pédagogiques -----------------------------------
    if mode == "Exemple prédéfini":
        example_name = st.selectbox(
            "Choisir un exemple",
            ["Web simplifié (5 pages)", "Étoile (hub central)", "Chaîne + boucle"],
        )
        if st.button("📥 Charger l'exemple", use_container_width=True):
            st.session_state.G = get_example_graph(example_name)
            st.session_state.pr_scores = None
            st.success(f"Graphe « {example_name} » chargé.")

    # ---- Mode 2 : génération aléatoire -------------------------------------
    elif mode == "Graphe aléatoire":
        nb_nodes = st.slider("Nombre de noeuds", 2, 40, 10)
        prob_edge = st.slider("Probabilité de lien (p)", 0.01, 1.0, 0.2)
        seed_input = st.number_input("Graine aléatoire (seed)", value=0, step=1)
        if st.button("🎲 Générer le graphe", use_container_width=True):
            st.session_state.G = nx.gnp_random_graph(
                nb_nodes, prob_edge, directed=True,
                seed=int(seed_input) if seed_input != 0 else None,
            )
            st.session_state.pr_scores = None
            st.success(
                f"Graphe généré : {nb_nodes} noeuds, "
                f"{st.session_state.G.number_of_edges()} liens."
            )

    # ---- Mode 3 : saisie manuelle ------------------------------------------
    else:
        st.caption("Ajoutez des liens un par un (format : Source Cible)")
        col_a, col_b = st.columns(2)
        with col_a:
            src = st.text_input("Source", value="", key="src_node")
        with col_b:
            dst = st.text_input("Cible", value="", key="dst_node")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Ajouter le lien", use_container_width=True):
                if src.strip() and dst.strip():
                    st.session_state.G.add_edge(src.strip(), dst.strip())
                    st.session_state.pr_scores = None
                    st.success(f"Lien ajouté : {src.strip()} → {dst.strip()}")
                else:
                    st.warning("Veuillez renseigner un noeud source et un noeud cible.")
        with c2:
            if st.button("🗑️ Réinitialiser", use_container_width=True):
                st.session_state.G = nx.DiGraph()
                st.session_state.pr_scores = None
                st.session_state.history = []
                st.info("Graphe réinitialisé.")

    st.divider()
    st.header("🎛️ Paramètres PageRank")

    damping = st.slider(
        "Facteur d'amortissement (d)", 0.05, 0.99, 0.85, step=0.01,
        help="Probabilité que le surfeur aléatoire suive un lien plutôt "
             "que de sauter vers une page quelconque. Valeur usuelle : 0.85.",
    )
    tol_exp = st.slider(
        "Tolérance (10^-x)", 2, 12, 8,
        help="Seuil de convergence : l'algorithme s'arrête quand la variation "
             "des scores entre deux itérations passe sous ce seuil.",
    )
    tol = 10 ** (-tol_exp)
    max_iter = st.slider("Nombre maximal d'itérations", 10, 500, 200)

    st.divider()
    run = st.button("▶️ Calculer PageRank", type="primary", use_container_width=True)


# ============================================================================
# 7. CALCUL DU PAGERANK (déclenché par le bouton de la sidebar)
# ============================================================================
G = st.session_state.G

if run:
    if G.number_of_nodes() == 0:
        st.warning("⚠️ Le graphe est vide. Créez ou générez un graphe avant de lancer le calcul.")
    else:
        nodes = list(G.nodes())
        adj = nx.to_numpy_array(G, nodelist=nodes)

        t0 = time.time()
        pr, n_iter, history = pagerank_power_iteration(
            adj, d=damping, tol=tol, max_iter=max_iter
        )
        elapsed_ms = (time.time() - t0) * 1000

        st.session_state.pr_scores = dict(zip(nodes, pr))
        st.session_state.history = history
        st.session_state.n_iter = n_iter
        st.session_state.elapsed_ms = elapsed_ms


# ============================================================================
# 8. TABLEAU DE BORD — MÉTRIQUES RAPIDES
# ============================================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Noeuds", G.number_of_nodes())
m2.metric("Liens", G.number_of_edges())
m3.metric("Itérations", st.session_state.n_iter if st.session_state.pr_scores else "—")
m4.metric(
    "Temps de calcul",
    f"{st.session_state.elapsed_ms:.2f} ms" if st.session_state.pr_scores else "—",
)

st.divider()


# ============================================================================
# 9. ONGLETS DE VISUALISATION
# ============================================================================
tab_graph, tab_conv, tab_table, tab_about = st.tabs(
    ["🕸️ Graphe & scores", "📉 Convergence", "📊 Classement", "ℹ️ À propos"]
)

# ---------------------------- Onglet 1 : Graphe ----------------------------
with tab_graph:
    if G.number_of_nodes() == 0:
        st.info("Construisez un graphe depuis le panneau latéral pour commencer.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=st.session_state.layout_seed)

        if st.session_state.pr_scores:
            pr_scores = st.session_state.pr_scores
            sizes = [4000 * pr_scores[n] + 250 for n in G.nodes()]
            colors = [pr_scores[n] for n in G.nodes()]
            nodes_drawing = nx.draw_networkx_nodes(
                G, pos, ax=ax, node_size=sizes, node_color=colors,
                cmap="viridis", alpha=0.92, edgecolors="white", linewidths=1.2,
            )
            fig.colorbar(nodes_drawing, ax=ax, label="Score PageRank", shrink=0.8)
            ax.set_title("Graphe — taille et couleur du noeud ∝ score PageRank",
                         fontsize=12, fontweight="bold")
        else:
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_size=500, node_color=PRIMARY_COLOR,
                alpha=0.9, edgecolors="white", linewidths=1.2,
            )
            ax.set_title("Graphe (lancez le calcul pour voir les scores)",
                         fontsize=12, fontweight="bold")

        nx.draw_networkx_edges(
            G, pos, ax=ax, arrowstyle="-|>", arrowsize=14,
            edge_color="#9CA3AF", connectionstyle="arc3,rad=0.05",
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)

        if st.session_state.pr_scores:
            top_node = max(st.session_state.pr_scores, key=st.session_state.pr_scores.get)
            st.caption(
                f"💡 Le noeud le plus influent est **{top_node}** avec un score de "
                f"**{st.session_state.pr_scores[top_node]:.4f}**."
            )

# ------------------------- Onglet 2 : Convergence --------------------------
with tab_conv:
    if not st.session_state.history:
        st.info("Lancez le calcul PageRank pour afficher la courbe de convergence.")
    else:
        history = st.session_state.history
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(
            range(1, len(history) + 1), history,
            marker="o", markersize=4, color=ACCENT_COLOR, linewidth=2,
        )
        ax2.set_yscale("log")
        ax2.set_xlabel("Itération")
        ax2.set_ylabel("Erreur L1 (échelle log)")
        ax2.set_title("Convergence de l'algorithme des puissances",
                      fontsize=12, fontweight="bold")
        ax2.grid(True, which="both", linestyle="--", alpha=0.4)
        st.pyplot(fig2, use_container_width=True)
        st.caption(
            f"Convergence atteinte en **{st.session_state.n_iter}** itérations "
            f"(tolérance = {tol:.0e})."
        )

# --------------------------- Onglet 3 : Tableau -----------------------------
with tab_table:
    if not st.session_state.pr_scores:
        st.info("Lancez le calcul PageRank pour afficher le classement des noeuds.")
    else:
        df = pd.DataFrame(
            sorted(st.session_state.pr_scores.items(), key=lambda x: -x[1]),
            columns=["Noeud", "Score PageRank"],
        )
        df.insert(0, "Rang", range(1, len(df) + 1))
        df["Score PageRank"] = df["Score PageRank"].round(6)

        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Score PageRank": st.column_config.ProgressColumn(
                    "Score PageRank", format="%.6f",
                    min_value=0, max_value=float(df["Score PageRank"].max()),
                )
            },
        )

        # Export CSV (bouton natif Streamlit, pas de boîte de dialogue système
        # nécessaire contrairement à la version Tkinter)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Exporter les résultats (CSV)",
            data=csv_buffer.getvalue(),
            file_name="pagerank_resultats.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ---------------------------- Onglet 4 : À propos ---------------------------
with tab_about:
    st.markdown(
        """
        ### À propos de cette application

        Cette application implémente l'algorithme **PageRank** — l'algorithme
        historiquement utilisé par Google pour classer les pages web — à
        l'aide de la **méthode des puissances (power iteration)**, codée
        "from scratch" en NumPy (sans appeler `networkx.pagerank`).

        **Principe :** chaque noeud reçoit une importance qui dépend du
        nombre et de l'importance des noeuds qui pointent vers lui. On
        modélise un "surfeur aléatoire" qui, à chaque étape, suit un lien
        sortant avec une probabilité `d` (facteur d'amortissement), ou saute
        vers une page quelconque avec une probabilité `1-d`.

        **Formule itérative :**

        `PR = (1-d)/n + d · (Mᵀ · PR + masse_dangling / n)`

        où `M` est la matrice de transition stochastique du graphe, et
        `masse_dangling` est la masse de probabilité des noeuds sans lien
        sortant, redistribuée uniformément pour que la somme des scores
        reste égale à 1.

        ---
        **Stack technique :** Python · Streamlit · NetworkX · NumPy · Matplotlib · Pandas
        """
    )
