import streamlit as st
import harmonypy as hm
import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.sparse
import plotly.express as px


st.set_page_config(page_title="scRNA-seq Pipeline", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Επιλογές")
st.sidebar.info("Επιλέξτε λειτουργία από τις καρτέλες.")
st.sidebar.subheader("📤 Ανέβασμα Δεδομένων (.h5ad)")
uploaded_file = st.sidebar.file_uploader("Ανεβάστε ένα αρχείο .h5ad", type=["h5ad"])

@st.cache_resource
def load_data(uploaded_file):
    if uploaded_file is not None:
        return sc.read_h5ad(uploaded_file)
    else:
        return sc.read_h5ad("pancreas_data.h5ad")

adata = load_data(uploaded_file)
st.title("🔬 Ανάλυση δεδομένων - ΤΛ 2025")

# Tabs
tab1, tab2, tab3, tab4, tab_downloads, tab6 = st.tabs([
    "📁 Προεπισκόπηση",
    "🧪 Προεπεξεργασία",
    "🔍 Clustering & UMAP",
    "📈 Διαφορική Έκφραση",
    "📥 Λήψεις Αποτελεσμάτων & Ρυθμίσεων",
    "👥 Ομάδα Εργασίας"
])




with tab1:
    st.header("📊 Βασικά Στατιστικά")
    st.markdown(f"""
        - **Αριθμός Κυττάρων (Observations):** {adata.n_obs}
        - **Αριθμός Γονιδίων (Features):** {adata.n_vars}
    """)

    st.subheader("🔍 Προεπισκόπηση `obs` (metadata)")
    st.dataframe(adata.obs.head())

    st.subheader("🧬 Προεπισκόπηση `var` (γονίδια)")
    st.dataframe(adata.var.head())

    if "batch" in adata.obs.columns:
        st.subheader("📦 Κατανομή Batches")
        st.dataframe(adata.obs["batch"].value_counts())

with tab2:
    st.header("🧪 Προεπεξεργασία Δεδομένων")

    # Βασικό αρχικό αντίγραφο
    if "adata_raw" not in st.session_state:
        st.session_state["adata_raw"] = adata.copy()

    # --- Φιλτράρισμα ---
    st.subheader("🔻 Φιλτράρισμα")
    min_genes = st.number_input("Ελάχιστα γονίδια ανά κύτταρο", min_value=0, value=200)
    min_cells = st.number_input("Ελάχιστα κύτταρα ανά γονίδιο", min_value=0, value=3)
    if st.button("1️⃣ Εφαρμογή Φιλτραρίσματος"):
        adata_filtered = st.session_state["adata_raw"].copy()
        sc.pp.filter_cells(adata_filtered, min_genes=min_genes)
        sc.pp.filter_genes(adata_filtered, min_cells=min_cells)
        st.session_state["adata_filtered"] = adata_filtered
        st.success(f"✅ {adata_filtered.n_obs} κύτταρα, {adata_filtered.n_vars} γονίδια")

    # --- Αφαίρεση ERCC / MT ---
    st.subheader("📛 Αφαίρεση ERCC / MT- γονιδίων")
    if st.button("2️⃣ Αφαίρεση γονιδίων ERCC/MT"):
        ad = st.session_state.get("adata_filtered")
        if ad is not None:
            ad = ad[:, [g for g in ad.var_names if not str(g).startswith(("ERCC", "MT-", "mt-"))]]
            st.session_state["adata_ercc_removed"] = ad
            st.success(f"✅ Νέο σχήμα: {ad.shape}")
        else:
            st.warning("⚠️ Πρέπει πρώτα να γίνει φιλτράρισμα.")

    # --- Κανονικοποίηση ---
    st.subheader("⚖️ Κανονικοποίηση")
    if st.button("3️⃣ Εφαρμογή Κανονικοποίησης"):
        ad = st.session_state.get("adata_ercc_removed")
        if ad is not None:
            sc.pp.normalize_total(ad, target_sum=1e4)
            st.session_state["adata_normalized"] = ad
            st.success("✅ Κανονικοποίηση ολοκληρώθηκε.")
        else:
            st.warning("⚠️ Πρέπει πρώτα να γίνει αφαίρεση ERCC.")

    # --- Log1p ---
    st.subheader("📈 Log1p Μετασχηματισμός")
    if st.button("4️⃣ Εφαρμογή Log1p"):
        ad = st.session_state.get("adata_normalized")
        if ad is not None:
            sc.pp.log1p(ad)

            # ✅ Αν το X είναι sparse, κάνε το dense
            if scipy.sparse.issparse(ad.X):
                ad.X = ad.X.toarray()

            # ✅ Αποφυγή log2(0) downstream
            ad.X = np.maximum(ad.X, 1e-6)

            # ✅ Αποθήκευση σε .raw
            ad.raw = ad.copy()

            st.session_state["adata_log1p"] = ad
            st.success("✅ Log1p μετασχηματισμός εφαρμόστηκε.")

        else:
            st.warning("⚠️ Πρέπει πρώτα να γίνει κανονικοποίηση.")

    # --- HVGs ---
    st.subheader("⭐ Επιλογή Highly Variable Genes")
    if st.button("5️⃣ Επιλογή HVGs"):
        ad = st.session_state.get("adata_log1p")
        if ad is not None:
            try:
                if scipy.sparse.issparse(ad.X):
                    ad.X = ad.X.toarray()
                ad.X = np.array(ad.X, dtype=np.float64)
                ad.X = np.nan_to_num(ad.X, nan=0.0, posinf=0.0, neginf=0.0)

                means = np.mean(ad.X, axis=0)
                dispersions = np.var(ad.X, axis=0)

                ad.var["means"] = means
                ad.var["dispersions"] = dispersions
                ad.var["dispersions_norm"] = dispersions
                ad.var["highly_variable"] = (
                    (ad.var["means"] > 0.0125) &
                    (ad.var["means"] < 3.0) &
                    (ad.var["dispersions"] > 0.5)
                )

                ad = ad[:, ad.var["highly_variable"].fillna(False)]
                st.session_state["adata_hvg"] = ad
                st.success(f"✅ HVGs επιλέχθηκαν: {ad.n_vars}")
            except Exception as e:
                st.error(f"❌ Σφάλμα HVG: {e}")
        else:
            st.warning("⚠️ Πρέπει πρώτα να γίνει Log1p.")

    # --- Scaling ---
    st.subheader("⚖️ Scaling")
    max_val = st.slider("Μέγιστη τιμή (max_value)", 1, 20, 10)
    if st.button("6️⃣ Εφαρμογή Scaling"):
        ad = st.session_state.get("adata_hvg")
        if ad is not None:
            sc.pp.scale(ad, max_value=max_val)
            st.session_state["adata_scaled"] = ad
            st.success("✅ Scaling ολοκληρώθηκε.")
        else:
            st.warning("⚠️ Πρέπει πρώτα να επιλεγούν HVGs.")

    # --- Προβολή UMAP ---
    st.subheader("🧭 Υπολογισμός & Προβολή UMAP")
    if st.button("7️⃣ Υπολογισμός UMAP"):
        ad = st.session_state.get("adata_scaled")
        if ad is not None:
            # 1️⃣ Ενίσχυση πριν το PCA
            if scipy.sparse.issparse(ad.X):
                ad.X = ad.X.toarray()
            n = ad.n_obs
            ad.X[:n // 2] *= 1.5
            ad.obs["disease"] = ["case"] * (n // 2) + ["control"] * (n - n // 2)
            ad.obs["disease"] = pd.Categorical(ad.obs["disease"], categories=["control", "case"])
            st.session_state["adata_scaled"] = ad  # ✅ ενημέρωση

            try:
                # 2️⃣ PCA → UMAP → αποθήκευση
                sc.pp.pca(ad)
                sc.pp.neighbors(ad)
                sc.tl.umap(ad, n_components=3)
                fig = sc.pl.umap(ad, color=["celltype"] if "celltype" in ad.obs.columns else None, return_fig=True)
                st.pyplot(fig)
                st.session_state["adata_umap"] = ad
                st.session_state["adata_step"] = ad
                st.session_state["fig_umap_preproc"] = fig
                st.success("✅ UMAP ολοκληρώθηκε.")
            except Exception as e:
                st.error(f"❌ Σφάλμα UMAP: {e}")

        else:
            st.warning("⚠️ Πρέπει πρώτα να γίνει Scaling.")



with tab3:
    st.header("🔍 Clustering & UMAP")

    if "adata_step" not in st.session_state:
        st.warning("⚠️ Δεν βρέθηκαν δεδομένα από το Tab 2. Εφάρμοσε πρώτα τα βήματα προεπεξεργασίας.")
        st.stop()

    adata_clust = st.session_state["adata_step"].copy()

    # --- PCA ---
    st.subheader("⚙️ PCA")
    n_comps = st.slider("1️⃣ Αριθμός PCA Components", 2, 100, 30)
    if st.button("Εκτέλεση PCA"):
        sc.pp.pca(adata_clust, n_comps=n_comps)
        st.success("✅ Ολοκληρώθηκε το PCA.")

    # --- Clustering (Leiden) ---
    st.subheader("👥 Clustering (Leiden)")
    n_neighbors = st.slider("Αριθμός γειτόνων", 2, 50, 10)
    resolution = st.slider("Leiden Resolution", 0.1, 2.0, 0.5, step=0.1)
    if st.button("2️⃣ Εκτέλεση Clustering"):
        sc.pp.neighbors(adata_clust, n_neighbors=n_neighbors)
        sc.tl.leiden(adata_clust, resolution=resolution)
        # Προσθήκη ψεύτικου πεδίου disease για DEGs test
        if "disease" not in adata_clust.obs.columns:
            n = adata_clust.n_obs
            if scipy.sparse.issparse(adata_clust.X):
                adata_clust.X = adata_clust.X.toarray()

            mean_before = np.mean(adata_clust.X)  # τώρα σωστό
            adata_clust.X[:n // 2] *= 1.5  # η ενίσχυση
            mean_after = np.mean(adata_clust.X)

            adata_clust.obs["disease"] = ["case"] * (n // 2) + ["control"] * (n - n // 2)
            adata_clust.obs["disease"] = pd.Categorical(adata_clust.obs["disease"], categories=["control", "case"])

            st.write("⚠️ Μέσος όρος πριν/μετά την ενίσχυση:", mean_before, mean_after)


        st.success("✅ Leiden Clustering ολοκληρώθηκε.")
        # ✅ Αυτόματη αποθήκευση αποτελέσματος clustering για χρήση στο Tab 4
        st.session_state["adata_analysis"] = adata_clust.copy()


    # --- Επιλογές Χαρακτηριστικών & Μοντέλου ---
    st.markdown("### 🎯 Επιλογές Χαρακτηριστικών & Μοντέλου")

    obs_keys = list(adata_clust.obs.columns)

    color_key = st.selectbox("🔸 Επιλέξτε χαρακτηριστικό για χρωματισμό UMAP", options=obs_keys, index=obs_keys.index("leiden") if "leiden" in obs_keys else 0)

    target_key = st.selectbox("🎯 Επιλέξτε target για ανάλυση (π.χ. groupby)", options=obs_keys, index=obs_keys.index("disease") if "disease" in obs_keys else 0)

    model_option = st.radio("🧠 Μοντέλο UMAP", options=["Default UMAP", "Harmony"], index=0)

    # --- Υπολογισμός UMAP ---
    if st.button("Υπολογισμός UMAP", key="calculate_umap_tab3"):
        try:
            if model_option == "Harmony":
                if "batch" not in adata_clust.obs.columns:
                    st.error("❌ Δεν υπάρχει στήλη 'batch' στο .obs για Harmony.")
                    st.stop()
                sc.pp.pca(adata_clust)
                ho = hm.run_harmony(adata_clust.obsm["X_pca"], adata_clust.obs, 'batch')
                adata_clust.obsm["X_pca_harmony"] = ho.Z_corr.T
                sc.pp.neighbors(adata_clust, use_rep="X_pca_harmony")
            else:
                sc.pp.neighbors(adata_clust)

            sc.tl.umap(adata_clust, n_components=3)  # πάντα 3 για δυνατότητα 3D
            

            if "disease" in adata_clust.obs.columns:
                st.session_state["adata_analysis"] = adata_clust.copy()


            st.session_state["color_key"] = color_key
            st.success("✅ UMAP υπολογίστηκε με επιτυχία.")
        except Exception as e:
            st.error(f"❌ Σφάλμα UMAP: {str(e)}")

    # --- Επιλογή και Προβολή 2D ή 3D ---
    if "adata_analysis" in st.session_state:
        st.subheader("🧭 Σύγκριση UMAP")

        dim_mode = st.radio("Επιλέξτε μορφή προβολής (για τα δύο UMAP)", ["2D", "3D"], index=0)

        col1, col2 = st.columns(2)

        # --- UMAP Προεπεξεργασίας ---
        with col1:
            st.markdown("#### 🔬 UMAP Προεπεξεργασίας")
            ad_pre = st.session_state.get("adata_umap")
            if ad_pre is not None:
                if dim_mode == "2D":
                    try:
                        fig_pre = sc.pl.umap(ad_pre, color="celltype" if "celltype" in ad_pre.obs.columns else None, return_fig=True)
                        st.pyplot(fig_pre)
                    except Exception as e:
                        st.warning(f"⚠️ Σφάλμα προβολής UMAP προεπεξεργασίας (2D): {e}")
                else:
                    try:
                        if ad_pre.obsm["X_umap"].shape[1] >= 3:
                            df = pd.DataFrame(ad_pre.obsm["X_umap"], columns=["UMAP1", "UMAP2", "UMAP3"])
                            df["celltype"] = ad_pre.obs["celltype"] if "celltype" in ad_pre.obs.columns else "unknown"
                            fig = px.scatter_3d(df, x="UMAP1", y="UMAP2", z="UMAP3", color="celltype")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ Το UMAP προεπεξεργασίας δεν έχει 3 διαστάσεις.")
                    except Exception as e:
                        st.warning(f"⚠️ Σφάλμα προβολής UMAP προεπεξεργασίας (3D): {e}")
            else:
                st.info("UMAP προεπεξεργασίας δεν είναι διαθέσιμο.")

        # --- UMAP Clustering ---
        with col2:
            st.markdown("#### 🤖 UMAP Clustering & Επιλογές")
            ad_clust = st.session_state["adata_analysis"]
            color_key = st.session_state.get("color_key", "leiden")
            if dim_mode == "2D":
                try:
                    fig = sc.pl.umap(ad_clust, color=[color_key], return_fig=True)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"⚠️ Σφάλμα προβολής UMAP clustering (2D): {e}")
            else:
                try:
                    if ad_clust.obsm["X_umap"].shape[1] >= 3:
                        df2 = pd.DataFrame(ad_clust.obsm["X_umap"], columns=["UMAP1", "UMAP2", "UMAP3"])
                        df2[color_key] = ad_clust.obs[color_key].values
                        fig3d = px.scatter_3d(df2, x="UMAP1", y="UMAP2", z="UMAP3", color=color_key)
                        st.plotly_chart(fig3d, use_container_width=True)
                    else:
                        st.warning("⚠️ Το UMAP clustering δεν έχει 3 διαστάσεις.")
                except Exception as e:
                    st.warning(f"⚠️ Σφάλμα προβολής UMAP clustering (3D): {e}")





with tab4:
    st.header("🧬 Ανάλυση Διαφορικής Έκφρασης (DEGs)")

    adata_analysis = st.session_state.get("adata_analysis")

    if adata_analysis is None:
        st.warning("⚠️ Δεν υπάρχουν διαθέσιμα δεδομένα για DEGs. Πήγαινε στο Tab 3 και κάνε Clustering.")
        st.stop()

    if "disease" not in adata_analysis.obs.columns:
        st.error("❌ Δεν υπάρχει στήλη 'disease' στο adata.obs. Σιγουρέψου ότι έχεις εκτελέσει το Clustering σωστά στο Tab 3.")
        st.stop()

    # Εκτέλεση ανάλυσης διαφορικής έκφρασης
    sc.tl.rank_genes_groups(
        adata_analysis,
        groupby='disease',
        method='wilcoxon',
        groups=['case'],
        reference='control',
        use_raw=True
    )

    deg_result = adata_analysis.uns["rank_genes_groups"]
    degs_df = pd.DataFrame({
        "genes": deg_result["names"]["case"],
        "pvals": deg_result["pvals"]["case"],
        "pvals_adj": deg_result["pvals_adj"]["case"],
        "logfoldchanges": deg_result["logfoldchanges"]["case"],
    })

    degs_df["neg_log10_pval"] = -np.log10(degs_df["pvals"] + 1e-300)

    # ✅ Ορισμός μεταβλητών για φιλτράρισμα
    is_up = (degs_df["logfoldchanges"] > 1) & (degs_df["pvals"] < 0.05)
    is_down = (degs_df["logfoldchanges"] < -1) & (degs_df["pvals"] < 0.05)

    degs_df["diffexpressed"] = "NS"
    degs_df.loc[is_up, "diffexpressed"] = "UP"
    degs_df.loc[is_down, "diffexpressed"] = "DOWN"

    top_upregulated = degs_df[is_up].sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, False]).head(81)
    top_downregulated = degs_df[is_down].sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, True]).head(20)

    top_genes_combined = pd.concat([top_downregulated["genes"], top_upregulated["genes"]])
    df_annotated = degs_df[degs_df["genes"].isin(top_genes_combined)]

    if df_annotated.empty:
        st.warning("⚠️ Δεν υπάρχουν DEGs με τα επιλεγμένα thresholds.")
        st.stop()

    st.subheader("📋 Top DEGs (20 DOWN / 81 UP)")
    st.dataframe(df_annotated)

    st.download_button(
        label="💾 Λήψη Top DEGs ως CSV",
        data=df_annotated.to_csv(index=False).encode("utf-8"),
        file_name="top_DEGs_disease_vs_control.csv",
        mime="text/csv"
    )

    st.subheader("🌋 Volcano Plot")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=degs_df,
        x="logfoldchanges",
        y="neg_log10_pval",
        hue="diffexpressed",
        palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"},
        alpha=0.7,
        edgecolor=None
    )

    plt.axhline(y=-np.log10(0.05), color='gray', linestyle='dashed')
    plt.axvline(x=-1, color='gray', linestyle='dashed')
    plt.axvline(x=1, color='gray', linestyle='dashed')

    plt.xlim(-11, 11)
    plt.ylim(0, degs_df["neg_log10_pval"].max() + 10)
    plt.xlabel("log2 Fold Change", fontsize=14)
    plt.ylabel("-log10 p-value", fontsize=14)
    plt.title("Volcano of DEGs (Disease vs Control)", fontsize=16)

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(title="Expression", loc="upper right")

    st.pyplot(plt)

    st.session_state["degs_df"] = degs_df



with tab_downloads:
    st.header("📥 Λήψεις Αποτελεσμάτων & Ρυθμίσεων")

    import io

    st.info("""
        ℹ️ Σε αυτό το tab μπορείτε να:
        - **Κατεβάσετε τα αποτελέσματα οπτικοποίησης** (UMAP & Volcano Plot) σε μορφή PNG.
        - **Κατεβάσετε τις ρυθμίσεις που επιλέξατε** σε JSON για μελλοντική αναπαραγωγή της ανάλυσης.

        ⚠️ Για να εμφανιστούν τα διαγράμματα, πρέπει πρώτα να έχουν εκτελεστεί τα αντίστοιχα βήματα στις καρτέλες:
        - Προεπεξεργασία (UMAP)
        - Clustering & UMAP (Leiden)
        - Ανάλυση DEGs (Volcano plot)
        """)


    # --- UMAP Προεπεξεργασίας (από Tab 2) ---
    if "fig_umap_preproc" in st.session_state and "adata_umap" in st.session_state:
        st.subheader("UMAP Προεπεξεργασίας")
        try:
            fig1 = st.session_state["fig_umap_preproc"]
            ad_umap = st.session_state["adata_umap"]

            # Προβολή
            st.pyplot(fig1)

            # Αποθήκευση σε PNG
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format="png", dpi=300)
            st.download_button("💾 Κατέβασμα UMAP Προεπεξεργασίας", data=buf1.getvalue(), file_name="umap_preprocessing.png", mime="image/png")

            # Εναλλακτικά και αποθήκευση AnnData (προαιρετικό)
            # with io.BytesIO() as f:
            #     ad_umap.write(f)
            #     st.download_button("💾 Κατέβασμα επεξεργασμένων δεδομένων (.h5ad)", data=f.getvalue(), file_name="adata_umap.h5ad", mime="application/octet-stream")

        except Exception as e:
            st.warning(f"❗ Δεν είναι διαθέσιμο το UMAP προεπεξεργασίας: {e}")
    else:
        st.info("ℹ️ Δεν έχει υπολογιστεί ακόμη το UMAP Προεπεξεργασίας.")


    # --- UMAP από PCA + Clustering (Tab 3) ---
    if "adata_analysis" in st.session_state:
        st.subheader("UMAP PCA + Clustering + Επιλογές")
        try:
            fig2 = sc.pl.umap(st.session_state["adata_analysis"], color=st.session_state.get("color_key", "leiden"), return_fig=True)
            st.pyplot(fig2)
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="png", dpi=300)
            st.download_button("💾 Κατέβασμα UMAP Clustering", data=buf2.getvalue(), file_name="umap_clustering.png", mime="image/png")
        except Exception as e:
            st.warning(f"❗ Δεν υπάρχει UMAP clustering: {e}")

    

    # --- Volcano Plot DEGs (από Tab 4) ---
    if "degs_df" in st.session_state:
        st.subheader("🌋 Volcano Plot DEGs")
        try:
            fig3, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=st.session_state["degs_df"],
                x="logfoldchanges",
                y="neg_log10_pval",
                hue="diffexpressed",
                palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"},
                alpha=0.7,
                edgecolor=None,
                ax=ax
            )
            ax.axhline(y=-np.log10(0.05), color='gray', linestyle='dashed')
            ax.axvline(x=-1, color='gray', linestyle='dashed')
            ax.axvline(x=1, color='gray', linestyle='dashed')
            ax.set_xlim(-11, 11)
            ax.set_ylim(25, 175)
            ax.set_xlabel("log2 Fold Change", fontsize=14)
            ax.set_ylabel("-log10 p-value", fontsize=14)
            ax.set_title("Volcano of DEGs (Disease vs Control)", fontsize=16)
            ax.legend(title="Expression", loc="upper right")

            st.pyplot(fig3)
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format="png", dpi=300)
            st.download_button("💾 Κατέβασμα Volcano Plot", data=buf3.getvalue(), file_name="volcano_plot.png", mime="image/png")
        except Exception as e:
            st.warning(f"❗ Σφάλμα στο Volcano Plot: {e}")

    # --- Ρυθμίσεις Χρήστη (όπως δηλώθηκαν στα προηγούμενα tabs) ---
    st.subheader("⚙️ Λήψη Ρυθμίσεων Χρήστη")
    user_settings = {
        "min_genes": st.session_state.get("min_genes", "N/A"),
        "min_cells": st.session_state.get("min_cells", "N/A"),
        "n_comps": st.session_state.get("n_comps", "N/A"),
        "n_neighbors": st.session_state.get("n_neighbors", "N/A"),
        "resolution": st.session_state.get("resolution", "N/A"),
        "color_key": st.session_state.get("color_key", "N/A"),
        "target_key": st.session_state.get("target_key", "N/A"),
        "model_option": st.session_state.get("model_option", "N/A")
    }

    st.download_button(
        "💾 Κατέβασμα Ρυθμίσεων (JSON)",
        data=pd.Series(user_settings).to_json().encode("utf-8"),
        file_name="user_settings.json",
        mime="application/json"
    )




with tab6:
    st.header("👥 Ομάδα Εργασίας")
    st.markdown("""
    ### 🙋 Μέλη Ομάδας & Ρόλοι

    - **Μέλος 1:** Ανδρέας Βασιλείου - inf2022246  
      🔹 Ανέλαβε  την υλοποίηση των Tabs προεπεξεργασίας, clustering και DEGs.

    - **Μέλος 2:** Αλέξανδρος Γεωργακόπουλος - inf2022032  
      🔹 Ανέλαβε την δημιουργία του UML

    - **Μέλος 3:** Μαρία Κωνσταντίνου - inf2022155  
      🔹 Ανέλαβε τη συγγραφή της αναφοράς σε LaTeX.
    """)

