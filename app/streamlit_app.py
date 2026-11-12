import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn import __version__ as sklearn_version

from soutnance_1.schema import FEATURE_COLUMNS, NOMINAL_CATEGORIES, ORDINAL_CATEGORIES

RESULTS_DIR = Path(os.getenv("STUDENT_RESULTS_ROOT", "runtime/results"))
RISK_NAMES = {"Low": "Faible", "Medium": "Moyen", "High": "Élevé"}
TEAL = "#117D74"
METRIC_NAMES = {
    "accuracy": "Exactitude",
    "precision_macro": "Précision macro",
    "recall_macro": "Rappel macro",
    "f1_macro": "F1 macro",
}
CATEGORY_NAMES = {
    "Low": "Faible", "Moderate": "Modérée", "High": "Élevée",
    "Poor": "Faible", "Average": "Moyenne", "Good": "Bonne",
    "Female": "Femme", "Male": "Homme", "Non-binary": "Non binaire",
    "No": "Non", "Yes": "Oui",
}


@st.cache_resource(max_entries=2)
def load_model(path: str):
    return joblib.load(path)


def published_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(ZoneInfo("Africa/Casablanca")).strftime("%d/%m/%Y à %H:%M")


def show_performance(run: dict) -> None:
    evaluations = {"Validation croisée": run["validation"]}
    final = run["final_test"]
    if final is not None:
        evaluations["Test final figé"] = final["metrics"]

    selected = st.radio("Évaluation", evaluations, horizontal=True)
    metrics = evaluations[selected]
    if selected == "Test final figé":
        st.caption(f"{final['model']} · {published_time(final['published_at'])} · {final['run_id']}")
    else:
        st.caption("Prédictions hors pli sur 5 plis. La sélection des hyperparamètres utilise ces mêmes plis.")

    for column, (key, label) in zip(st.columns(4), METRIC_NAMES.items()):
        column.metric(label, f"{metrics[key]:.1%}")
    st.caption(f"{metrics['observations']:,} observations évaluées".replace(",", " "))

    left, right = st.columns([1.15, 1])
    classes = metrics["classes"]
    labels = [RISK_NAMES[label] for label in classes]
    with left:
        st.subheader("Matrice de confusion")
        fig = px.imshow(
            metrics["confusion_matrix"], x=labels, y=labels, text_auto=True,
            labels={"x": "Classe prédite", "y": "Classe réelle", "color": "Effectif"},
            color_continuous_scale=["#EAF4F2", TEAL, "#0C4350"],
        )
        fig.update_layout(height=330, margin=dict(l=12, r=12, t=16, b=12), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        st.subheader("Scores par classe")
        report = pd.DataFrame.from_dict(metrics["per_class"], orient="index").reindex(classes)
        report.index = pd.Index(labels, name="Classe")
        report = report.rename(columns={
            "precision": "Précision", "recall": "Rappel", "f1-score": "F1", "support": "Effectif",
        })
        report["Effectif"] = report["Effectif"].astype(int)
        st.dataframe(report.round(3), width="stretch")

    st.subheader("Comparaison des modèles")
    comparison = pd.DataFrame(run["comparison"])
    chart = px.bar(comparison, x="f1_macro", y="model", orientation="h", text_auto=".3f", color_discrete_sequence=[TEAL])
    chart.update_layout(
        height=255, margin=dict(l=12, r=12, t=12, b=12),
        xaxis_title="F1 macro moyen en validation croisée", yaxis_title=None,
        yaxis_autorange="reversed", xaxis_range=[0, 1],
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
    with st.expander("Scores par modèle et optimisation"):
        st.dataframe(comparison.rename(columns={
            "model": "Modèle", "f1_std": "Écart-type F1", "roc_auc_ovr": "ROC-AUC macro OvR", **METRIC_NAMES,
        }).round(3), hide_index=True, width="stretch")
        tuning = pd.DataFrame(run["tuning"]).rename(columns={
            "model": "Modèle", "f1_macro": "F1 macro après optimisation", "params": "Paramètres retenus",
        })
        tuning["Paramètres retenus"] = tuning["Paramètres retenus"].map(
            lambda params: ", ".join(f"{key.removeprefix('classifier__')}={value}" for key, value in params.items())
        )
        st.dataframe(tuning.round(3), hide_index=True, width="stretch")


def select_category(column: str, label: str):
    options = (ORDINAL_CATEGORIES | NOMINAL_CATEGORIES)[column]
    return st.selectbox(label, options, format_func=lambda value: CATEGORY_NAMES.get(value, str(value)))


def show_simulation(run: dict) -> None:
    st.subheader("Simuler un profil étudiant")
    st.info("Démonstration pédagogique — résultat non clinique, non diagnostique et non destiné à une décision individuelle.")
    st.caption("Utilise un profil fictif. Les valeurs du formulaire restent dans la session.")

    with st.form("synthetic_student", border=False):
        left, right = st.columns(2)
        with left:
            st.markdown("**Études**")
            age = st.number_input("Âge", 17, 29, 21)
            year = select_category("Year_of_Study", "Année d’études")
            gender = select_category("Gender", "Genre")
            study = st.number_input("Heures d’étude par semaine", 0.0, 168.0, 20.0, step=0.5)
            gpa = st.number_input("GPA sur 4", 0.0, 4.0, 2.8, step=0.05)
            attendance = st.number_input("Assiduité (%)", 0.0, 100.0, 85.0, step=1.0)
            job = select_category("Part_Time_Job", "Emploi à temps partiel")
        with right:
            st.markdown("**Habitudes et contexte**")
            sleep = st.number_input("Heures de sommeil par jour", 0.0, 24.0, 7.0, step=0.5)
            screen = st.number_input("Temps d’écran par jour (h)", 0.0, 24.0, 6.0, step=0.5)
            social_media = st.number_input("Réseaux sociaux par jour (h)", 0.0, 24.0, 3.0, step=0.5)
            physical = select_category("Physical_Activity", "Activité physique")
            diet = select_category("Diet_Quality", "Qualité de l’alimentation")
            stress = st.number_input("Stress financier sur 10", 0.0, 10.0, 5.0, step=0.5)
            support = st.number_input("Soutien familial sur 10", 0.0, 10.0, 6.0, step=0.5)
            history = select_category("Previous_Mental_Health_History", "Antécédents de santé mentale")
        submitted = st.form_submit_button("Calculer la prédiction", type="primary")

    if not submitted:
        return
    if social_media > screen:
        st.error("Le temps passé sur les réseaux sociaux doit être inférieur ou égal au temps d’écran total.")
        return
    if run["sklearn_version"] != sklearn_version:
        st.error("La version de scikit-learn a changé. Reconstruis les images et relance le DAG.")
        return

    profile = {
        "Age": age, "Year_of_Study": year, "Gender": gender, "Study_Hours_Per_Week": study,
        "GPA": gpa, "Attendance_Rate": attendance, "Part_Time_Job": job, "Sleep_Hours": sleep,
        "Screen_Time_Hours": screen, "Social_Media_Hours": social_media, "Physical_Activity": physical,
        "Diet_Quality": diet, "Financial_Stress": stress, "Family_Support": support,
        "Previous_Mental_Health_History": history,
    }
    sample = pd.DataFrame([profile], columns=FEATURE_COLUMNS)
    try:
        model = load_model(str(RESULTS_DIR / run["model_file"]))
        prediction = model.predict(sample)[0]
        scores = model.predict_proba(sample)[0]
    except (OSError, ValueError):
        st.error("Le modèle publié est indisponible. Vérifie la tâche evaluate_model dans Airflow.")
        return

    st.success(f"Classe estimée pour ce profil fictif · {RISK_NAMES[prediction]}")
    probabilities = pd.DataFrame({"Classe": [RISK_NAMES[label] for label in model.classes_], "Score": scores})
    st.bar_chart(probabilities, x="Classe", y="Score", color=TEAL, height=230)


def show_data(summary: dict) -> None:
    left, middle, right = st.columns(3)
    left.metric("Observations", f"{summary['observations']:,}".replace(",", " "))
    middle.metric("Variables explicatives", summary["features"])
    right.metric("Valeurs manquantes", sum(summary["missing_by_feature"].values()))

    st.subheader("Répartition de la cible")
    counts = pd.Series(summary["class_counts"]).rename(index=RISK_NAMES).rename("Effectif")
    st.bar_chart(counts, color=TEAL, height=245)

    st.subheader("Valeurs manquantes par variable")
    missing = pd.Series(summary["missing_by_feature"], name="Valeurs manquantes").sort_values(ascending=False)
    st.dataframe(missing.rename_axis("Variable"), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Risque étudiant | Évaluation du modèle", layout="wide")
    st.title("Risque étudiant")
    try:
        run = json.loads((RESULTS_DIR / "latest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.info("En attente des résultats. Lance student_risk_pipeline dans Airflow, puis recharge cette page.")
        return
    except (OSError, json.JSONDecodeError):
        st.error("Le rapport est illisible. Vérifie la tâche evaluate_model dans Airflow.")
        return

    st.caption(f"{run['model']} · Mis à jour le {published_time(run['published_at'])}")
    performance, simulation, data = st.tabs(["Performances", "Simulation", "Données"])
    with performance:
        show_performance(run)
    with simulation:
        show_simulation(run)
    with data:
        show_data(run["data"])


if __name__ == "__main__":
    main()
