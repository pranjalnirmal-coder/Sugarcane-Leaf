"""Streamlit dashboard for sugarcane leaf disease detection."""
from __future__ import annotations

import json

import streamlit as st

from config import CONFIDENCE_THRESHOLD, LABELS_PATH, MODEL_PATH, OUTPUT_DIR
from src import database
from src.advisory import get_advisory
from src.gradcam import create_overlay, generate_gradcam
from src.predict import load_model, predict_image
from src.preprocessing import ImageValidationError, open_validated_image, preprocess_image

st.set_page_config(page_title="Sugarcane Leaf Disease Detection", page_icon="🌱", layout="wide")


@st.cache_resource(show_spinner="Loading trained model…")
def cached_model():
    return load_model()


def login_screen() -> None:
    st.title("🌱 Sugarcane Leaf Disease Detection and Advisory System")
    st.write("Sign in to analyze an image and keep a private local prediction history.")
    login_tab, create_tab = st.tabs(["Sign in", "Create local account"])
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            user_id = database.authenticate(username, password)
            if user_id is None:
                st.error("Username or password is incorrect.")
            else:
                st.session_state.user_id = user_id
                st.session_state.username = username.strip().lower()
                st.rerun()
    with create_tab:
        with st.form("register_form"):
            username = st.text_input("Choose a username (3–40 characters)", key="new_user")
            password = st.text_input("Choose a password (at least 8 characters)", type="password", key="new_password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account")
        if submitted:
            if password != confirm:
                st.error("The passwords do not match.")
            else:
                try:
                    user_id = database.create_user(username, password)
                    st.session_state.user_id = user_id
                    st.session_state.username = username.strip().lower()
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    st.caption("Accounts and history are stored in a local SQLite file on this computer.")


def show_history(user_id: int) -> None:
    history = database.get_history(user_id)
    if not history:
        st.info("No saved predictions yet.")
        return
    st.dataframe([{
        "Date and time": row["created_at"], "Image": row["image_name"],
        "Prediction": row["class_name"], "Confidence": f"{row['confidence']:.1%}",
        "Status": "Uncertain" if row["uncertain"] else "Above threshold",
    } for row in history], use_container_width=True, hide_index=True)


def show_model_evaluation() -> None:
    """Show the real training and held-out evaluation artifacts, when available."""
    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    curves_path = OUTPUT_DIR / "training_curves.png"
    matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    if not metrics_path.is_file():
        st.info("Training results are not available yet. Train the model with `python -m src.train`.")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    accuracy = float(metrics["test_accuracy"])
    st.metric("Held-out test accuracy", f"{accuracy:.1%}",
              help="Share of images in the separate test partition classified correctly.")
    st.caption(f"Evaluation used {metrics['test_image_count']} held-out images. Confidence for one uploaded image is not the same as test accuracy.")

    if curves_path.is_file():
        st.image(str(curves_path), caption="Training and validation accuracy and loss by epoch", use_container_width=True)
    else:
        st.warning("Training curves image is missing.")

    if matrix_path.is_file():
        st.image(str(matrix_path), caption="Confusion matrix for the held-out test set", use_container_width=True)
    else:
        st.warning("Confusion matrix image is missing.")

    report = metrics.get("classification_report", {})
    class_rows = [
        {"Class": name, "Precision": values.get("precision"),
         "Recall": values.get("recall"), "F1 score": values.get("f1-score"),
         "Test images": int(values.get("support", 0))}
        for name, values in report.items()
        if isinstance(values, dict) and "precision" in values
    ]
    if class_rows:
        st.markdown("**Per-class test results**")
        st.dataframe(class_rows, use_container_width=True, hide_index=True,
                     column_config={
                         "Precision": st.column_config.NumberColumn(format="%.1%"),
                         "Recall": st.column_config.NumberColumn(format="%.1%"),
                         "F1 score": st.column_config.NumberColumn(format="%.1%"),
                     })


def main() -> None:
    database.initialize_database()
    if "user_id" not in st.session_state:
        login_screen()
        return

    with st.sidebar:
        st.markdown(f"**Signed in as:** {st.session_state.username}")
        if st.button("Log out"):
            st.session_state.pop("user_id", None)
            st.session_state.pop("username", None)
            st.session_state.pop("analysis", None)
            st.rerun()
        st.divider()
        st.caption(f"Low-confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            st.warning("The trained model is not available yet.")
            st.code("python -m src.train", language="bash")

    st.title("Sugarcane Leaf Disease Detection and Advisory System")
    st.write("Upload a sugarcane leaf image to see the model prediction, confidence, visual explanation, and preliminary guidance.")
    

    if "upload_generation" not in st.session_state:
        st.session_state.upload_generation = 0
    upload_key = f"leaf_upload_{st.session_state.upload_generation}"
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"], key=upload_key)
    if uploaded is not None:
        try:
            original = open_validated_image(uploaded)
            st.image(original, caption=f"Uploaded image: {uploaded.name}", width=420)
        except ImageValidationError as exc:
            st.error(str(exc))
            original = None
    else:
        original = None

    left, right = st.columns([1, 1])
    with left:
        analyze = st.button("Analyze leaf", type="primary", disabled=uploaded is None or original is None)
    with right:
        if st.button("Clear image and result"):
            st.session_state.pop("analysis", None)
            st.session_state.upload_generation += 1
            st.rerun()

    if analyze and uploaded is not None and original is not None:
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            st.error("Predictions are unavailable until a model has been trained. Follow the training setup instructions in the README.")
        else:
            try:
                with st.spinner("Analyzing image…"):
                    model, class_names = cached_model()
                    result = predict_image(model, uploaded, class_names)
                    advisory = get_advisory(result["class_name"])
                    batch = preprocess_image(uploaded)
                    heatmap = generate_gradcam(model, batch, class_index=class_names.index(result["class_name"]))
                    overlay = create_overlay(original, heatmap)
                    database.save_prediction(st.session_state.user_id, uploaded.name,
                                             result["class_name"], result["confidence"],
                                             result["uncertain"], MODEL_PATH.name)
                    st.session_state.analysis = {
                        "result": result, "advisory": advisory,
                        "heatmap": heatmap, "overlay": overlay,
                        "original": original.copy(),
                    }
            except ImageValidationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Analysis could not be completed: {exc}")

    analysis = st.session_state.get("analysis")
    if analysis:
        result, advisory = analysis["result"], analysis["advisory"]
        st.divider()
        st.subheader(f"Model prediction (not a confirmed diagnosis): {advisory['display_name']}")
        st.metric("Model confidence", f"{result['confidence']:.1%}")
        if result["uncertain"]:
            st.warning(result["message"])
        st.caption("A softmax confidence score does not reliably detect every non-leaf or out-of-distribution image. Upload a clear sugarcane leaf photo and confirm important decisions with an expert.")
        original_col, heat_col, overlay_col = st.columns(3)
        with original_col:
            st.image(analysis["original"], caption="Original image", use_container_width=True)
        with heat_col:
            st.image(analysis["heatmap"], caption="Grad-CAM heatmap", clamp=True, use_container_width=True)
        with overlay_col:
            st.image(analysis["overlay"], caption="Heatmap overlay", use_container_width=True)
        st.caption("Grad-CAM highlights image regions that influenced the prediction; it does not prove that those regions contain disease.")
        st.subheader("Disease information and preliminary guidance")
        st.write(advisory["description"])
        advice_col, prevention_col = st.columns(2)
        with advice_col:
            st.markdown("**General management**")
            for item in advisory["management"]:
                st.write(f"• {item}")
        with prevention_col:
            st.markdown("**Prevention**")
            for item in advisory["prevention"]:
                st.write(f"• {item}")
        st.warning(advisory["expert"])
        st.markdown("**Sources**: " + " · ".join(f"[agricultural guidance]({url})" for url in advisory["sources"]))

    st.divider()
    with st.expander("Model training and evaluation"):
        show_model_evaluation()
    with st.expander("Prediction history"):
        show_history(st.session_state.user_id)
    st.caption("Supported model classes are determined by the dataset: Healthy, Mosaic, RedRot, Rust, and Yellow. The Yellow dataset label is not a specific disease diagnosis.")


if __name__ == "__main__":
    main()
