"""Local image upload demo: streamlit run app.py -- --output artifacts"""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

from classifier import image_to_array

@st.cache_resource
def open_model(folder: str):
    path = Path(folder)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(path / "model.keras")
    return model, metadata


def main():
    st.set_page_config(page_title="Fashion Image Classification", page_icon="👕")
    st.title("Product Image Classification MVP")
    st.caption("TensorFlow × Keras · image category classification")
    args = sys.argv[1:]
    output = args[args.index("--output") + 1] if "--output" in args else "artifacts"
    try:
        model, metadata = open_model(output)
    except (OSError, ValueError, IndexError) as error:
        st.error(f"Please train a model first: {error}")
        st.stop()

    if metadata["dataset"] == "fashion":
        st.info("This demo uses 28×28 grayscale Fashion-MNIST articles. Real product photos or AOI defects need a separate model and evaluation.")
    else:
        st.info("Category predictions are valid only for product types represented in this model's training data.")

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
    use_sample = (
        st.checkbox("Try the included Fashion-MNIST sample", value=False)
        if metadata["dataset"] == "fashion" else False
    )
    if uploaded is None and not use_sample:
        return
    source = uploaded if uploaded is not None else Path(__file__).with_name("sample.png")
    try:
        with Image.open(source) as image:
            st.image(image, caption="Uploaded image" if uploaded is not None else "Held-out test sample · true class: Ankle boot", width=280)
            scores = model.predict(image_to_array(image, metadata), verbose=0)[0]
    except (OSError, ValueError) as error:
        st.error(f"Could not classify this image: {error}")
        return
    ranked = np.argsort(scores)[::-1][:3]
    st.subheader(f"Prediction: {metadata['classes'][ranked[0]]}")
    for index in ranked:
        st.write(f"{metadata['classes'][index]}: {scores[index]:.1%}")
    st.caption("Displayed softmax scores are not calibrated probabilities. Review uncertain cases manually.")


if __name__ == "__main__":
    main()
