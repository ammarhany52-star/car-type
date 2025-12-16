# Gradio GUI for car type classification

import gradio as gr
import numpy as np
import tensorflow as tf
import cv2, os, random
from PIL import Image

BASE_IMAGES = "/kaggle/working/sample_images"


CLASS_FOLDERS = {
    "AM General Hummer SUV 2000": "AM_General_Hummer_SUV_2000",
    "Audi 100 Sedan 1994": "Audi_100_Sedan_1994",
    "Audi R8 Coupe 2012": "Audi_R8_Coupe_2012",
    "Chrysler Crossfire Convertible 2008": "Chrysler_Crossfire_Convertible_2008",
    "Ford Focus Sedan 2007": "Ford_Focus_Sedan_2007",
    "GMC Terrain SUV 2012": "GMC_Terrain_SUV_2012",
    "Hyundai Tucson SUV 2012": "Hyundai_Tucson_SUV_2012",
    "Jeep Wrangler SUV 2012": "Jeep_Wrangler_SUV_2012",
    "Lamborghini Diablo Coupe 2001": "Lamborghini_Diablo_Coupe_2001",
    "Volvo 240 Sedan 1993": "Volvo_240_Sedan_1993",
}

CLASS_NAMES = list(CLASS_FOLDERS.keys())


# LOAD MODELS
MODELS = {
    "ResNet50": {
        "model": tf.keras.models.load_model("resnet50_cars_final.keras"),
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
        "last_conv": "conv5_block3_out"
    },
    "EfficientNet": {
        "model": tf.keras.models.load_model("efficientnet_10classes_final.keras"),
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "last_conv": "top_conv"
    },
    "MobileNetV2": {
        "model": tf.keras.models.load_model("mobilenetv2_10classes_final.keras"),
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
        "last_conv": "Conv_1"
    }
}

# GRAD-CAM
def gradcam(model, last_conv, img_array, class_idx):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


# PREDICTION
def predict(img, model_name):
    cfg = MODELS[model_name]
    model = cfg["model"]

    img_resized = img.resize((224,224))
    x = np.expand_dims(np.array(img_resized), axis=0)
    x = cfg["preprocess"](x)

    preds = model.predict(x)[0]
    top3 = np.argsort(preds)[-3:][::-1]

    result_text = ""
    for i, idx in enumerate(top3):
        result_text += f"{i+1}. {CLASS_NAMES[idx]} — {preds[idx]*100:.2f}%\n"

    heatmap = gradcam(model, cfg["last_conv"], x, top3[0])
    heatmap = cv2.resize(heatmap, img.size)
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

    return result_text, Image.fromarray(overlay)

# LOAD RANDOM IMAGE
def load_random_image():
    items = []
    for class_name, folder in CLASS_FOLDERS.items():
        folder_path = os.path.join(BASE_IMAGES, folder)
        if os.path.exists(folder_path):
            for img in os.listdir(folder_path):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((class_name, os.path.join(folder_path, img)))

    true_class, img_path = random.choice(items)
    img = Image.open(img_path).convert("RGB")

    return img, f"**True Class:** {true_class}"

# GUI
with gr.Blocks(css="""
#main {max-width: 100%;}
.gradio-container {width: 100vw;}
""") as demo:

    gr.Markdown(
        "##Car Classification with Grad-CAM\n"

    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Car Image", height=420)

            true_label = gr.Markdown("**True Class:** —")

            random_btn = gr.Button(" Load Random Image (10 Classes)")
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="ResNet50",
                label="Choose Model"
            )

            predict_btn = gr.Button("Run Prediction", variant="primary")

        with gr.Column(scale=1):
            preds_out = gr.Textbox(label="Top-3 Predictions", lines=6)
            gradcam_out = gr.Image(label="Grad-CAM Visualization", height=420)

    random_btn.click(load_random_image, outputs=[image_input, true_label])
    predict_btn.click(
        predict,
        inputs=[image_input, model_choice],
        outputs=[preds_out, gradcam_out]
    )

demo.launch()


