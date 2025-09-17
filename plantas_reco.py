
# -----------------------------
# Importamos librerías necesarias:
# - streamlit: para crear la interfaz web
# - PIL (Python Imaging Library): para abrir y manipular imágenes
# - numpy: para manejar arrays y cálculos numéricos
# - tensorflow.keras: para construir y usar el modelo de IA (MobileNetV2, capas y modelo)
# -----------------------------
import streamlit as st
from PIL import Image
# Importamos el numpy para trabajar con las imagenes y las predicciones.
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

# -----------------------------
# Estilos generales
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #F0FFF0; }
.titulo { text-align: center; color: black; font-size: 36px; font-weight: bold; margin-bottom: 0; }
.subtitulo { text-align: center; color: #4CAF50; font-size: 18px; margin-top: 0; margin-bottom: 30px; }
div.stButton > button:first-child {
    background-color: #4CAF50; color: white; height: 50px; width: 200px; border-radius: 10px;
    border: none; font-size: 18px; font-weight: bold; cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Título
# -----------------------------
st.markdown("<h1 style='text-align: center; color: black; font-size: 36px; font-weight: bold; margin-bottom: 0;'>Detector de salud en plantas</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitulo'>Sube una foto de la hoja de tu planta y obtén un diagnóstico inmediato.</p>", unsafe_allow_html=True)

# -----------------------------
# Definimos el modelo MobileNetV2 preentrenado
# -----------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
x = GlobalAveragePooling2D()(base_model.output)
# Crea una capa de 3 neuronas para cada clase de sana, enferma y seca
output = Dense(3, activation='softmax')(x)  # 3 clases: sana, enferma, seca
# Combina todo en un modelo completo: recibe una imagen de 224x224 píxeles y produce una salida de 3 neuronas
model = Model(inputs=base_model.input, outputs=output)

# Usamos el modelo para inferencia, no entrenamiento.
model.trainable = False

# -----------------------------
# Widget para subir la imagen desde la app
# -----------------------------
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])

# Si el usuario sube un archivo, lo abrimos y lo mostramos
if uploaded_file is not None:
    # Abrimos la imagen con PIL y la convertimos a RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Mostramos la imagen subida en la app con el ancho del contenedor
    st.image(image, caption="Imagen subida", use_container_width=True)

    if st.button("Analizar"):
        # Redimensionamos la imagen a 224x224 píxeles
        img_array = np.array(image.resize((224, 224))) / 255.0

        # Agregamos una dimensión extra para indicar que es un batch de 1 imagen
        img_array = np.expand_dims(img_array, axis=0)

        # Realizamos la predicción usando el modelo cargado
        pred = model.predict(img_array)

        # Obtenemos la clase con mayor probabilidad
        clase = np.argmax(pred)

        # Calculamos el porcentaje de certeza de la clase predicha
        porcentaje = pred[0][clase] * 100

        # Definimos colores para cada clase: verde=sana, rojo=enferma, naranja=seca
        colores = ["green", "red", "orange"]

        # Nombres de las clases correspondientes
        nombres = ["sana", "enferma", "seca"]

        # Mostramos el resultado en Streamlit con formato HTML y color según la clase
        st.markdown(
            f"<p style='color:{colores[clase]}; font-size:18px;'>"
            f"La planta está <b>{nombres[clase]}</b> con {porcentaje:.2f}% de certeza</p>",
            unsafe_allow_html=True
        )

