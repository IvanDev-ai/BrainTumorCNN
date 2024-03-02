import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import gdown

# Cargar el modelo preentrenado
# Descargar el modelo desde Google Drive
url = 'https://drive.google.com/uc?id=1U05CZh3092LBYCgqhMaYNdWjEx02dWXz'
output_model = 'modelo_entrenado.h5'
gdown.download(url, output_model, quiet=False)
model = load_model(output_model)

# Diccionario de etiquetas
labels = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}

# Función para predecir la clase de la imagen
def predict(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    prediction = np.argmax(result)
    return labels[prediction]

# Interfaz de usuario con Streamlit
st.title('Brain Tumor Classifier')
st.write('Carga una imagen para predecir si hay un tumor cerebral o no.')

# Widget para cargar la imagen
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Imagen cargada', width=300)
    st.write("")
    st.write("Clasificación:")

    # Predecir la clase de la imagen cargada
    prediction = predict(uploaded_file)
    if prediction == 'no_tumor':
        st.write("No hay ningún tumor en la imagen.")
    else:
        st.write("Tumor cerebral detectado:", prediction)

