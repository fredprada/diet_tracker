import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageClassification

# Configuração do título
st.title("DeepSeek-R1 Image Classifier")
st.write("Faça upload de uma imagem para que o modelo DeepSeek-R1 identifique o que está nela.")

# Carregar modelo localmente
@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    return model, processor

model, processor = load_model()

# Upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Exibir a imagem no Streamlit
    st.image(image, caption="Imagem Carregada", use_column_width=True)

    # Pré-processar imagem
    inputs = processor(images=image, return_tensors="pt")

    # Fazer inferência
    with torch.no_grad():
        outputs = model(**inputs)

    # Obter os resultados
    scores = outputs.logits.softmax(dim=-1)
    labels = model.config.id2label

    # Mostrar os resultados
    top_5 = torch.topk(scores, 5)
    for i in range(5):
        label_id = top_5.indices[0][i].item()
        st.write(f"🔹 **{labels[label_id]}** - {top_5.values[0][i].item():.4f}")
