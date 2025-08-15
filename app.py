import streamlit as st
import openai
import base64
import pandas as pd
from io import StringIO
import tempfile
from pdf2image import convert_from_path
import os
import re

st.set_page_config(page_title="OCR Inteligente para Residuos", page_icon="♻️")

st.title("♻️ OCR Inteligente con GPT-4o (Residuos Sólidos Urbanos)")
st.markdown("""
Carga tu PDF escaneado y obtén una tabla limpia con los **residuos autorizados** para su manejo.

*Necesitas tu clave de API de [OpenAI](https://platform.openai.com/api-keys) (nunca la compartas, solo cópiala aquí temporalmente para usar la app).*
""")

api_key = st.text_input("Introduce tu clave OpenAI API", type="password")
uploaded_file = st.file_uploader("Sube tu PDF escaneado", type=["pdf"])

if uploaded_file and api_key:
    openai.api_key = api_key

    # Guarda PDF temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Convierte PDF a imágenes
    with st.spinner("Convirtiendo PDF a imágenes..."):
        images = convert_from_path(tmp_path, dpi=300)
    all_texts = []

    st.info(f"El PDF tiene {len(images)} páginas. Procesando...")

    # Procesa cada página con GPT-4o Vision
    for i, page in enumerate(images):
        st.write(f"Procesando página {i+1} de {len(images)}...")
        img_temp_path = tmp_path + f"_page_{i+1}.jpg"
        page.save(img_temp_path, 'JPEG')
        with open(img_temp_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        messages = [
            {"role": "system", "content": "Eres un experto en interpretación de OCR y textos escaneados."},
            {"role": "user", "content": [
                {"type": "text", "text": "Lee la imagen adjunta, extrae el texto limpio. Si hay tablas de residuos autorizados, extrae en CSV."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]}
        ]
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2048,
        )
        text = response.choices[0].message.content
        all_texts.append(text)

    st.success("¡OCR y limpieza completados!")
    st.markdown("### Extrayendo la tabla final con IA...")

    # Unir texto, extraer tabla como antes
    full_text = "\n\n".join(all_texts)
    prompt_tabla = (
        "A partir del siguiente texto, crea una tabla estructurada únicamente con los residuos autorizados para su manejo. "
        "Columnas: clave, residuo, cantidad generada (kg/día), destino. Devuelve la tabla en formato CSV. Si hay varias, unifícalas."
        "\n\n" + full_text
    )

    response_tabla = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un experto en estructuración de residuos sólidos."},
            {"role": "user", "content": prompt_tabla}
        ],
        max_tokens=2048,
    )

    csv_text = response_tabla.choices[0].message.content

    # Limpieza rápida para pandas
    if "```csv" in csv_text:
        csv_text = csv_text.split("```csv")[1]
    if "```" in csv_text:
        csv_text = csv_text.split("```")[0]
    csv_text = csv_text.strip()

    # Solo líneas relevantes
    lines = csv_text.splitlines()
    data_lines = [l for l in lines if l.strip().lower().startswith("clave") or l.strip().startswith('"') or re.match(r'^[A-Z]', l)]
    csv_ready = "\n".join(data_lines)

    # Limpia comas de miles
    def clean_commas_in_numbers(row):
        if '"' in row:
            parts = row.split('"')
            for i in range(1, len(parts), 2):  # odd indexes are inside quotes, leave as is
                pass
            for i in range(0, len(parts), 2):  # even indexes are outside quotes, clean numbers
                parts[i] = re.sub(r'(\d+),(\d{3})', r'\1\2', parts[i])
            return '"'.join(parts)
        else:
            return re.sub(r'(\d+),(\d{3})', r'\1\2', row)

    csv_ready = "\n".join([clean_commas_in_numbers(row) for row in csv_ready.splitlines()])

    # Lee como DataFrame
    try:
        df = pd.read_csv(StringIO(csv_ready))
        st.success("¡Tabla extraída exitosamente!")
        st.dataframe(df)

        # Descarga como CSV
        st.download_button(
            "Descargar tabla CSV",
            data=df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="residuos_autorizados.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"No se pudo leer el CSV: {e}")
        st.text(csv_ready)
else:
    st.info("Sube un PDF y tu clave de OpenAI para comenzar.")
