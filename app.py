import streamlit as st
import openai
import base64
import pandas as pd
from io import StringIO
import tempfile
import os
import re
import fitz  # PyMuPDF

st.set_page_config(page_title="OCR Inteligente para Residuos", page_icon="♻️")

st.title("♻️ OCR Inteligente con GPT-4o (Residuos Sólidos Urbanos)")
st.markdown("""
Carga tu PDF escaneado y obtén una tabla limpia con los **residuos autorizados** para su manejo.

*Necesitas tu clave de API de [OpenAI](https://platform.openai.com/api-keys) (nunca la compartas públicamente).*
""")

api_key = st.text_input("Introduce tu clave OpenAI API", type="password")
uploaded_file = st.file_uploader("Sube tu PDF escaneado", type=["pdf"])

if uploaded_file and api_key:
    openai.api_key = api_key

    # Guarda PDF temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Convierte PDF a imágenes con fitz (PyMuPDF)
    with st.spinner("Convirtiendo PDF a imágenes..."):
        doc = fitz.open(tmp_path)
        image_paths = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img_path = os.path.join(tempfile.gettempdir(), f"page_{i+1}.png")
            pix.save(img_path)
            image_paths.append(img_path)

    all_texts = []

    st.info(f"El PDF tiene {len(image_paths)} páginas. Procesando...")

    # Procesa cada imagen con GPT-4o
    for i, img_path in enumerate(image_paths):
        st.write(f"Procesando página {i+1} de {len(image_paths)}...")

        with open(img_path, "rb") as img_file:
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
        all_texts.append(response.choices[0].message.content)

    st.success("¡OCR y limpieza completados!")
    st.markdown("### Extrayendo la tabla final con IA...")

    # Extraer tabla desde texto unificado
    full_text = "\n\n".join(all_texts)
    prompt_tabla = (
        "A partir del siguiente texto, crea una tabla estructurada únicamente con los residuos autorizados para su manejo. "
        "Columnas: clave, residuo, cantidad generada (kg/día), destino. Devuelve la tabla en formato CSV. "
        "Si hay varias, unifícalas en una sola tabla.\n\n" + full_text
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

    # Limpieza rápida
    if "```csv" in csv_text:
        csv_text = csv_text.split("```csv")[1]
    if "```" in csv_text:
        csv_text = csv_text.split("```")[0]
    csv_text = csv_text.strip()

    lines = csv_text.splitlines()
    data_lines = [l for l in lines if l.strip().lower().startswith("clave") or l.strip().startswith('"') or re.match(r'^[A-Z]', l)]
    csv_ready = "\n".join(data_lines)

    def clean_commas_in_numbers(row):
        if '"' in row:
            parts = row.split('"')
            for i in range(0, len(parts), 2):
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