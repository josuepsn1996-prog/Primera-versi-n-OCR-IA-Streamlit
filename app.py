import streamlit as st
import openai
import base64
import pandas as pd
from io import StringIO
import tempfile
from pdf2image import convert_from_path
import os
import re

st.set_page_config(page_title="OCR Inteligente por Página", page_icon="📄")

st.title("📄 OCR Inteligente con GPT-4o - Página por Página")
st.markdown("""
Sube tu PDF escaneado con tablas de **residuos autorizados**, y GPT-4o extraerá la tabla con mayor precisión, procesando **una página a la vez desde la imagen**.

*Necesitas una clave API de [OpenAI](https://platform.openai.com/api-keys). Nunca la compartas.*
""")

api_key = st.text_input("Introduce tu clave OpenAI API", type="password")
uploaded_file = st.file_uploader("Sube tu PDF escaneado", type=["pdf"])

if uploaded_file and api_key:
    openai.api_key = api_key

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Convirtiendo PDF a imágenes..."):
        images = convert_from_path(tmp_path, dpi=300)

    st.info(f"El PDF tiene {len(images)} páginas. Procesando por separado...")

    csv_rows = []
    header = None

    for i, page in enumerate(images):
        st.write(f"Procesando página {i + 1}...")

        img_path = tmp_path + f"_page_{i + 1}.jpg"
        page.save(img_path, 'JPEG')
        with open(img_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        messages = [
            {"role": "system", "content": "Eres un experto en interpretación de OCR y textos escaneados."},
            {"role": "user", "content": [
                {"type": "text", "text": "Lee la imagen, extrae el texto limpio. Si hay tablas de residuos autorizados, extrae en CSV con columnas: clave, residuo, cantidad generada (kg/día), destino."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]}
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2048,
        )

        content = response.choices[0].message.content

        if "```csv" in content:
            csv_part = content.split("```csv")[1].split("```", 1)[0].strip()
        else:
            csv_part = content.strip()

        # Limpia separadores de miles
        def clean_commas_in_numbers(row):
            if '"' in row:
                parts = row.split('"')
                for i in range(0, len(parts), 2):
                    parts[i] = re.sub(r'(\d+),(\d{3})', r'\1\2', parts[i])
                return '"'.join(parts)
            else:
                return re.sub(r'(\d+),(\d{3})', r'\1\2', row)

        lines = [clean_commas_in_numbers(l) for l in csv_part.splitlines() if l.strip()]

        if not header:
            header = lines[0]
        csv_rows.extend(lines[1:])

    if header and csv_rows:
        full_csv = header + "\n" + "\n".join(csv_rows)
        try:
            df = pd.read_csv(StringIO(full_csv))
            st.success("¡Tabla final extraída con alta precisión!")
            st.dataframe(df)

            st.download_button(
                "Descargar tabla unificada (CSV)",
                data=df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="residuos_autorizados_preciso.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error al convertir CSV a tabla: {e}")
            st.text(full_csv)
    else:
        st.warning("No se detectaron tablas en el documento.")

else:
    st.info("Sube un PDF escaneado y tu clave de API para comenzar.")
