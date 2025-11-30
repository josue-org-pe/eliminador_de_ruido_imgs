import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import random

st.set_page_config(page_title="MTC-A", layout="wide")
st.title(" SIMULADOR MTC-A ")

tabs = st.tabs([" TEXTO ", " IMÁGENES", " EXCEL"])

# =========================
# FUNCIONES HAMMING
# =========================
def hamming_encode_bits(bits):
    salida = ""
    for i in range(0, len(bits), 4):
        bloque = bits[i:i+4].ljust(4, "0")
        d1, d2, d3, d4 = map(int, bloque)
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p4 = d2 ^ d3 ^ d4
        salida += f"{p1}{p2}{d1}{p4}{d2}{d3}{d4}"
    return salida

def hamming_decode_bits(bits):
    salida = ""
    for i in range(0, len(bits), 7):
        b = list(map(int, bits[i:i+7]))
        if len(b) < 7:
            continue
        s1 = b[0] ^ b[2] ^ b[4] ^ b[6]
        s2 = b[1] ^ b[2] ^ b[5] ^ b[6]
        s4 = b[3] ^ b[4] ^ b[5] ^ b[6]
        pos = s1*1 + s2*2 + s4*4
        if pos != 0:
            b[pos-1] ^= 1
        salida += f"{b[2]}{b[4]}{b[5]}{b[6]}"
    return salida

def bytes_to_bits(data):
    return ''.join(format(b, '08b') for b in data)

def bits_to_bytes(bits):
    n = len(bits) - (len(bits) % 8)
    bits = bits[:n]
    out = bytearray()
    for i in range(0, n, 8):
        out.append(int(bits[i:i+8], 2))
    return bytes(out)

# =====================================================
# Para  TEXTO — 3 FASES
# =====================================================
with tabs[0]:
    st.header(" TEXTO")

    modo = st.selectbox("Modo:", [" Escribir", " Subir .txt"])
    texto = ""

    if modo == "✍ Escribir":
        texto = st.text_area("Escriba el texto:", height=180)
    else:
        archivo = st.file_uploader("Sube archivo TXT", type=["txt"])
        if archivo:
            texto = archivo.read().decode("utf-8")
            st.text_area("Contenido del TXT:", texto, height=180)

    prob = st.slider("Probabilidad de error por bit", 0.0, 0.3, 0.05, 0.01)

    if st.button("ENVIAR") and texto.strip() != "":

        bytes_data = texto.encode("utf-8")
        bits = bytes_to_bits(bytes_data)
        bits_hamming = hamming_encode_bits(bits)

        bits_ruido = list(bits_hamming)
        errores = 0
        for i in range(len(bits_ruido)):
            if random.random() < prob:
                bits_ruido[i] = '1' if bits_ruido[i] == '0' else '0'
                errores += 1
        bits_ruido = ''.join(bits_ruido)

        texto_corrupto = bits_to_bytes(bits_ruido).decode("utf-8", errors="replace")

        bits_recuperados = hamming_decode_bits(bits_ruido)
        texto_recuperado = bits_to_bytes(bits_recuperados).decode("utf-8", errors="ignore")

        min_len = min(len(bits), len(bits_recuperados))
        err_final = sum(1 for i in range(min_len) if bits[i] != bits_recuperados[i])
        ber = err_final / max(1, min_len)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader(" TEXTO ENVIADO")
            st.text_area("", texto, height=220)

        with c2:
            st.subheader(" TEXTO CORROMPIDO")
            st.text_area("", texto_corrupto, height=220)
            st.write("Bits alterados:", errores)

        with c3:
            st.subheader(" TEXTO RECUPERADO")
            st.text_area("", texto_recuperado, height=220)
            st.write("Errores finales:", err_final)
            st.write("BER:", round(ber, 6))

# =====================================================
# Para  IMÁGENES — RUIDO
# =====================================================
with tabs[1]:
    st.header(" IMÁGENES — Ruido y Recuperación")

    img = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg", "bmp"])

    if img:
        imagen = Image.open(img).convert("L")
        arr = np.array(imagen)

        c1, c2, c3 = st.columns(3)
        c1.image(imagen, caption="Original")

        if st.button(" AGREGAR RUIDO"):
            ruido = np.random.normal(0, 25, arr.shape)
            ruidosa = np.clip(arr + ruido, 0, 255).astype(np.uint8)
            st.session_state["ruidosa"] = ruidosa

        if "ruidosa" in st.session_state:
            c2.image(st.session_state["ruidosa"], caption="Con Ruido")

        if st.button(" QUITAR RUIDO"):
            if "ruidosa" in st.session_state:
                limpia = cv2.fastNlMeansDenoising(st.session_state["ruidosa"], None, 10, 7, 21)
                st.session_state["limpia"] = limpia

        if "limpia" in st.session_state:
            c3.image(st.session_state["limpia"], caption="Recuperada")


 # Para  EXCEL — DAÑO Y RECUPERACIÓN
with tabs[2]:
    st.header(" EXCEL — Reconstrucción")

    excel = st.file_uploader("Sube archivo Excel", type=["xlsx"])

    if excel:
        df = pd.read_excel(excel)
        st.subheader(" Original")
        st.dataframe(df)

        if st.button(" SIMULAR DAÑO"):
            df_ruido = df.copy()
            for col in df_ruido.columns:
                if df_ruido[col].dtype != object:
                    df_ruido[col] *= np.random.uniform(0.9, 1.1)
            st.session_state["dañado"] = df_ruido

        if "dañado" in st.session_state:
            st.subheader(" Dañado")
            st.dataframe(st.session_state["dañado"])

        if st.button(" RECONSTRUIR"):
            if "dañado" in st.session_state:
                df_rec = st.session_state["dañado"].round(2)
                df_rec.to_excel("excel_recuperado.xlsx", index=False)
                with open("excel_recuperado.xlsx", "rb") as f:
                    st.download_button("⬇ Descargar Excel Recuperado", f, file_name="excel_recuperado.xlsx")
                st.subheader(" Recuperado")
                st.dataframe(df_rec)
