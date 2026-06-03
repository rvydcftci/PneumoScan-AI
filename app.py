import os
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


st.set_page_config(
    page_title="PneumoScan AI",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: #f4f9fc;
    color: #123047;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #072b43 0%, #0d4868 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #0b3551 0%, #256f91 55%, #b8e5f4 100%);
    padding: 42px;
    border-radius: 30px;
    margin-bottom: 28px;
    box-shadow: 0 18px 45px rgba(11, 53, 81, 0.22);
}

.hero h1 {
    color: white;
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero h3 {
    color: #eaf7ff;
    font-size: 22px;
    font-weight: 500;
}

.hero p {
    color: #f0fbff;
    font-size: 16px;
    max-width: 850px;
}

.card {
    background: white;
    border: 1px solid #d8edf6;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 12px 28px rgba(13, 71, 105, 0.08);
    margin-bottom: 18px;
}

.result-risk {
    background: linear-gradient(135deg, #d12b2b, #8b1d1d);
    color: white;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-size: 25px;
    font-weight: 800;
}

.result-normal {
    background: linear-gradient(135deg, #12a87b, #087356);
    color: white;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-size: 25px;
    font-weight: 800;
}

.metric-card {
    background: #f8fcff;
    border: 1px solid #d8edf6;
    border-radius: 18px;
    padding: 18px;
    margin-top: 14px;
}

.metric-label {
    color: #557086;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #0b3551;
    font-size: 30px;
    font-weight: 800;
}

.info-box {
    background: #edf8fd;
    border-left: 5px solid #2c7ea3;
    padding: 16px;
    border-radius: 14px;
    color: #123c5a;
    margin-top: 14px;
}

.warn-box {
    background: #fff6e8;
    border-left: 5px solid #f59e0b;
    padding: 16px;
    border-radius: 14px;
    color: #7c4a00;
    margin-top: 14px;
}

.error-box {
    background: #fff1f2;
    border-left: 5px solid #e11d48;
    padding: 18px;
    border-radius: 14px;
    color: #881337;
    margin-top: 18px;
}

.feature-card {
    background: white;
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #d8edf6;
    text-align: center;
    min-height: 145px;
    box-shadow: 0 12px 28px rgba(13, 71, 105, 0.08);
}

.feature-card h3 {
    color: #123c5a;
    font-size: 21px;
}

.feature-card p {
    color: #5b7180;
    font-size: 14px;
}

.prob-row {
    margin-top: 14px;
    margin-bottom: 10px;
}

.prob-label {
    display: flex;
    justify-content: space-between;
    color: #123c5a;
    font-weight: 700;
    margin-bottom: 6px;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #77bdd8;
    padding: 18px;
    border-radius: 20px;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #123c5a, #2c7ea3);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1.2rem;
    font-weight: 700;
}

h1, h2, h3 {
    color: #123c5a;
}

hr {
    border-color: #d8edf6;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ai_model():
    model_paths = [
        "pnomoni_modeli.h5",
        "pneumonia_model.h5",
        "pneumonia_cnn_model.h5",
        "best_model.h5",
        "model.h5",
    ]

    selected_path = None

    for path in model_paths:
        if os.path.exists(path):
            selected_path = path
            break

    if selected_path is None:
        st.error(
            "Model dosyası bulunamadı. app.py ile aynı klasöre "
            "pnomoni_modeli.h5 veya model.h5 dosyanı koymalısın."
        )
        st.stop()

    model = load_model(selected_path)
    model(np.zeros((1, 224, 224, 3), dtype=np.float32))

    return model


def prepare_image_for_model(img):
    resized = img.resize((224, 224))
    arr = image.img_to_array(resized)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)

    return arr


def image_processing_steps(img):
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(gray)

    blurred = cv2.GaussianBlur(
        contrast,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    return gray, contrast, edges


def is_probably_chest_xray(img):
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    aspect_ratio = width / height

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    mean_saturation = np.mean(saturation)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    center_crop = gray[
        int(height * 0.12):int(height * 0.92),
        int(width * 0.10):int(width * 0.90)
    ]

    dark_ratio = np.sum(center_crop < 80) / center_crop.size
    bright_ratio = np.sum(center_crop > 180) / center_crop.size

    left_half = gray[:, :width // 2]
    right_half = gray[:, width // 2:]
    symmetry_diff = abs(np.mean(left_half) - np.mean(right_half))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    upper_half = gray[:height // 2, :]
    lower_half = gray[height // 2:, :]

    upper_std = np.std(upper_half)
    lower_std = np.std(lower_half)

    center_vertical = gray[:, int(width * 0.35):int(width * 0.65)]
    side_regions = np.concatenate([
        gray[:, :int(width * 0.25)].flatten(),
        gray[:, int(width * 0.75):].flatten()
    ])

    center_mean = np.mean(center_vertical)
    side_mean = np.mean(side_regions)

    lung_like_dark_sides = side_mean < center_mean + 15
    chest_texture_condition = upper_std > 25
    not_long_bone_condition = not (
        aspect_ratio < 0.85 and lower_std > upper_std
    )

    conditions = [
        0.60 <= aspect_ratio <= 1.60,
        mean_saturation < 18,
        45 <= mean_brightness <= 185,
        std_brightness > 30,
        dark_ratio > 0.18,
        bright_ratio > 0.015,
        symmetry_diff < 35,
        0.015 <= edge_density <= 0.18,
        chest_texture_condition,
        lung_like_dark_sides,
        not_long_bone_condition
    ]

    score = sum(conditions)

    return score >= 8, {
        "score": score,
        "aspect_ratio": round(float(aspect_ratio), 2),
        "mean_saturation": round(float(mean_saturation), 2),
        "mean_brightness": round(float(mean_brightness), 2),
        "std_brightness": round(float(std_brightness), 2),
        "dark_ratio": round(float(dark_ratio), 3),
        "bright_ratio": round(float(bright_ratio), 3),
        "symmetry_diff": round(float(symmetry_diff), 2),
        "edge_density": round(float(edge_density), 3),
        "upper_std": round(float(upper_std), 2),
        "lower_std": round(float(lower_std), 2),
        "center_mean": round(float(center_mean), 2),
        "side_mean": round(float(side_mean), 2)
    }

def create_saliency_heatmap(img, model):
    img_array = prepare_image_for_model(img)
    input_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor)
        loss = prediction[:, 0]

    grads = tape.gradient(loss, input_tensor)

    if grads is None:
        return None

    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    saliency = saliency / (np.max(saliency) + 1e-8)

    saliency = cv2.resize(saliency, img.size)
    heatmap = np.uint8(255 * saliency)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original = np.array(img)

    overlay = cv2.addWeighted(
        original,
        0.65,
        heatmap_color,
        0.35,
        0
    )

    return overlay


def probability_chart(normal_prob, pneumonia_prob):
    fig, ax = plt.subplots(figsize=(4.4, 2.7), dpi=120)

    labels = ["NORMAL", "PNEUMONIA"]
    values = [normal_prob, pneumonia_prob]

    bars = ax.bar(labels, values)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Olasılık (%)", fontsize=9)
    ax.set_title("Model Sınıf Olasılıkları", fontsize=11)
    ax.tick_params(labelsize=8)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 2, 96),
            f"%{value:.1f}",
            ha="center",
            fontsize=8
        )

    fig.tight_layout()

    return fig


def histogram_chart(gray):
    fig, ax = plt.subplots(figsize=(4.6, 2.7), dpi=120)

    ax.hist(gray.ravel(), bins=90)

    ax.set_title("Gri Seviye Histogramı", fontsize=11)
    ax.set_xlabel("Piksel Yoğunluğu", fontsize=9)
    ax.set_ylabel("Frekans", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.tight_layout()

    return fig


def create_pdf_report(result, confidence, risk, normal_prob, pneumonia_prob):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, height - 60, "PneumoScan AI - Analiz Raporu")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, height - 100, f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    pdf.drawString(50, height - 130, f"Tahmin Sonucu: {result}")
    pdf.drawString(50, height - 160, f"Guven Orani: %{confidence * 100:.2f}")
    pdf.drawString(50, height - 190, f"Risk Seviyesi: {risk}")
    pdf.drawString(50, height - 220, f"NORMAL Olasiligi: %{normal_prob:.2f}")
    pdf.drawString(50, height - 250, f"PNEUMONIA Olasiligi: %{pneumonia_prob:.2f}")

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, height - 300, "Analiz Ozeti")

    pdf.setFont("Helvetica", 10)

    lines = [
        "Yuklenen akciger radyografisi yapay zeka modeli tarafindan analiz edilmistir.",
        "Projede gri tonlama, CLAHE kontrast iyilestirme, Canny kenar cikarimi,",
        "histogram analizi ve model odak haritasi kullanilmistir.",
        "Bu rapor akademik proje amaciyla olusturulmustur.",
        "Kesin tibbi tani icin uzman doktor degerlendirmesi gerekir."
    ]

    y = height - 325

    for line in lines:
        pdf.drawString(50, y, line)
        y -= 20

    pdf.save()
    buffer.seek(0)

    return buffer


def report_text(result, confidence, risk, normal_prob, pneumonia_prob):
    return f"""PneumoScan AI - Akciğer Radyografisi Analiz Raporu

Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}

Tahmin Sonucu: {result}
Güven Oranı: %{confidence * 100:.2f}
Risk Seviyesi: {risk}

NORMAL Olasılığı: %{normal_prob:.2f}
PNEUMONIA Olasılığı: %{pneumonia_prob:.2f}

Kullanılan Görüntü İşleme Yöntemleri:
- Gri tonlama
- CLAHE kontrast iyileştirme
- Gaussian Blur
- Canny kenar çıkarımı
- Histogram analizi
- Model odak haritası

Model Test Performansı:
- Accuracy: %82.7
- NORMAL Recall: %69
- PNEUMONIA Recall: %91
- Weighted F1 Score: %82

Not:
Bu sistem akademik proje amacıyla geliştirilmiştir.
Tıbbi tanı yerine geçmez.
"""
model = load_ai_model()

if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.markdown("## PneumoScan AI")
    st.markdown("Akciğer radyografilerinden pnömoni tespiti")

    st.markdown("---")
    st.markdown("### Proje Bilgileri")
    st.write("Model: CNN")
    st.write("Test Accuracy: %82.7")
    st.write("Sınıflar: NORMAL / PNEUMONIA")
    st.write("Görüntü İşleme: OpenCV")
    st.write("Arayüz: Streamlit")

    st.markdown("---")
    st.warning("Bu sistem eğitim amaçlıdır. Tıbbi tanı yerine geçmez.")

    st.markdown("---")
    st.markdown("### Geçmiş Analizler")

    if len(st.session_state.history) == 0:
        st.write("Henüz analiz yok.")
    else:
        for item in reversed(st.session_state.history[-5:]):
            st.write(f"{item['time']} - {item['result']} - %{item['confidence']:.2f}")


st.markdown("""
<div class="hero">
    <h1>PneumoScan AI</h1>
    <h3>Akciğer Radyografilerinden Pnömoni Tespiti</h3>
    <p>
        Yüklenen akciğer röntgen görüntüsü; yapay zeka, görüntü işleme ve model odak haritası
        kullanılarak analiz edilir. Sistem, pnömoni ihtimalini ve güven oranını kullanıcıya sunar.
    </p>
</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Röntgen görüntüsü yükleyin",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is None:
    st.markdown("""
    <div class="card">
        <h2>Analize başlamak için bir akciğer röntgen görüntüsü yükleyin.</h2>
        <p>
            Sistem önce yüklenen görüntünün akciğer radyografisine benzerliğini kontrol eder.
            Uygun görüntülerde pnömoni tahmini, görüntü işleme çıktıları, model odak haritası
            ve indirilebilir rapor oluşturulur.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3>Yapay Zeka Analizi</h3>
            <p>CNN modeli ile NORMAL ve PNEUMONIA sınıflandırması yapılır.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3>Görüntü İşleme</h3>
            <p>Kontrast iyileştirme, kenar çıkarımı ve histogram analizi uygulanır.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <h3>Görüntü Doğrulama</h3>
            <p>Sistem önce görüntünün akciğer röntgeni olup olmadığını kontrol eder.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    img = Image.open(uploaded_file).convert("RGB")

    is_xray, validation_info = is_probably_chest_xray(img)

    if not is_xray:
        st.markdown("""
        <div class="error-box">
            <b>Geçersiz görüntü uyarısı:</b><br>
            Yüklenen görsel akciğer radyografisine benzemiyor. Lütfen geçerli bir akciğer röntgen görüntüsü yükleyiniz.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Teknik kontrol detayları"):
            st.write(validation_info)

        st.image(img, caption="Yüklenen görüntü", use_container_width=True)

    else:
        model_input = prepare_image_for_model(img)
        prediction = float(model.predict(model_input)[0][0])

        pneumonia_prob = prediction * 100
        normal_prob = (1 - prediction) * 100

        if prediction > 0.5:
            result = "PNEUMONIA"
            confidence = prediction
            risk = "Yüksek Risk"
        else:
            result = "NORMAL"
            confidence = 1 - prediction
            risk = "Düşük Risk"

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M"),
            "result": result,
            "confidence": confidence * 100
        })

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.subheader("Yüklenen Röntgen Görüntüsü")
            st.image(img, use_container_width=True)

        with col2:
            st.subheader("Analiz Sonucu")

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tanı Sonucu</div>
                <div class="metric-value">{result}</div>
            </div>
            """, unsafe_allow_html=True)

            if result == "PNEUMONIA":
                st.markdown('<div class="result-risk">PNEUMONIA</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-normal">NORMAL</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Güven Oranı</div>
                <div class="metric-value">%{confidence * 100:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Seviyesi</div>
                <div class="metric-value">{risk}</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(confidence))

            st.markdown(f"""
            <div class="info-box">
                <b>Model Çıktısı:</b> {prediction:.4f}<br>
                <b>NORMAL Olasılığı:</b> %{normal_prob:.2f}<br>
                <b>PNEUMONIA Olasılığı:</b> %{pneumonia_prob:.2f}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="warn-box">
                Bu analiz akademik proje amacıyla geliştirilmiştir. Klinik karar için uzman doktor değerlendirmesi gerekir.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("Olasılık Dağılımı")

        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">
                <span>NORMAL</span>
                <span>%{normal_prob:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(normal_prob / 100)

        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">
                <span>PNEUMONIA</span>
                <span>%{pneumonia_prob:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(pneumonia_prob / 100)

        st.markdown("---")

        gray, contrast, edges = image_processing_steps(img)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Sınıf Olasılık Grafiği")
            fig_prob = probability_chart(normal_prob, pneumonia_prob)
            st.pyplot(fig_prob, use_container_width=False)

        with chart_col2:
            st.subheader("Histogram Analizi")
            fig_hist = histogram_chart(gray)
            st.pyplot(fig_hist, use_container_width=False)

        st.markdown("---")

        st.subheader("Görüntü İşleme Aşamaları")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.write("Gri Tonlama")
            st.image(gray, clamp=True, use_container_width=True)

        with p2:
            st.write("CLAHE Kontrast İyileştirme")
            st.image(contrast, clamp=True, use_container_width=True)

        with p3:
            st.write("Canny Kenar Çıkarımı")
            st.image(edges, clamp=True, use_container_width=True)

        st.markdown("---")

        st.subheader("Model Odak Haritası")

        heatmap_img = create_saliency_heatmap(img, model)

        h1, h2 = st.columns(2)

        with h1:
            st.write("Orijinal Görüntü")
            st.image(img, use_container_width=True)

        with h2:
            st.write("Modelin Odaklandığı Alanlar")
            if heatmap_img is not None:
                st.image(heatmap_img, use_container_width=True)
            else:
                st.warning("Model odak haritası oluşturulamadı.")

        st.markdown("---")

        st.subheader("Model Test Performansı")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Accuracy", "%82.7")
        with m2:
            st.metric("NORMAL Recall", "%69")
        with m3:
            st.metric("PNEUMONIA Recall", "%91")
        with m4:
            st.metric("Weighted F1", "%82")

        st.markdown("""
        <div class="info-box">
            Test sonuçları 624 görüntülük test veri seti üzerinde hesaplanmıştır.
            Confusion Matrix: [[161, 73], [35, 355]]
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("Otomatik Analiz Raporu")

        txt_report = report_text(
            result,
            confidence,
            risk,
            normal_prob,
            pneumonia_prob
        )

        pdf_report = create_pdf_report(
            result,
            confidence,
            risk,
            normal_prob,
            pneumonia_prob
        )

        st.text_area("Rapor İçeriği", txt_report, height=260)

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                label="TXT Raporu İndir",
                data=txt_report,
                file_name="pneumoscan_analiz_raporu.txt",
                mime="text/plain"
            )

        with d2:
            st.download_button(
                label="PDF Raporu İndir",
                data=pdf_report,
                file_name="pneumoscan_analiz_raporu.pdf",
                mime="application/pdf"
            )