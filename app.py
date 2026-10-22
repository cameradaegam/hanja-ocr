import streamlit as st
import easyocr
import hanja
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(page_title="한자 OCR", layout="centered")

st.title("📱 한자 OCR & 한글음 변환기")

# EasyOCR 모델 로딩
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_tra'], gpu=False)

try:
    reader = load_ocr()
except Exception as e:
    st.error(f"OCR 로딩 오류: {e}")

uploaded_file = st.file_uploader("한자 이미지를 선택하거나 직접 촬영하세요", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_image)

        with st.spinner("한자를 인식하고 있습니다..."):
            results = reader.readtext(img_np)

        pure_hanja_list = []
        pure_korean_list = []

        draw = ImageDraw.Draw(pil_image)

        if results:
            for bbox, text, prob in results:
                pt1 = (int(bbox[0][0]), int(bbox[0][1]))
                pt2 = (int(bbox[2][0]), int(bbox[2][1]))

                kor_text = hanja.translate(text, 'substitution')
                pure_hanja_list.append(text)
                pure_korean_list.append(kor_text)

                draw.rectangle([pt1, pt2], outline=(0, 255, 0), width=3)

        st.image(pil_image, caption="인식 위치 확인", use_container_width=True)

        hanja_str = " ".join(pure_hanja_list) if pure_hanja_list else "인식된 한자가 없습니다."
        kor_str = " ".join(pure_korean_list) if pure_korean_list else "-"

        st.subheader("=== 원본 한자 ===")
        st.code(hanja_str, language="text")

        st.subheader("=== 한글변환 ===")
        st.code(kor_str, language="text")

    except Exception as e:
        st.error(f"처리 중 오류 발생: {e}")

# --- 하단 서명 ---
st.markdown("---")
st.markdown("<h4 style='text-align: center; color: #888888;'>- 카메라대감 -</h4>", unsafe_allow_html=True)
