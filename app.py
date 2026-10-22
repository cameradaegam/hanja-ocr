import os
import torch

# PyTorch 메모리 및 스레드 제한 (Streamlit RAM 초과 방지)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import streamlit as st
import easyocr
import hanja
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

st.set_page_config(page_title="한자 & 한글 OCR", layout="centered")

st.title("📱 한자·한글 OCR & 독음 변환")

# 고용량 이미지 메모리 방지용 자동 축소 함수
def resize_image(image, max_size=1024):
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

# EasyOCR 모델 로딩 (['ch_tra', 'en'])
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_tra', 'en'], gpu=False, quantize=True)

try:
    reader = load_ocr()
except Exception as e:
    st.error(f"OCR 로딩 중 오류가 발생했습니다: {e}")

uploaded_file = st.file_uploader("이미지를 선택하거나 직접 촬영하세요 (한글/한자 혼용 가능)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        pil_image = resize_image(pil_image, max_size=1024)

        # --- OCR 인식률 향상을 위한 이미지 전처리 (흑백 변환 및 명암 대비 강화) ---
        gray_image = pil_image.convert("L")
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(2.0)  # 대비 2배 강화
        img_np = np.array(enhanced_image.convert("RGB"))

        with st.spinner("한글 및 한자를 인식하고 있습니다..."):
            # paragraph=True로 설정하여 글자가 쪼개져 인식되는 오인식 방지
            results = reader.readtext(img_np, paragraph=True)

        pure_original_list = []
        pure_korean_list = []

        draw = ImageDraw.Draw(pil_image)

        if results:
            for item in results:
                bbox, text = item[0], item[1]

                pt1 = (int(bbox[0][0]), int(bbox[0][1]))
                pt2 = (int(bbox[2][0]), int(bbox[2][1]))

                # 한자는 한글 독음으로 변환
                converted_text = hanja.translate(text, 'substitution')
                
                pure_original_list.append(text)
                pure_korean_list.append(converted_text)

                draw.rectangle([pt1, pt2], outline=(0, 255, 0), width=3)

        st.image(pil_image, caption="인식 위치 확인", use_container_width=True)

        original_str = " ".join(pure_original_list) if pure_original_list else "인식된 글자가 없습니다."
        korean_str = " ".join(pure_korean_list) if pure_korean_list else "-"

        st.subheader("=== 인식된 원본 ===")
        st.code(original_str, language="text")

        st.subheader("=== 한글 변환 결과 ===")
        st.code(korean_str, language="text")

    except Exception as e:
        st.error(f"이미지 처리 중 오류가 발생했습니다: {e}")

# 하단 서명
st.markdown("---")
st.markdown("<h3 style='text-align: center; color: #888888;'>- 카메라대감 -</h3>", unsafe_allow_html=True)
