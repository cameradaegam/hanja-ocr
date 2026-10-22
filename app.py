import os
import re
import torch

# PyTorch 메모리 및 스레드 제한 (Streamlit RAM 초과 방지)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import streamlit as st
import easyocr
import hanja
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(page_title="한자 & 한글 OCR", layout="centered")

st.title("📱 한자·한글 OCR & 독음 변환기")

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

# 2개의 모델을 각각 로딩 (충돌 방지 및 인식률 극대화)
@st.cache_resource
def load_ocr_kor():
    return easyocr.Reader(['ko', 'en'], gpu=False, quantize=True)

@st.cache_resource
def load_ocr_hanja():
    return easyocr.Reader(['ch_tra'], gpu=False, quantize=True)

try:
    reader_kor = load_ocr_kor()
    reader_hanja = load_ocr_hanja()
except Exception as e:
    st.error(f"OCR 로딩 중 오류가 발생했습니다: {e}")

uploaded_file = st.file_uploader("이미지를 선택하거나 직접 촬영하세요 (한글/한자 혼용 가능)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        pil_image = resize_image(pil_image, max_size=1024)
        img_np = np.array(pil_image)

        with st.spinner("한글과 한자를 동시에 분석하고 있습니다..."):
            # 1. 한글 전용 인식
            results_kor = reader_kor.readtext(img_np, paragraph=False)
            # 2. 한자 전용 인식
            results_hanja = reader_hanja.readtext(img_np, paragraph=False)

        pure_original_list = []
        pure_korean_list = []

        draw = ImageDraw.Draw(pil_image)

        # 1차: 한글 추출
        if results_kor:
            for bbox, text, prob in results_kor:
                # pure 한글 및 기본 글자 수집
                if prob > 0.3 and text.strip():
                    pt1 = (int(bbox[0][0]), int(bbox[0][1]))
                    pt2 = (int(bbox[2][0]), int(bbox[2][1]))
                    
                    pure_original_list.append(text)
                    pure_korean_list.append(text)
                    draw.rectangle([pt1, pt2], outline=(0, 255, 0), width=2)

        # 2차: 한자만 골라내어 추가
        if results_hanja:
            for bbox, text, prob in results_hanja:
                hanja_chars = re.findall(r'[\u4e00-\u9fff]', text)
                if hanja_chars:
                    found_hanja = "".join(hanja_chars)
                    pt1 = (int(bbox[0][0]), int(bbox[0][1]))
                    pt2 = (int(bbox[2][0]), int(bbox[2][1]))

                    converted_text = hanja.translate(found_hanja, 'substitution')
                    
                    pure_original_list.append(f"[{found_hanja}]")
                    pure_korean_list.append(f"[{converted_text}]")
                    draw.rectangle([pt1, pt2], outline=(255, 0, 0), width=3) # 한자는 빨간색 박스

        st.image(pil_image, caption="인식 위치 (초록: 한글 / 빨강: 한자)", use_container_width=True)

        original_str = " ".join(pure_original_list) if pure_original_list else "인식된 글자가 없습니다."
        korean_str = " ".join(pure_korean_list) if pure_korean_list else "-"

        st.subheader("=== 인식된 원본 (한글 + [한자]) ===")
        st.code(original_str, language="text")

        st.subheader("=== 한글 최종 변환 결과 ===")
        st.code(korean_str, language="text")

    except Exception as e:
        st.error(f"이미지 처리 중 오류가 발생했습니다: {e}")

# 하단 서명
st.markdown("---")
st.markdown("<h3 style='text-align: center; color: #888888;'>- 카메라대감 -</h3>", unsafe_allow_html=True)
