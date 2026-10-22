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
from PIL import Image, ImageDraw, ImageEnhance

st.set_page_config(page_title="한자 OCR & 독음 변환", layout="centered")

st.title("📱 한자 OCR & 한글음 변환")

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

# EasyOCR 번체 한자 전용 모델 로딩
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_tra'], gpu=False, quantize=True)

try:
    reader = load_ocr()
except Exception as e:
    st.error(f"OCR 로딩 중 오류가 발생했습니다: {e}")

uploaded_file = st.file_uploader("한자 이미지를 선택하거나 직접 촬영하세요", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        pil_image = resize_image(pil_image, max_size=1024)

        # 이미지 전처리 (그레이스케일 및 명암 대비 강화)
        gray_image = pil_image.convert("L")
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(1.8)
        img_np = np.array(enhanced_image.convert("RGB"))

        with st.spinner("한자를 인식하고 있습니다..."):
            # paragraph=False로 설정하여 노이즈 글자끼리 강제로 엮이는 현상 방지
            results = reader.readtext(img_np, paragraph=False)

        pure_original_list = []
        pure_korean_list = []

        draw = ImageDraw.Draw(pil_image)

        if results:
            for bbox, text, prob in results:
                # 1. 인식 신뢰도가 너무 낮은 잡음 제거 (30% 미만 스킵)
                if prob < 0.3:
                    continue

                # 2. 정규식을 이용하여 텍스트 중 순수 '한자'만 추출 (\u4e00 ~ \u9fff)
                hanja_only = "".join(re.findall(r'[\u4e00-\u9fff]+', text))

                # 3. 추출된 진짜 한자가 있을 때만 결과에 반영
                if hanja_only:
                    pt1 = (int(bbox[0][0]), int(bbox[0][1]))
                    pt2 = (int(bbox[2][0]), int(bbox[2][1]))

                    converted_text = hanja.translate(hanja_only, 'substitution')
                    
                    pure_original_list.append(hanja_only)
                    pure_korean_list.append(converted_text)

                    # 박스 그리기
                    draw.rectangle([pt1, pt2], outline=(0, 255, 0), width=3)

        st.image(pil_image, caption="인식 위치 확인", use_container_width=True)

        original_str = " ".join(pure_original_list) if pure_original_list else "인식된 한자가 없습니다."
        korean_str = " ".join(pure_korean_list) if pure_korean_list else "-"

        st.subheader("=== 원본 한자 ===")
        st.code(original_str, language="text")

        st.subheader("=== 한글 변환(독음) ===")
        st.code(korean_str, language="text")

    except Exception as e:
        st.error(f"이미지 처리 중 오류가 발생했습니다: {e}")

# 하단 서명
st.markdown("---")
st.markdown("<h3 style='text-align: center; color: #888888;'>- 카메라대감 -</h3>", unsafe_allow_html=True)
