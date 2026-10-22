import streamlit as st
import easyocr
import hanja
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="한자 OCR", layout="centered")

st.title("📱 한자 OCR & 한글음 변환기")

# EasyOCR 모델 캐싱 (최초 1회만 로드)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_tra'], gpu=False)

reader = load_ocr()

# 모바일 카메라 촬영 및 갤러리 업로드 지원
uploaded_file = st.file_uploader("한자 이미지를 선택하거나 직접 촬영하세요", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("한자를 인식하고 있습니다..."):
        results = reader.readtext(img)

    pure_hanja_list = []
    pure_korean_list = []

    if results:
        for i, (bbox, text, prob) in enumerate(results):
            pt1 = (int(bbox[0][0]), int(bbox[0][1]))
            pt2 = (int(bbox[2][0]), int(bbox[2][1]))

            kor_text = hanja.translate(text, 'substitution')
            pure_hanja_list.append(text)
            pure_korean_list.append(kor_text)

            # 인식 위치 박스 및 번호 표기
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(img, str(i+1), (pt1[0], max(pt1[1] - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 1. 인식 이미지 화면 출력
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="인식 위치 확인", use_column_width=True)

    # 2. 결과 텍스트 출력
    hanja_str = " ".join(pure_hanja_list)
    kor_str = " ".join(pure_korean_list)

    st.subheader("=== 원본 한자 ===")
    st.code(hanja_str, language="text")

    st.subheader("=== 한글변환 ===")
    st.code(kor_str, language="text")

st.markdown("---")
st.markdown(
    "<h4 style='text-align: center; color: #888888;'>- 카메라대감 -</h4>", 
    unsafe_allow_html=True
)
