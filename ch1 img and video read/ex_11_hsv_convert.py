import cv2
import numpy as np

# 1) 이미지 읽기
img_path = "img/young_pw_resize.jpg"
img = cv2.imread(img_path)

if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# 2) BGR → HSV 변환
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV 채널 분리
h, s, v = cv2.split(hsv)

# 3) S(채도) 50% 감소
s_half = (s * 0.5).astype(np.uint8)

# 4) V(명도) 50% 감소
v_half = (v * 0.5).astype(np.uint8)

# 5) 다시 합쳐서 BGR로 변환

# (1) S만 줄인 영상
hsv_s_half = cv2.merge([h, s_half, v])
bgr_s_half = cv2.cvtColor(hsv_s_half, cv2.COLOR_HSV2BGR)

# (2) V만 줄인 영상
hsv_v_half = cv2.merge([h, s, v_half])
bgr_v_half = cv2.cvtColor(hsv_v_half, cv2.COLOR_HSV2BGR)

# 6) 새 창에 출력
cv2.imshow("Original", img)
cv2.imshow("S reduced 50%", bgr_s_half)
cv2.imshow("V reduced 50%", bgr_v_half)

cv2.waitKey(0)
cv2.destroyAllWindows()
