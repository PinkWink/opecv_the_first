import cv2
import numpy as np

# 1) 이미지 경로
img_path = "img/young_pw_resize.jpg"

# 2) 이미지 읽기
img = cv2.imread(img_path)

if img is None:
    print("이미지를 불러오지 못했습니다. 경로를 확인해주세요.")
    exit()

# 3) BGR 채널 분리
b, g, r = cv2.split(img)

# 4) 빈 채널(0으로 채운 이미지) 생성
zeros = np.zeros_like(b)

# 5) 각 색상만 살린 컬러 이미지 생성
blue_img  = cv2.merge([b, zeros, zeros])
green_img = cv2.merge([zeros, g, zeros])
red_img   = cv2.merge([zeros, zeros, r])

# 6) 각 색상별로 출력
cv2.imshow("Blue Channel (Color)", blue_img)
cv2.imshow("Green Channel (Color)", green_img)
cv2.imshow("Red Channel (Color)", red_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
