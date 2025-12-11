import cv2
import matplotlib.pyplot as plt

# 1) 이미지 읽기 (OpenCV = BGR)
img_path = "img/young_pw_resize.jpg"
img_bgr = cv2.imread(img_path)

if img_bgr is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# 2) BGR → RGB 변환
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 3) Matplotlib으로 새창에 출력
plt.figure("Matplotlib Image Window")
plt.imshow(img_rgb)
plt.axis("off")  # 축 제거
plt.show()
