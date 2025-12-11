import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) 카메라 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 2) 첫 프레임 읽어서 초기화용으로 사용
ret, frame = cap.read()
if not ret:
    print("첫 프레임을 읽지 못했습니다.")
    cap.release()
    exit()

# 3) 크기 줄이기 (320 x 240)
frame_resized = cv2.resize(frame, (320, 240))

# 4) BGR → RGB
rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

b, g, r = cv2.split(frame_resized)
zeros = np.zeros_like(b)

red_img   = cv2.merge([zeros, zeros, r])
green_img = cv2.merge([zeros, g, zeros])
blue_img  = cv2.merge([b, zeros, zeros])

red_img   = cv2.cvtColor(red_img,   cv2.COLOR_BGR2RGB)
green_img = cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB)
blue_img  = cv2.cvtColor(blue_img,  cv2.COLOR_BGR2RGB)

# 5) Figure & Subplots 생성 (이미지 핸들 준비)
fig = plt.figure(figsize=(12, 6))

ax1 = plt.subplot(2, 3, 1)
im1 = ax1.imshow(rgb)
ax1.set_title("Original RGB")
ax1.axis('off')

ax2 = plt.subplot(2, 3, 2)
im2 = ax2.imshow(gray, cmap='gray')
ax2.set_title("Gray")
ax2.axis('off')

ax3 = plt.subplot(2, 3, 4)
im3 = ax3.imshow(red_img)
ax3.set_title("Red Channel")
ax3.axis('off')

ax4 = plt.subplot(2, 3, 5)
im4 = ax4.imshow(green_img)
ax4.set_title("Green Channel")
ax4.axis('off')

ax5 = plt.subplot(2, 3, 6)
im5 = ax5.imshow(blue_img)
ax5.set_title("Blue Channel")
ax5.axis('off')

plt.tight_layout()
plt.ion()   # interactive mode ON
plt.show()

# 6) 루프 돌면서 영상 업데이트
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽지 못했습니다.")
        break

    # matplotlib 창이 닫혔는지 확인
    if not plt.fignum_exists(fig.number):
        print("창이 닫혀서 종료합니다.")
        break

    # 크기 줄이기
    frame_resized = cv2.resize(frame, (320, 240))

    # BGR → RGB / Gray
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(frame_resized)
    zeros = np.zeros_like(b)

    red_img   = cv2.merge([zeros, zeros, r])
    green_img = cv2.merge([zeros, g, zeros])
    blue_img  = cv2.merge([b, zeros, zeros])

    red_img   = cv2.cvtColor(red_img,   cv2.COLOR_BGR2RGB)
    green_img = cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB)
    blue_img  = cv2.cvtColor(blue_img,  cv2.COLOR_BGR2RGB)

    # 이미지 데이터 업데이트
    im1.set_data(rgb)
    im2.set_data(gray)
    im3.set_data(red_img)
    im4.set_data(green_img)
    im5.set_data(blue_img)

    # 화면 갱신
    plt.pause(0.001)   # 너무 작게/크게 조절해보면서 FPS 감각 보기

# 7) 자원 해제
cap.release()
plt.ioff()
plt.close('all')
