import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1) 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 2) 첫 프레임 읽어서 초기 그림 세팅에 사용
ret, frame = cap.read()
if not ret:
    print("첫 프레임을 읽지 못했습니다.")
    cap.release()
    exit()

# 3) 크기 줄이기 (320 x 240)
frame_resized = cv2.resize(frame, (320, 240))

# 4) BGR → RGB
rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

# 5) 채널 분리
r = rgb[:, :, 0]
g = rgb[:, :, 1]
b = rgb[:, :, 2]
zeros = np.zeros_like(r)

# R/G 분리 영상 (색상 그대로 보이도록)
red_img   = np.stack([r, zeros, zeros], axis=2)
green_img = np.stack([zeros, g, zeros], axis=2)

# 처음에는 가중치 1,1,1 적용한 결과 (원본과 동일)
weighted_img = rgb.copy()

# 6) Figure / Subplots 만들기
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)  # 슬라이더 공간 확보

ax_orig   = axs[0, 0]
ax_red    = axs[0, 1]
ax_green  = axs[1, 0]
ax_weight = axs[1, 1]

im_orig = ax_orig.imshow(rgb)
ax_orig.set_title("Original RGB")
ax_orig.axis("off")

im_red = ax_red.imshow(red_img)
ax_red.set_title("Red Channel")
ax_red.axis("off")

im_green = ax_green.imshow(green_img)
ax_green.set_title("Green Channel")
ax_green.axis("off")

im_weight = ax_weight.imshow(weighted_img)
ax_weight.set_title("Weighted RGB")
ax_weight.axis("off")

# 7) 슬라이더 세팅 (0.0 ~ 2.0, 기본값 1.0)
ax_r = fig.add_axes([0.1, 0.15, 0.8, 0.03])
ax_g = fig.add_axes([0.1, 0.10, 0.8, 0.03])
ax_b = fig.add_axes([0.1, 0.05, 0.8, 0.03])

slider_r = Slider(ax_r, "R weight", 0.0, 2.0, valinit=1.0)
slider_g = Slider(ax_g, "G weight", 0.0, 2.0, valinit=1.0)
slider_b = Slider(ax_b, "B weight", 0.0, 2.0, valinit=1.0)

plt.show()

# 8) 루프 돌면서 실시간 업데이트
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

    # BGR → RGB
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # 채널 분리
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    zeros = np.zeros_like(r)

    red_img   = np.stack([r, zeros, zeros], axis=2)
    green_img = np.stack([zeros, g, zeros], axis=2)

    # 슬라이더 값 읽기 (가중치)
    wr = slider_r.val
    wg = slider_g.val
    wb = slider_b.val

    # 가중치 적용 (float 연산 후 클리핑)
    rgb_float = rgb.astype(np.float32) / 255.0
    rgb_float[:, :, 0] *= wr  # R
    rgb_float[:, :, 1] *= wg  # G
    rgb_float[:, :, 2] *= wb  # B
    rgb_float = np.clip(rgb_float, 0.0, 1.0)
    weighted_img = (rgb_float * 255).astype(np.uint8)

    # 이미지 데이터 업데이트
    im_orig.set_data(rgb)
    im_red.set_data(red_img)
    im_green.set_data(green_img)
    im_weight.set_data(weighted_img)

    # 화면 갱신
    plt.pause(0.001)

# 9) 자원 해제
cap.release()
plt.ioff()
plt.close("all")
