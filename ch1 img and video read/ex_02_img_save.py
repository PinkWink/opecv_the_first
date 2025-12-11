import cv2

# 1) 이미지 경로
img_path = "img/young_pw.jpg"

# 2) 이미지 읽기
img = cv2.imread(img_path)

if img is None:
    print("이미지를 불러오지 못했습니다. 경로를 확인해주세요.")
    exit()

# 3) 원래 이미지 크기 얻기
h, w, c = img.shape

# 4) 절반 크기로 계산
new_w = w // 2
new_h = h // 2

# 5) 이미지 리사이즈
img_resized = cv2.resize(img, (new_w, new_h))

# 6) 리사이즈된 이미지 파일로 저장
save_path = "img/young_pw_resize.jpg"
cv2.imwrite(save_path, img_resized)

# 7) 크기 출력
print(f"원래 이미지 크기 : width={w}, height={h}")
print(f"줄어든 이미지 크기 : width={new_w}, height={new_h}")
print(f"리사이즈 이미지 저장 완료 → {save_path}")

# 8) 줄어든 이미지만 화면에 표시
cv2.imshow("Resized Image (1/2)", img_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
