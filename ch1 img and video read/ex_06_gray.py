import cv2

# 1) 이미지 읽기
img_path = "img/young_pw_resize.jpg"
img = cv2.imread(img_path)

if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# 2) BGR → GRAY 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3) 그레이 이미지 새 창에 띄우기
cv2.imshow("Gray Image", gray)

# 4) 그레이 이미지 저장 (선택 사항)
save_path = "img/young_pw_gray.jpg"
cv2.imwrite(save_path, gray)

print(f"그레이 이미지 저장 완료 → {save_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()
