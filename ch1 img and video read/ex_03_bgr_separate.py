import cv2

# 1) 이미지 경로
img_path = "img/young_pw_resize.jpg"

# 2) 이미지 읽기
img = cv2.imread(img_path)

if img is None:
    print("이미지를 불러오지 못했습니다. 경로를 확인해주세요.")
    exit()

# 3) BGR 채널 분리
b, g, r = cv2.split(img)

# 4) 각 채널을 창으로 표시
cv2.imshow("Blue Channel", b)
cv2.imshow("Green Channel", g)
cv2.imshow("Red Channel", r)

cv2.waitKey(0)
cv2.destroyAllWindows()
