import cv2

# 1) 카메라 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    # 2) 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽지 못했습니다.")
        break

    # 3) 프레임 크기 줄이기 (320x240)
    resized = cv2.resize(frame, (320, 240))

    # 4) Gray 변환 후 동일 크기로 resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (320, 240))

    # 5) 두 개의 창으로 출력
    cv2.imshow("Camera Color 320x240", resized)
    cv2.imshow("Camera Gray 320x240", gray_resized)

    # 6) 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7) 자원 해제
cap.release()
cv2.destroyAllWindows()
