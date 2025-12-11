import cv2

# 1) 카메라 열기 (0번 = 기본 카메라)
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

    # 3) 프레임 크기 줄이기 (320 x 240)
    resized = cv2.resize(frame, (320, 240))

    # 4) 줄인 프레임 출력
    cv2.imshow("Camera 320x240", resized)

    # 5) q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6) 자원 해제
cap.release()
cv2.destroyAllWindows()
