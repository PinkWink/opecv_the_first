import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class HSVSVCameraApp:
    def __init__(self, cam_index: int = 0):
        self.cam_index = cam_index

        # 1) 카메라 열기
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다. index={self.cam_index}")

        # 2) 첫 프레임 읽어서 초기 화면 구성
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError("첫 프레임을 읽지 못했습니다.")

        # 3) 크기 줄이기 (원하는 해상도로 조정 가능)
        frame_resized = cv2.resize(frame, (320, 240))

        # 4) BGR → RGB
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # HSV로 변환 (H/S/V는 매 프레임마다 다시 계산하지만,
        # 여기서는 초기 화면용으로 한 번 계산해서 그림 세팅에 사용)
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 초기 가중치
        self.init_ws = 1.0
        self.init_wv = 1.0

        # Figure, Subplot, Slider 세팅
        self._setup_figure_and_widgets(rgb, h, s, v)

    def _apply_sv_to_frame(self, frame_bgr, ws: float, wv: float):
        """
        한 프레임(BGR)에 대해 S/V 가중치를 적용한
        S-only, V-only, S&V 결과 이미지를 RGB로 반환
        """
        # BGR → HSV
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # float 연산
        s_float = s.astype(np.float32) * ws
        v_float = v.astype(np.float32) * wv

        # 범위 클리핑 후 uint8 변환
        s_adj = np.clip(s_float, 0, 255).astype(np.uint8)
        v_adj = np.clip(v_float, 0, 255).astype(np.uint8)

        # (1) S만 조절 (V는 원래값)
        hsv_s_only = cv2.merge([h, s_adj, v])
        bgr_s_only = cv2.cvtColor(hsv_s_only, cv2.COLOR_HSV2BGR)
        rgb_s_only = cv2.cvtColor(bgr_s_only, cv2.COLOR_BGR2RGB)

        # (2) V만 조절 (S는 원래값)
        hsv_v_only = cv2.merge([h, s, v_adj])
        bgr_v_only = cv2.cvtColor(hsv_v_only, cv2.COLOR_HSV2BGR)
        rgb_v_only = cv2.cvtColor(bgr_v_only, cv2.COLOR_BGR2RGB)

        # (3) S, V 둘 다 조절
        hsv_sv = cv2.merge([h, s_adj, v_adj])
        bgr_sv = cv2.cvtColor(hsv_sv, cv2.COLOR_HSV2BGR)
        rgb_sv = cv2.cvtColor(bgr_sv, cv2.COLOR_BGR2RGB)

        return rgb_s_only, rgb_v_only, rgb_sv

    def _setup_figure_and_widgets(self, rgb_init, h_init, s_init, v_init):
        """matplotlib Figure, Subplots, Sliders를 초기화"""
        plt.ion()

        # 초기 프레임에 대한 S/V 조절 결과
        hsv_init = cv2.merge([h_init, s_init, v_init])
        bgr_init = cv2.cvtColor(hsv_init, cv2.COLOR_HSV2BGR)
        rgb_init = cv2.cvtColor(bgr_init, cv2.COLOR_BGR2RGB)

        rgb_s_only_init, rgb_v_only_init, rgb_sv_init = self._apply_sv_to_frame(
            cv2.cvtColor(rgb_init, cv2.COLOR_RGB2BGR),
            self.init_ws,
            self.init_wv,
        )

        # 2x2 subplot
        self.fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        plt.subplots_adjust(left=0.1, bottom=0.25)  # 슬라이더 공간

        ax_orig, ax_s_only = axs[0]
        ax_v_only, ax_sv = axs[1]

        # 원본은 HSV 조작 전의 RGB (카메라 프레임에서 바로 변환한 것)
        self.im_orig = ax_orig.imshow(rgb_init)
        ax_orig.set_title("Original RGB")
        ax_orig.axis("off")

        self.im_s_only = ax_s_only.imshow(rgb_s_only_init)
        ax_s_only.set_title("S adjusted")
        ax_s_only.axis("off")

        self.im_v_only = ax_v_only.imshow(rgb_v_only_init)
        ax_v_only.set_title("V adjusted")
        ax_v_only.axis("off")

        self.im_sv = ax_sv.imshow(rgb_sv_init)
        ax_sv.set_title("S & V adjusted")
        ax_sv.axis("off")

        # 슬라이더 영역
        ax_ws = self.fig.add_axes([0.1, 0.15, 0.8, 0.03])
        ax_wv = self.fig.add_axes([0.1, 0.08, 0.8, 0.03])

        self.slider_ws = Slider(ax_ws, "S weight", 0.0, 2.0, valinit=self.init_ws)
        self.slider_wv = Slider(ax_wv, "V weight", 0.0, 2.0, valinit=self.init_wv)

        # 슬라이더 콜백은 굳이 프레임 재계산을 여기서 할 필요는 없고,
        # run() 루프 내부에서 매 프레임마다 slider 값 읽어 쓰면 되므로
        # 여기서는 draw만 유도하는 간단 콜백으로 둔다.
        def _on_slider_change(_val):
            self.fig.canvas.draw_idle()

        self.slider_ws.on_changed(_on_slider_change)
        self.slider_wv.on_changed(_on_slider_change)

    def run(self):
        """카메라 루프 실행 (Matplotlib 창을 닫을 때까지)"""
        plt.show()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다. 종료합니다.")
                break

            # matplotlib 창이 닫혔는지 확인
            if not plt.fignum_exists(self.fig.number):
                print("창이 닫혔습니다. 종료합니다.")
                break

            # 크기 줄이기
            frame_resized = cv2.resize(frame, (320, 240))

            # BGR → RGB (원본)
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # 슬라이더 값 읽기
            ws = self.slider_ws.val
            wv = self.slider_wv.val

            # HSV 기반 S/V 조절 결과 생성
            rgb_s_only, rgb_v_only, rgb_sv = self._apply_sv_to_frame(
                frame_resized, ws, wv
            )

            # 이미지 데이터 업데이트
            self.im_orig.set_data(rgb)
            self.im_s_only.set_data(rgb_s_only)
            self.im_v_only.set_data(rgb_v_only)
            self.im_sv.set_data(rgb_sv)

            # 화면 갱신
            plt.pause(0.001)

        # 종료 처리
        self.cap.release()
        plt.ioff()
        plt.close("all")


def main():
    app = HSVSVCameraApp(cam_index=0)
    app.run()


if __name__ == "__main__":
    main()
