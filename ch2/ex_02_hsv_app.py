import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class HSVSVSliderApp:
    def __init__(self, img_path: str):
        self.img_path = img_path

        # 1) 이미지 읽기 (BGR)
        img_bgr = cv2.imread(self.img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"이미지를 불러오지 못했습니다: {self.img_path}")

        # 필요하면 리사이즈 가능 (주석 해제해서 사용)
        # img_bgr = cv2.resize(img_bgr, (640, 360))

        # 2) BGR → RGB (원본 표시용)
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 3) BGR → HSV
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        self.h, self.s, self.v = cv2.split(img_hsv)

        # 초기 가중치
        self.init_ws = 1.0
        self.init_wv = 1.0

        # 4) Figure, Subplots, Slider 세팅
        self._setup_figure_and_widgets()

    def _apply_sv(self, ws: float, wv: float):
        """
        주어진 S, V 가중치(ws, wv)를 적용한
        S-only, V-only, S&V 결과 이미지를 RGB로 반환
        """
        # float 연산
        s_float = self.s.astype(np.float32) * ws
        v_float = self.v.astype(np.float32) * wv

        # 범위 클리핑 후 uint8로 변환
        s_adj = np.clip(s_float, 0, 255).astype(np.uint8)
        v_adj = np.clip(v_float, 0, 255).astype(np.uint8)

        # (1) S만 조절 (V는 원래값)
        hsv_s_only = cv2.merge([self.h, s_adj, self.v])
        bgr_s_only = cv2.cvtColor(hsv_s_only, cv2.COLOR_HSV2BGR)
        rgb_s_only = cv2.cvtColor(bgr_s_only, cv2.COLOR_BGR2RGB)

        # (2) V만 조절 (S는 원래값)
        hsv_v_only = cv2.merge([self.h, self.s, v_adj])
        bgr_v_only = cv2.cvtColor(hsv_v_only, cv2.COLOR_HSV2BGR)
        rgb_v_only = cv2.cvtColor(bgr_v_only, cv2.COLOR_BGR2RGB)

        # (3) S, V 둘 다 조절
        hsv_sv = cv2.merge([self.h, s_adj, v_adj])
        bgr_sv = cv2.cvtColor(hsv_sv, cv2.COLOR_HSV2BGR)
        rgb_sv = cv2.cvtColor(bgr_sv, cv2.COLOR_BGR2RGB)

        return rgb_s_only, rgb_v_only, rgb_sv

    def _setup_figure_and_widgets(self):
        """matplotlib Figure, Subplot, Slider 초기 구성"""
        # 초기 가중치 적용 이미지 생성
        rgb_s_only_init, rgb_v_only_init, rgb_sv_init = self._apply_sv(
            self.init_ws, self.init_wv
        )

        # Figure / Subplots
        self.fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        plt.subplots_adjust(left=0.1, bottom=0.25)  # 슬라이더 영역 확보

        ax_orig, ax_s_only = axs[0]
        ax_v_only, ax_sv = axs[1]

        # 이미지 표시 (핸들 저장)
        self.im_orig = ax_orig.imshow(self.img_rgb)
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

        # Slider 영역 추가
        ax_ws = self.fig.add_axes([0.1, 0.15, 0.8, 0.03])
        ax_wv = self.fig.add_axes([0.1, 0.08, 0.8, 0.03])

        self.slider_ws = Slider(ax_ws, "S weight", 0.0, 2.0, valinit=self.init_ws)
        self.slider_wv = Slider(ax_wv, "V weight", 0.0, 2.0, valinit=self.init_wv)

        # 슬라이더 변경 시 콜백 등록
        self.slider_ws.on_changed(self._on_slider_change)
        self.slider_wv.on_changed(self._on_slider_change)

    def _on_slider_change(self, val):
        """슬라이더 변경 시 호출되는 콜백 함수"""
        ws = self.slider_ws.val
        wv = self.slider_wv.val

        rgb_s_only, rgb_v_only, rgb_sv = self._apply_sv(ws, wv)

        self.im_s_only.set_data(rgb_s_only)
        self.im_v_only.set_data(rgb_v_only)
        self.im_sv.set_data(rgb_sv)

        self.fig.canvas.draw_idle()

    def run(self):
        """앱 실행 (블로킹)"""
        plt.show()


def main():
    img_path = "img/macau.jpeg"
    app = HSVSVSliderApp(img_path)
    app.run()


if __name__ == "__main__":
    main()
