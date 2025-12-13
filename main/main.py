import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ================= 설정 (CONFIGURATION) =================
TABLE_WIDTH_CM = 60.0  # 테이블의 실제 너비 (cm)
TABLE_HEIGHT_CM = 45.0 # 테이블의 실제 높이 (cm)

# 객체 모델 경로만
MODEL_ITEM_PATH  = r"C:\dev\weight.pt" # 실제 모델 파일 경로로 변경하세요

CONFIDENCE_ITEM = 0.5 # 객체 감지 신뢰도 임계값
# ================================================

def transform_value(value, old_min, old_max, new_min, new_max):
    """
    값을 원래 범위에서 새로운 범위로 선형적으로 변환하고,
    범위를 벗어나는 값은 새로운 범위의 최대/최소값으로 클리핑합니다.
    """
    # 1. 분모가 0인 경우(old_min == old_max) 예외 처리
    if old_max - old_min == 0:
        return new_min
    
    # 2. 클리핑(Clamping)
    # value가 old_min보다 작으면 new_min으로 바로 변환
    if value <= old_min:
        return new_min
    # value가 old_max보다 크면 new_max로 바로 변환
    if value >= old_max:
        return new_max
    
    # 3. 정규화 및 스케일링 (범위 내 값만 이 단계 실행)
    
    # 정규화: (value - old_min) / (old_max - old_min) -> 0.0 ~ 1.0 사이의 값
    normalized_value = (value - old_min) / (old_max - old_min)
    
    # 새로운 범위로 스케일링 및 이동: normalized_value * (new_max - new_min) + new_min
    new_value = normalized_value * (new_max - new_min) + new_min
    
    return new_value

def transform_coordinates(x, y):
    """
    주어진 조건에 따라 x와 y 좌표를 변환합니다.
    x : -3 ~ 63 ==> 3 ~ 57
    y : -3 ~ 48 ==> 3 ~ 42
    """
    # x 변환
    x_old_min, x_old_max = -3, 63
    x_new_min, x_new_max = 3, 57
    x_prime = transform_value(x, x_old_min, x_old_max, x_new_min, x_new_max)
    
    # y 변환
    y_old_min, y_old_max = -3, 48
    y_new_min, y_new_max = 3, 42
    y_prime = transform_value(y, y_old_min, y_old_max, y_new_min, y_new_max)
    
    return x_prime, y_prime

class Smoother:
    """좌표를 부드럽게 처리합니다 (이동 평균)"""
    def __init__(self, buffer_size=5):
        self.buffer = deque(maxlen=buffer_size)
    
    def update(self, val):
        """새 값을 추가하고 부드럽게 처리된 값을 반환합니다."""
        self.buffer.append(val)
        return sum(self.buffer) / len(self.buffer)

# 스무더 인스턴스
smooth_x = Smoother(buffer_size=5)
smooth_y = Smoother(buffer_size=5)

# 마우스 클릭을 위한 전역 변수
calibration_corners = [] # 보정(Calibration)에 사용될 4개의 코너 픽셀 좌표
is_calibrated = False # 보정이 완료되었는지 여부
additional_points = [] # 캘리브레이션 후 추가로 찍은 점들

def mouse_callback(event, x, y, flags, param):
    """테이블 코너 수동 설정 및 추가 점 클릭 처리"""
    global calibration_corners, is_calibrated, additional_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not is_calibrated:
            if len(calibration_corners) < 4:
                calibration_corners.append((x, y))
                print(f"점 {len(calibration_corners)}: {x}, {y}")
        else:
            # 캘리브레이션 완료 후 추가 점 클릭
            additional_points.append((x, y))
            print(f"추가 점 클릭: 픽셀 좌표 ({x}, {y})")

def order_points(pts):
    """코너 정렬: TL(왼쪽 위), TR(오른쪽 위), BR(오른쪽 아래), BL(왼쪽 아래)"""
    # 원본 코드의 순서는 [0,0], [W,0], [W,H], [0,H]에 대응되도록 정렬됩니다:
    # TL, TR, BR, BL 순서 (Sum, Diff 이용)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 최소 합: TL (왼쪽 위)
    rect[2] = pts[np.argmax(s)] # 최대 합: BR (오른쪽 아래)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 최소 차이 (y-x): TR (오른쪽 위)
    rect[3] = pts[np.argmax(diff)] # 최대 차이 (y-x): BL (왼쪽 아래)
    return rect
    
# 원근 변환 행렬을 이해하는 데 도움이 될 수 있습니다.
# 

def draw_grid_and_axes(img, matrix, w_cm, h_cm):
    """시각화를 위해 테이블 위에 격자(Grid)를 그립니다"""
    try:
        # 역행렬을 사용하여 CM 좌표를 픽셀 좌표로 변환
        _, inv_matrix = cv2.invert(matrix)
        
        # 10cm마다 격자 그리기
        for x in range(0, int(w_cm) + 1, 10):
            # (x, 0)에서 (x, h_cm)까지의 수직선
            p1 = cv2.perspectiveTransform(np.array([[[x, 0]]], dtype=np.float32), inv_matrix)
            p2 = cv2.perspectiveTransform(np.array([[[x, h_cm]]], dtype=np.float32), inv_matrix)
            cv2.line(img, tuple(p1[0][0].astype(int)), tuple(p2[0][0].astype(int)), (0, 255, 255), 1)
        
        for y in range(0, int(h_cm) + 1, 10):
            # (0, y)에서 (w_cm, y)까지의 수평선
            p1 = cv2.perspectiveTransform(np.array([[[0, y]]], dtype=np.float32), inv_matrix)
            p2 = cv2.perspectiveTransform(np.array([[[w_cm, y]]], dtype=np.float32), inv_matrix)
            cv2.line(img, tuple(p1[0][0].astype(int)), tuple(p2[0][0].astype(int)), (0, 255, 255), 1)
            
        # 좌표 원점 (0,0) 표시
        origin = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), inv_matrix)
        cv2.circle(img, tuple(origin[0][0].astype(int)), 5, (0, 0, 255), -1)
        
    except: pass

def main():
    global is_calibrated, calibration_corners, additional_points
    
    print("--- 객체 모델 로딩 중 ---")
    try:
        net_item = YOLO(MODEL_ITEM_PATH)
    except Exception as e:
        print(f"모델 오류: {e}")
        return

    # 카메라 열기
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 0 또는 1을 시도해 보세요
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    cap.set(3, 1280) # 너비 설정
    cap.set(4, 720)  # 높이 설정

    # 창 및 마우스 이벤트 설정
    cv2.namedWindow("Work Area")
    cv2.setMouseCallback("Work Area", mouse_callback)

    # 테이블의 실제 모서리 좌표 (cm 단위)
    # [0, 0]이 테이블의 왼쪽 위 코너가 됩니다.
    real_corners = np.float32([
        [0, 0],                            # TL
        [TABLE_WIDTH_CM, 0],               # TR
        [TABLE_WIDTH_CM, TABLE_HEIGHT_CM], # BR
        [0, TABLE_HEIGHT_CM]               # BL
    ])
    
    perspective_matrix = None # 원근 변환 행렬

    print("\n=== 사용 안내 ===")
    print("테이블의 4개 코너를 어떤 순서로든 클릭하세요.")
    print("캘리브레이션 후 추가로 점을 클릭하면 cm 좌표가 표시됩니다.")
    print("'R' 키를 눌러 전체 초기화, 'C' 키를 눌러 추가 점만 초기화하세요.")
    print("'Q' 키를 눌러 종료하세요.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 보정(CALIBRATION) 단계
        if not is_calibrated:
            cv2.putText(frame, f"코너 클릭: {len(calibration_corners)}/4", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 클릭된 점 표시
            for pt in calibration_corners:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)

            # 4개의 점이 모이면 행렬 계산
            if len(calibration_corners) == 4:
                pts_src = np.array(calibration_corners, dtype="float32")
                pts_src = order_points(pts_src) # 순서 정렬
                perspective_matrix = cv2.getPerspectiveTransform(pts_src, real_corners)
                is_calibrated = True
                print("보정이 완료되었습니다! 객체 찾기를 시작합니다.")

        # 2. 작업 단계 (행렬이 있을 때)
        else:
            if perspective_matrix is not None:
                # 좌표 격자 그리기
                draw_grid_and_axes(frame, perspective_matrix, TABLE_WIDTH_CM, TABLE_HEIGHT_CM)
                
                # 추가로 클릭한 점들 표시 및 cm 좌표 계산
                for pt in additional_points:
                    # 점 표시
                    cv2.circle(frame, pt, 8, (255, 0, 255), -1)  # 보라색 원
                    
                    # cm 좌표로 변환
                    vec = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
                    real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
                    cm_x = real_pt[0][0][0]
                    cm_y = real_pt[0][0][1]
                    
                    # 좌표 변환 적용
                    cm_x, cm_y = transform_coordinates(cm_x, cm_y)
                    
                    # 텍스트 표시
                    text = f"({cm_x:.1f}, {cm_y:.1f}) cm"
                    cv2.putText(frame, text, (pt[0] + 10, pt[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    print(f"추가 점의 실제 좌표: X={cm_x:.1f}, Y={cm_y:.1f} cm")
                
                # --- 객체 찾기 (YOLO) ---
                results = net_item(frame, verbose=False, conf=CONFIDENCE_ITEM)[0]
                
                if results.boxes:
                    for i, box in enumerate(results.boxes):
                        # 바운딩 박스 좌표 얻기
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 객체의 중심점 결정
                        # (Keypoints가 있다면 Keypoint 사용, 없다면 바운딩 박스 중심 사용)
                        
                        cx, cy = 0, 0
                        has_keypoint = False

                        if results.keypoints is not None and len(results.keypoints.data) > i:
                            kpt = results.keypoints.data[i][0] # 첫 번째 키포인트 사용
                            if kpt[2] > 0.5: # 신뢰도가 높으면
                                cx = int((x1 + x2) / 2)
                                cy = int((y1 + y2) / 2)
                                has_keypoint = True
                        
                        # 키포인트가 없으면, 사각형의 중심 사용
                        if not has_keypoint:
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)

                        # 객체 및 중심점 그리기
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                        # --- CM 단위로 변환 ---
                        vec = np.array([[[cx, cy]]], dtype=np.float32)
                        # 원근 변환 적용 (픽셀 -> CM)
                        real_pt = cv2.perspectiveTransform(vec, perspective_matrix)
                        
                        raw_x = real_pt[0][0][0]
                        raw_y = real_pt[0][0][1]

                        # 스무딩(평활화)
                        val_x = smooth_x.update(raw_x)
                        val_y = smooth_y.update(raw_y)

                        # 좌표 변환 적용
                        val_x, val_y = transform_coordinates(val_x, val_y)

                        # 테이블 경계 확인
                        in_bounds = (0 <= val_x <= TABLE_WIDTH_CM) and (0 <= val_y <= TABLE_HEIGHT_CM)
                        color_text = (0, 255, 0) if in_bounds else (0, 0, 255) # 초록색: 경계 내, 빨간색: 경계 밖

                        # 텍스트 출력
                        text = f"X:{val_x:.1f} Y:{val_y:.1f} cm"
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2)

        # 키보드 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break # 'q'를 누르면 종료
        if key == ord('r'): # 'r'을 누르면 보정 초기화
            calibration_corners = []
            is_calibrated = False
            perspective_matrix = None
            additional_points = []
            print("보정이 초기화되었습니다.")
        if key == ord('c'): # 'c'를 누르면 추가 점만 초기화
            additional_points = []
            print("추가 점이 초기화되었습니다.")

        cv2.imshow("Work Area", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()