import cv2
import mediapipe as mp
import serial
import serial.tools.list_ports
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ========================================================================================================
# Serial Communication Setup
# ========================================================================================================

def auto_detect_port():
    """Arduino 포트 자동 감지"""
    print("[Serial] Scanning for Arduino devices...")
    
    try:
        ports = serial.tools.list_ports.comports()
        arduino_ports = []
        
        for port in ports:
            arduino_keywords = [
                'Arduino', 'CH340', 'CP210', 'FTDI', 
                'USB Serial', 'USB-SERIAL', 'USB 직렬', 'ttyUSB', 'ttyACM'
            ]
            
            port_info = f"{port.device} - {port.description} - {port.manufacturer}"
            
            # Windows COM 포트는 모두 허용 또는 Arduino 관련 키워드 확인
            is_windows_com = port.device.startswith('COM')
            is_arduino_keyword = any(keyword.lower() in port_info.lower() for keyword in arduino_keywords)
            
            if is_windows_com or is_arduino_keyword:
                arduino_ports.append(port)
                print(f"  ✓ Found: {port.device}")
                print(f"    Description: {port.description}")
        
        if not arduino_ports:
            print("[Serial] No Arduino-like devices found")
            return None
        
        if len(arduino_ports) == 1:
            selected_port = arduino_ports[0].device
            print(f"[Serial] Auto-selected: {selected_port}")
            return selected_port
        
        # 여러 개 발견된 경우 - 첫 번째 자동 선택
        selected_port = arduino_ports[0].device
        print(f"[Serial] Multiple devices found. Auto-selected: {selected_port}")
        return selected_port
        
    except Exception as e:
        print(f"[Serial] Error during port detection: {e}")
        return None

def connect_serial(port=None, baud_rate=115200):
    """시리얼 포트 연결"""
    try:
        if port is None:
            port = auto_detect_port()
        
        if port is None:
            print("[Serial] No port available")
            return None
        
        arduino = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Arduino 리셋 대기
        print(f"[Serial] Connected to {port}")
        return arduino
    except Exception as e:
        print(f"[Serial] Connection failed: {e}")
        return None

def send_motor_command(arduino, motor_positions):
    """
    모터 위치 명령 전송
    motor_positions: 7개 모터의 위치값 리스트 [M1, M2, M3, M4, M5, M6, M7]
    """
    if arduino is None:
        return False
    
    try:
        positions_int = [int(pos) for pos in motor_positions]
        command = f"0,{','.join(map(str, positions_int))}*"
        print(f"[Command] {command}")
        arduino.write(command.encode('utf-8'))
        print(f"[TX] {command}")
        return True
    except Exception as e:
        print(f"[Serial TX] {e}")
        return False

# ========================================================================================================

# ========================================================================================================
# Main Program
# ========================================================================================================

# Arduino 연결
arduino = connect_serial()

# 카메라 사용 가능 여부 확인
# for i in range(5):
#     cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
#     if cap.isOpened():
#         print(f"카메라 {i} 사용 가능")
#         cap.release()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 모터 제어 변수
MOTOR_CENTER = [512,380]        # 모터 중앙 위치 (0~1023의 중간값)
MOTOR_RANGE = [(130,130),(120,100)]      # 중앙에서 좌우로 얼마나 움직일지 (±200)
MOTOR_MIN = [MOTOR_CENTER[0] - MOTOR_RANGE[0][0], MOTOR_CENTER[1] - MOTOR_RANGE[1][0]]  # 312 (최소값)
MOTOR_MAX = [MOTOR_CENTER[0] + MOTOR_RANGE[0][0], MOTOR_CENTER[1] + MOTOR_RANGE[1][1]]  # 712 (최대값)

# 7개 모터 위치 초기화 (Base 모터만 제어, 나머지는 기본값)
motor_positions = [
    MOTOR_CENTER[0], # M1 (Base) - 코 추적으로 제어
    512,           # M2 (Shoulder)
    MOTOR_CENTER[1], # M3 (Upper_Arm)
    800,           # M4 (Elbow)
    615,           # M5 (Forearm)
    512,           # M6 (Wrist)
    512            # M7 (Hand)
]

GAIN = [1.0, 1.0]      # 반응 민감도 (클수록 빠르게 반응)
DEAD_ZONE = 10           # 이 픽셀 이내 오차는 무시 (떨림 방지)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전
        h, w, _ = frame.shape # 프레임의 높이와 너비 가져오기(480, 640)

        # 중심 가이드 라인 그리기 (십자가)
        center_x = w // 2
        center_y = h // 2
        
        # 수평선
        cv2.line(frame, (0, center_y), (w, center_y), (128, 128, 128), 1)
        # 수직선
        cv2.line(frame, (center_x, 0), (center_x, h), (128, 128, 128), 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 손바닥 중앙 계산 (손목(0)과 중지 MCP(9)의 중점)
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]
                
                palm_center_x = (wrist.x + middle_mcp.x) / 2
                palm_center_y = (wrist.y + middle_mcp.y) / 2
                
                palm_x = int(palm_center_x * w)
                palm_y = int(palm_center_y * h)

                # 손바닥 중앙 위치를 모터 위치로 직접 매핑 (절대 위치 제어)
                # palm_x 범위: 0 ~ w (640)
                # 모터 범위: MOTOR_MIN ~ MOTOR_MAX
                # 좌우 (M1) - 대칭이므로 기존 방식 OK
                motor_positions[0] = MOTOR_MIN[0] + palm_x * (MOTOR_MAX[0] - MOTOR_MIN[0]) / w

                # 상하 (M3) - 비대칭이므로 중앙 기준으로 분리
                if palm_y < h/2:  # 위쪽 → 아래로 매핑
                    motor_positions[2] = MOTOR_CENTER[1] - (h/2 - palm_y) * (MOTOR_CENTER[1] - MOTOR_MIN[1]) / (h/2)
                else:  # 아래쪽 → 위로 매핑
                    motor_positions[2] = MOTOR_CENTER[1] + (palm_y - h/2) * (MOTOR_MAX[1] - MOTOR_CENTER[1]) / (h/2)
                
                # 범위 제한
                motor_positions[0] = max(MOTOR_MIN[0], min(MOTOR_MAX[0], motor_positions[0]))
                motor_positions[2] = max(MOTOR_MIN[1], min(MOTOR_MAX[1], motor_positions[2]))
                
                # Arduino로 명령 전송
                send_motor_command(arduino, motor_positions)
                
                print(f"Base 모터: {int(motor_positions[0])}, palm_x: {palm_x}px")
                print(f"Upper_Arm 모터: {int(motor_positions[2])}, palm_y: {palm_y}px")

                # 손바닥 중앙 위치 강조 표시
                cv2.circle(frame, (palm_x, palm_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (palm_x, palm_y), 12, (255, 255, 255), 2)

                # 출력할 좌표값
                text = f"Palm Center ({palm_x}, {palm_y})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7
                thickness = 2

                # 좌표값 크기 측정
                (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

                # 좌표값이 손바닥 중앙 아래에 보이도록 y를 약간 아래로 이동
                offset = 20  # 좌표값을 손바닥 중앙 아래로 내리는 픽셀 수

                # 중앙 정렬 + 아래로 offset
                text_x = palm_x - text_w // 2
                text_y = palm_y + text_h + offset

                # 좌표값 출력
                cv2.putText(frame, text, (text_x, text_y), font, scale, (0, 255, 0), thickness)
                
                # 모터 정보 표시
                motor_text = f"Base Motor: {int(motor_positions[0])} ({int(motor_positions[0] * 300 / 1023)}deg)"
                cv2.putText(frame, motor_text, (10, 30), font, 0.7, (255, 255, 255), 2)

        cv2.imshow("Hand Palm Center Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 종료 시 시리얼 연결 해제
if arduino:
    arduino.close()
    print("[Serial] Connection closed")
            
cap.release()
cv2.destroyAllWindows()
