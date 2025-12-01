import cv2
import numpy as np
import os

# ================= НАСТРОЙКИ =================
OUTPUT_DIR = "my_item_dataset" # Новая папка для предметов
# =============================================

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

manual_points = []
saved_count = 0

# Ищем последний номер файла
existing = os.listdir(f"{OUTPUT_DIR}/images")
if existing:
    nums = [int(f.split('_')[1].split('.')[0]) for f in existing if 'img_' in f]
    if nums: saved_count = max(nums) + 1

def mouse_callback(event, x, y, flags, param):
    global manual_points
    # Записываем клики (максимум 2 точки: левый-верх и правый-низ)
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(manual_points) < 2:
            manual_points.append((x, y))

def find_active_camera():
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            for _ in range(5): cap.read()
            return cap
    return None

def main():
    global manual_points, saved_count
    
    cap = find_active_camera()
    if not cap:
        print("Камера не найдена")
        return

    cap.set(3, 1280)
    cap.set(4, 720)

    cv2.namedWindow("Item Creator")
    cv2.setMouseCallback("Item Creator", mouse_callback)

    print("--- ИНСТРУКЦИЯ ДЛЯ ПРЕДМЕТА ---")
    print("1. Кликни Левый-Верхний угол предмета.")
    print("2. Кликни Правый-Нижний угол предмета.")
    print("   (Скрипт сам найдет центр захвата)")
    print("3. Нажми 's' для сохранения.")
    print("4. 'r' - сброс.")
    print("-------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]

        # Рисуем точки кликов
        for pt in manual_points:
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)

        # Если есть 2 точки - рисуем прямоугольник и центр
        if len(manual_points) == 2:
            pt1 = manual_points[0]
            pt2 = manual_points[1]
            
            # Рисуем зеленую рамку
            cv2.rectangle(display_frame, pt1, pt2, (0, 255, 0), 2)
            
            # Вычисляем центр (точка захвата)
            cx = int((pt1[0] + pt2[0]) / 2)
            cy = int((pt1[1] + pt2[1]) / 2)
            cv2.circle(display_frame, (cx, cy), 8, (0, 255, 255), -1) # Желтая точка
            
            cv2.putText(display_frame, "CENTER (GRASP)", (cx + 10, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.putText(display_frame, "PRESS 's' TO SAVE", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(display_frame, f"Items Saved: {saved_count}", (w - 250, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Item Creator", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # === СОХРАНЕНИЕ ===
        if key == ord('s') and len(manual_points) == 2:
            x1, y1 = manual_points[0]
            x2, y2 = manual_points[1]
            
            # Упорядочиваем координаты (вдруг ты кликнула сначала низ, потом верх)
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)

            # Вычисляем данные для YOLO (нормированные 0..1)
            box_w = (max_x - min_x) / w
            box_h = (max_y - min_y) / h
            box_cx = ((min_x + max_x) / 2) / w
            box_cy = ((min_y + max_y) / 2) / h
            
            # Точка захвата (центр бокса)
            kpt_x = box_cx
            kpt_y = box_cy

            # Формат строки: Class 0 (Item) Box_CX Box_CY W H Kpt_X Kpt_Y Vis
            # Мы используем ID 0, потому что будем учить отдельную модель для предмета
            yolo_line = f"0 {box_cx:.6f} {box_cy:.6f} {box_w:.6f} {box_h:.6f} {kpt_x:.6f} {kpt_y:.6f} 2"

            filename = f"img_{saved_count:04d}"
            cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}.jpg", frame)
            with open(f"{OUTPUT_DIR}/labels/{filename}.txt", "w") as f:
                f.write(yolo_line)

            print(f"Предмет сохранен: {filename}")
            saved_count += 1
            manual_points = [] # Сброс

        elif key == ord('r'):
            manual_points = []
            print("Сброс.")
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
