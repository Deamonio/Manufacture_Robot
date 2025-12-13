def transform_value(value, old_min, old_max, new_min, new_max):
    """
    값을 원래 범위에서 새로운 범위로 선형적으로 변환합니다.
    """
    # 분모가 0인 경우(old_min == old_max) 예외 처리
    if old_max - old_min == 0:
        return new_min
    
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

# --- 테스트 ---
# 원래 범위의 최솟값 테스트: (-3, -3) -> (3, 3) 이 되어야 함
x1, y1 = -3, -3
x1_prime, y1_prime = transform_coordinates(x1, y1)
print(f"({x1}, {y1}) -> ({x1_prime:.2f}, {y1_prime:.2f})") # 출력: (-3, -3) -> (3.00, 3.00)

# 원래 범위의 최댓값 테스트: (63, 48) -> (57, 42) 이 되어야 함
x2, y2 = 63, 48
x2_prime, y2_prime = transform_coordinates(x2, y2)
print(f"({x2}, {y2}) -> ({x2_prime:.2f}, {y2_prime:.2f})") # 출력: (63, 48) -> (57.00, 42.00)

# 중간값 테스트 (예: 30, 20)
x3, y3 = 30, 20
x3_prime, y3_prime = transform_coordinates(x3, y3)
print(f"({x3}, {y3}) -> ({x3_prime:.2f}, {y3_prime:.2f})") # 출력: (30, 20) -> (29.64, 25.12)