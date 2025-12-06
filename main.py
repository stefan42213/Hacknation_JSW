import cv2
import numpy as np
import time  # <--- NOWE: do mierzenia czasu
from datetime import datetime # <--- NOWE: do ładnej daty w pliku

def process_frame_user_logic(frame):
    # --- TA FUNKCJA POZOSTAJE BEZ ZMIAN ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 9)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 19, 1.2
    )
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.erode(binary, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2) 

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    final_mask = np.zeros_like(cleaned)
    
    min_area = 80
    min_dimension = 60 

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        if area >= min_area and (width > min_dimension or height > min_dimension):
            final_mask[labels == i] = 255

    return final_mask

def log_warning_to_file():
    """Funkcja zapisująca ostrzeżenie do pliku"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] OSTRZEZENIE: Wykryto zatrzymanie tasmy!\n"
        with open("warning.txt", "a") as f: # "a" oznacza append (dopisywanie na końcu)
            f.write(msg)
        print(msg.strip())
    except Exception as e:
        print(f"Błąd zapisu pliku: {e}")

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Błąd: Nie można otworzyć wideo/kamery.")
        return

    frame_count = 0
    
    # --- ZMIENNE STABILIZACJI POZYCJI ---
    last_valid_start = None
    last_valid_end = None
    max_jump_limit = 30

    # --- ZMIENNE DO WYKRYWANIA ZATRZYMANIA (NOWE) ---
    last_movement_time = time.time() # Czas ostatniego ruchu
    prev_center_pos = 0              # Poprzednia pozycja środka
    is_stopped_flag = False          # Czy już zgłosiliśmy błąd?
    
    # KONFIGURACJA ZATRZYMANIA
    MOVEMENT_THRESHOLD = 2  # Ile pikseli musi się ruszyć, żeby uznać to za ruch
    TIME_TO_WAIT = 10        # Ile sekund bez ruchu oznacza awarię

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        height, width = frame.shape[:2]

        # 1. ALGORYTM 
        algorithm_mask = process_frame_user_logic(frame)

        # 2. KOPIA DO RYSOWANIA
        draw_frame = frame.copy()

        # 3. POMIAR 
        scan_y = int(height * 0.85)
        row_pixels = algorithm_mask[scan_y, :]
        white_indices = np.where(row_pixels > 0)[0]

        current_start = None
        current_end = None
        min_neighbors = 7
        search_window = 15
        
        if len(white_indices) > min_neighbors:
            for i in range(len(white_indices) - min_neighbors):
                curr = white_indices[i]
                ahead = white_indices[i + min_neighbors]
                if ahead - curr < search_window:
                    current_start = curr
                    break 
            
            for i in range(len(white_indices) - 1, min_neighbors, -1):
                curr = white_indices[i]
                behind = white_indices[i - min_neighbors]
                if curr - behind < search_window:
                    current_end = curr
                    break

        # Stabilizacja
        if current_start is not None and current_end is not None and current_end > current_start:
            if last_valid_start is None:
                last_valid_start = current_start
                last_valid_end = current_end
            else:
                prev_center_calc = (last_valid_start + last_valid_end) / 2
                curr_center_calc = (current_start + current_end) / 2
                
                if abs(curr_center_calc - prev_center_calc) < max_jump_limit:
                    last_valid_start = current_start
                    last_valid_end = current_end
                else:
                    pass

        # --- LOGIKA WYKRYWANIA ZATRZYMANIA (NOWA) ---
        if last_valid_start is not None and last_valid_end is not None:
            # Obliczamy aktualny środek elementu
            current_center_pos = (last_valid_start + last_valid_end) / 2
            
            # Sprawdzamy, czy zmienił się względem poprzedniej klatki
            diff = abs(current_center_pos - prev_center_pos)
            
            if diff > MOVEMENT_THRESHOLD:
                # JEST RUCH
                last_movement_time = time.time() # Resetujemy licznik czasu
                prev_center_pos = current_center_pos
                is_stopped_flag = False # Resetujemy flagę alarmu
            else:
                # BRAK RUCHU
                # Sprawdzamy ile czasu minęło od ostatniego ruchu
                elapsed_time = time.time() - last_movement_time
                
                if elapsed_time > TIME_TO_WAIT:
                    # Jeśli stoi dłużej niż limit
                    if not is_stopped_flag:
                        log_warning_to_file()
                        is_stopped_flag = True # Zapobiega zapisywaniu co klatkę
                    
                    # Rysujemy wielki napis na ekranie
                    cv2.putText(draw_frame, "ALARM: STOP TASMY!", (50, height // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)


        # --- RYSOWANIE ---
        cv2.line(draw_frame, (0, scan_y), (width, scan_y), (100, 100, 100), 1)

        if last_valid_start is not None and last_valid_end is not None:
            distance = last_valid_end - last_valid_start
            cv2.arrowedLine(draw_frame, (last_valid_start, scan_y), (last_valid_end, scan_y), (0, 0, 255), 3, tipLength=0.05)
            cv2.arrowedLine(draw_frame, (last_valid_end, scan_y), (last_valid_start, scan_y), (0, 0, 255), 3, tipLength=0.05)
            cv2.circle(draw_frame, (last_valid_start, scan_y), 6, (0, 255, 0), -1)
            cv2.circle(draw_frame, (last_valid_end, scan_y), 6, (0, 255, 0), -1)
            cv2.putText(draw_frame, f"SZEROKOSC: {distance} px", (last_valid_start + 10, scan_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(draw_frame, "Szukam krawedzi...", (50, scan_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- ŁĄCZENIE I WIZUALIZACJA ---
        mask_bgr = cv2.cvtColor(algorithm_mask, cv2.COLOR_GRAY2BGR)
        combined_view = cv2.hconcat([draw_frame, mask_bgr])
        
        scale = 0.4 
        h_comb, w_comb = combined_view.shape[:2]
        final_preview = cv2.resize(combined_view, (int(w_comb * scale), int(h_comb * scale)))

        cv2.imshow('System Inspekcji', final_preview)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_source = 'test1.mkv' 
main(video_source)