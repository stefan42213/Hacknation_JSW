import cv2
import numpy as np
import time
from datetime import datetime

def process_frame_user_logic(frame):
    # --- TA FUNKCJA POZOSTAJE BEZ ZMIAN (Wykrywanie krawędzi) ---
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
    """Zapisuje alarm do pliku"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] OSTRZEZENIE: Taśma produkcyjna zatrzymana!\n"
        with open("warning.txt", "a") as f:
            f.write(msg)
        print(msg.strip())
    except Exception as e:
        print(f"Błąd zapisu pliku: {e}")

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Błąd: Nie można otworzyć wideo.")
        return

    frame_count = 0
    
    # Zmienne do pomiaru szerokości
    last_valid_start = None
    last_valid_end = None
    max_jump_limit = 30

    # --- ZMIENNE DO WYKRYWANIA RUCHU TAŚMY (PIXELE) ---
    prev_gray_frame = None       # Tu zapiszemy poprzednią klatkę
    last_movement_time = time.time()
    is_stopped_flag = False
    
    # KONFIGURACJA ZATRZYMANIA
    # Ile sekund braku zmian oznacza awarię:
    TIME_TO_WAIT = 3.0           
    
    # Czułość na zmianę koloru piksela (0-255). 
    # 25 oznacza, że piksel musi zmienić się znacząco, żeby uznać to za ruch (eliminuje szum kamery).
    PIXEL_DIFF_THRESHOLD = 2   
    
    # Ile % ekranu musi się ruszać, żeby uznać, że taśma jedzie.
    # Np. 0.005 oznacza 0.5% pikseli. Jeśli taśma jest jednolita, daj mało. Jeśli wzorzysta, można więcej.
    MIN_MOTION_PERCENTAGE = 0.001 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        height, width = frame.shape[:2]
        total_pixels = height * width

        # 1. PRZYGOTOWANIE DO WYKRYWANIA RUCHU (GLOBALNEGO)
        # Konwersja na szary i lekkie rozmycie, żeby szum nie był traktowany jako ruch
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_current_blurred = cv2.GaussianBlur(gray_current, (21, 21), 0)

        motion_detected = False
        motion_ratio = 0.0

        if prev_gray_frame is not None:
            # Obliczamy różnicę między klatką obecną a poprzednią
            frame_delta = cv2.absdiff(prev_gray_frame, gray_current_blurred)
            
            # Progowanie: zaznaczamy na biało tylko te piksele, które zmieniły się mocno
            thresh = cv2.threshold(frame_delta, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            
            # Liczymy ile pikseli jest białych (zmienionych)
            changed_pixels_count = cv2.countNonZero(thresh)
            
            # Obliczamy jaki to procent całego obrazu
            motion_ratio = changed_pixels_count / total_pixels

            # Decyzja: czy jest ruch?
            if motion_ratio > MIN_MOTION_PERCENTAGE:
                motion_detected = True
                last_movement_time = time.time()
                is_stopped_flag = False
            else:
                motion_detected = False

        # Zapisujemy obecną klatkę jako "poprzednią" dla następnej pętli
        prev_gray_frame = gray_current_blurred

        # --- OBSŁUGA ALARMU ---
        elapsed_time = time.time() - last_movement_time
        
        draw_frame = frame.copy() # Kopia do rysowania

        if elapsed_time > TIME_TO_WAIT:
            # ALARM: Taśma stoi
            if not is_stopped_flag:
                log_warning_to_file()
                is_stopped_flag = True
            
            cv2.putText(draw_frame, f"ALARM: STOP TASMY! ({elapsed_time:.1f}s)", (50, height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Opcjonalnie: Wyświetlanie poziomu ruchu na ekranie (dla debugowania)
        color_status = (0, 255, 0) if motion_detected else (0, 165, 255)
        cv2.putText(draw_frame, f"Ruch pixeli: {motion_ratio:.5f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)


        # 2. ALGORYTM POMIARU (To co było wcześniej)
        algorithm_mask = process_frame_user_logic(frame)

        # 3. POMIAR SZEROKOŚCI
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

        if current_start is not None and current_end is not None and current_end > current_start:
            if last_valid_start is None:
                last_valid_start = current_start
                last_valid_end = current_end
            else:
                prev_c = (last_valid_start + last_valid_end) / 2
                curr_c = (current_start + current_end) / 2
                if abs(curr_c - prev_c) < max_jump_limit:
                    last_valid_start = current_start
                    last_valid_end = current_end

        # --- RYSOWANIE LINII POMIAROWYCH ---
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
