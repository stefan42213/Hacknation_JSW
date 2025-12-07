import cv2
import time
import os
from datetime import datetime

# Importujemy nasze moduły (muszą być w tym samym folderze)
from size import TapeMeasurer
from sections import SectionDetector

# --- KONFIGURACJA ---
VIDEO_SOURCE = 'videos/test2.mkv'
DATA_FILE = "data/pomiary.txt" # Plik z surowymi danymi (czas + pomiar)
WARNING_FILE = "data/warning.txt"
STATS_FILE = "data/max_min_values.txt" # Nowy plik do statystyk

MEASUREMENT_LOG_INTERVAL = 0.5  # Co ile sekund zapisywać szerokość do pliku pomiary
TIME_TO_WAIT_FOR_STOP = 3.0      # Alarm braku ruchu

def log_data(filename, text):
    """Uniwersalna funkcja zapisu"""
    try:
        with open(filename, "a") as f:
            f.write(text)
    except Exception as e:
        print(f"Błąd zapisu do {filename}: {e}")

def save_section_stats(section_id, measurements):
    """Oblicza min/max i zapisuje do pliku max_min_values.txt"""
    if not measurements:
        return # Brak danych, nic nie zapisujemy

    min_val = min(measurements)
    max_val = max(measurements)
    avg_val = sum(measurements) / len(measurements)
    
    ts_str = datetime.now().strftime("%H:%M:%S")
    
    # Zmieniony format statystyk: z czasem i średnią
    line = (f"[{ts_str}] Sekcja {section_id} -> "
            f"MIN: {min_val:.2f} | MAX: {max_val:.2f} | Średnia: {avg_val:.2f}\n")
    
    log_data(STATS_FILE, line)
    print(f"--- ZAPISANO STATYSTYKI: {line.strip()} ---")

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    measurer = TapeMeasurer()
    section_detector = SectionDetector()

    # Zmienne stanu
    prev_gray_frame = None
    last_movement_time = time.time()
    last_measurement_log_time = 0
    is_stopped_flag = False

    # --- NOWE ZMIENNE DO STATYSTYK ---
    current_section_widths = [] # Tu zbieramy pomiary z obecnej sekcji
    section_counter = 1         # Numeracja sekcji

    # Tworzymy/Czyścimy plik statystyk na starcie
    with open(STATS_FILE, "w") as f:
        f.write(f"--- Start analizy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    print(f"Start systemu.")
    print(f"Logi: {DATA_FILE}")
    print(f"Statystyki: {STATS_FILE}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        
        # 1. Obsługa alarmu STOP (Detekcja ruchu)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        motion_detected = False
        if prev_gray_frame is not None:
            delta = cv2.absdiff(prev_gray_frame, gray_blur)
            thresh = cv2.threshold(delta, 2, 255, cv2.THRESH_BINARY)[1]
            if cv2.countNonZero(thresh) / (frame.shape[0]*frame.shape[1]) > 0.001:
                motion_detected = True
                last_movement_time = current_time
                is_stopped_flag = False
        prev_gray_frame = gray_blur

        if current_time - last_movement_time > TIME_TO_WAIT_FOR_STOP:
            if not is_stopped_flag:
                # ZMIANA: Ujednolicony format timestamp
                ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_data(WARNING_FILE, f"[{ts_str}] OSTRZEZENIE: Taśma produkcyjna zatrzymana!\n")
                print("ALARM: STOP TAŚMY")
                is_stopped_flag = True
            cv2.putText(frame, "ALARM: STOP!", (50, frame.shape[0]//2), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # 2. Pomiar szerokości
        mask = measurer.get_mask(frame)
        width_px, start_x, end_x, scan_y = measurer.measure(frame, mask)
        
        draw_frame = frame.copy()
        cv2.line(draw_frame, (0, scan_y), (frame.shape[1], scan_y), (100, 100, 100), 1)

        section_found_now = False

        if width_px is not None:
            # 3. Sprawdzamy czy to NOWA SEKCJA
            is_section, score = section_detector.check_for_section(frame, start_x, end_x, scan_y)
            
            if is_section:
                # A) Zapisujemy statystyki POPRZEDNIEJ sekcji
                save_section_stats(section_counter, current_section_widths)
                
                # B) Zapisujemy separator ---section--- do pliku pomiary.txt
                log_data(DATA_FILE, "---section---\n")
                
                # C) Resetujemy pod nową sekcję
                section_counter += 1
                current_section_widths = [] 
                
                print(f"!!! WYKRYTO NOWĄ SEKCJĘ (nr {section_counter}) !!!")
                section_found_now = True # Używamy tego do pomijania logowania w tym samym cyklu

            # Wizualizacja
            cv2.arrowedLine(draw_frame, (start_x, scan_y), (end_x, scan_y), (0, 0, 255), 3)
            cv2.arrowedLine(draw_frame, (end_x, scan_y), (start_x, scan_y), (0, 0, 255), 3)
            cv2.putText(draw_frame, f"W: {width_px}", (start_x, scan_y - 10), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 4. Zbieranie danych (jeśli nie ma właśnie łączenia)
            if not section_found_now:
                # Dodajemy KAŻDY poprawny pomiar do listy statystyk sekcji
                current_section_widths.append(width_px)

                # Zapis do pliku pomiary.txt co interwał (żeby nie zapchać dysku)
                if current_time - last_measurement_log_time > MEASUREMENT_LOG_INTERVAL:
                    
                    # ZMIANA: Uproszczony i jednolity format TIMESTAMP dla pomiarów
                    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    log_data(DATA_FILE, f"{ts_str};{width_px}\n")
                    last_measurement_log_time = current_time
        
        else:
            cv2.putText(draw_frame, "Szukam krawedzi...", (50, scan_y - 15), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Informacja na ekranie (wizualna)
        if current_time - section_detector.last_detection_time < 1.0:
             cv2.putText(draw_frame, f"SEKCJA {section_counter} START!", (frame.shape[1]//2 - 150, scan_y - 50), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

        # Podgląd
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = cv2.hconcat([draw_frame, mask_bgr])
        final = cv2.resize(combined, (0,0), fx=0.5, fy=0.5)
        
        cv2.imshow('System', final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # PO ZAKOŃCZENIU PĘTLI (np. klawisz 'q')
    # Zapisz statystyki ostatniej, niedokończonej sekcji
    if current_section_widths:
        print("Zapisywanie ostatniej sekcji przy wyjściu...")
        save_section_stats(section_counter, current_section_widths)
        # Zamykamy ostatnią sekcję separatorem
        log_data(DATA_FILE, "---section---\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()