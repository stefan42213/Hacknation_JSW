import cv2
import numpy as np
import time

class SectionDetector:
    # 1. ZMIANA: Dodajemy argument 'plik_zapisu', żeby klasa wiedziała, gdzie dopisać separator
    def __init__(self, plik_zapisu="pomiary.txt"):
        self.last_detection_time = 0
        self.cooldown = 3.0       # Sekundy przerwy po wykryciu
        self.threshold = 40.0     # Czułość
        self.plik_zapisu = plik_zapisu  # Zapamiętujemy nazwę pliku

    def check_for_section(self, frame, start_x, end_x, scan_y):
        """
        Sprawdza czy w danym miejscu jest poziome łączenie.
        Jeśli tak -> AUTOMATYCZNIE dopisuje '---section---' do pliku.
        """
        current_time = time.time()
        
        # Jeśli jesteśmy w trakcie "odpoczynku" po ostatnim wykryciu -> ignoruj
        if (current_time - self.last_detection_time) < self.cooldown:
            return False, 0.0

        # Wycinek obrazu (tylko w miejscu pomiaru)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_y_start = max(0, scan_y - 5)
        roi_y_end = min(gray.shape[0], scan_y + 5)
        
        # Pobieramy pasek taśmy
        belt_roi = gray[roi_y_start:roi_y_end, start_x:end_x]
        
        if belt_roi.size == 0:
            return False, 0.0

        # Wykrywanie poziomych linii (Sobel Y)
        sobel_y = cv2.Sobel(belt_roi, cv2.CV_64F, 0, 1, ksize=3)
        edge_score = np.mean(np.abs(sobel_y))

        if edge_score > self.threshold:
            self.last_detection_time = current_time
            
            # 2. ZMIANA: Wywołujemy funkcję zapisu
            self._zapisz_separator()
            
            print(f"[WYKRYCIE] Znaleziono sekcję! (Siła: {edge_score:.2f})")
            return True, edge_score
            
        return False, edge_score

    # 3. ZMIANA: Nowa funkcja pomocnicza do zapisu
    def _zapisz_separator(self):
        try:
            with open(self.plik_zapisu, "a") as f:
                f.write("---section---\n")
        except Exception as e:
            print(f"Błąd zapisu sekcji do pliku: {e}")