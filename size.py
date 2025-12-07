import cv2
import numpy as np

class TapeMeasurer:
    def __init__(self):
        # Zmienne do stabilizacji pomiaru (pamięć między klatkami)
        self.last_valid_start = None
        self.last_valid_end = None
        self.max_jump_limit = 30  # Maksymalny skok w pikselach między klatkami
        
        # Ustawienia algorytmu
        self.min_area = 80
        self.min_dimension = 60
        self.scan_ratio = 0.85    # W którym miejscu (wysokość) mierzymy (0.85 = 85%)

    def get_mask(self, frame):
        """Tworzy czarno-białą maskę taśmy"""
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

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            if area >= self.min_area and (width > self.min_dimension or height > self.min_dimension):
                final_mask[labels == i] = 255
        
        return final_mask

    def measure(self, frame, mask):
        """Zwraca: (szerokość, start_x, end_x, y_scan) lub (None, None, None, y_scan)"""
        height, width = frame.shape[:2]
        scan_y = int(height * self.scan_ratio)
        
        row_pixels = mask[scan_y, :]
        white_indices = np.where(row_pixels > 0)[0]

        current_start = None
        current_end = None
        min_neighbors = 7
        search_window = 15

        if len(white_indices) > min_neighbors:
            # Szukamy lewej krawędzi
            for i in range(len(white_indices) - min_neighbors):
                if white_indices[i + min_neighbors] - white_indices[i] < search_window:
                    current_start = white_indices[i]
                    break 
            # Szukamy prawej krawędzi
            for i in range(len(white_indices) - 1, min_neighbors, -1):
                if white_indices[i] - white_indices[i - min_neighbors] < search_window:
                    current_end = white_indices[i]
                    break

        # Stabilizacja wyniku (filtracja błędnych skoków)
        if current_start is not None and current_end is not None and current_end > current_start:
            if self.last_valid_start is None:
                self.last_valid_start = current_start
                self.last_valid_end = current_end
            else:
                prev_c = (self.last_valid_start + self.last_valid_end) / 2
                curr_c = (current_start + current_end) / 2
                if abs(curr_c - prev_c) < self.max_jump_limit:
                    self.last_valid_start = current_start
                    self.last_valid_end = current_end
        
        if self.last_valid_start is not None and self.last_valid_end is not None:
            width_px = self.last_valid_end - self.last_valid_start
            return width_px, self.last_valid_start, self.last_valid_end, scan_y
        
        return None, None, None, scan_y