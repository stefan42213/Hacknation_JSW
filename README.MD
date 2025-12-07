

-----

# VisionBeltMonitor: System Monitorowania i Pomiaru Taśmy Produkcyjnej

VisionBeltMonitor to system wizyjny oparty na **OpenCV**, zaprojektowany do ciągłego monitorowania, pomiaru szerokości elementów na taśmie produkcyjnej oraz generowania alarmów w przypadku przestojów.

-----

## Wprowadzenie i Kluczowe Funkcjonalności

System automatycznie segmentuje strumień wideo na logiczne sekcje produkcyjne, mierzy każdy element i generuje szczegółowe statystyki oraz ostrzeżenia w czasie rzeczywistym.

  * **Precyzyjny Pomiar Szerokości:** Wykorzystuje zaawansowane przetwarzanie obrazu (`size.py`) do stabilnego określania szerokości elementu w pikselach.
  * **Segmentacja Sekcji:** Automatycznie wykrywa przejścia między mierzonymi elementami/blokami (`sections.py`).
  * **Logowanie Statystyk:** Dla każdej wykrytej sekcji oblicza i zapisuje wartości **Min**, **Max** i **Średnią** szerokość.
  * **Alarm STOP:** Monitoruje ruch i generuje alarm w przypadku dłuższego braku aktywności na taśmie.

-----

## Struktura Projektu i Instalacja

### 1\. Struktura Folderów

Projekt wymaga następującej struktury katalogów do poprawnego działania:

```
/VisionBeltMonitor
├── main.py         # Logika główna, detekcja ruchu, I/O
├── size.py         # Klasa TapeMeasurer (Pomiar szerokości)
├── sections.py     # Klasa SectionDetector (Detekcja łączeń)
├── /video          # Źródła wideo
│     └── test2.mkv
└── /data           # Katalog docelowy dla wszystkich plików wynikowych
         └── pomiary.txt
         └── warning.txt
         └── max_min_values.txt
```

### 2\. Wymagania

Wymagane biblioteki Python:

```bash
pip install opencv-python numpy
```

-----

## Konfiguracja (Plik `main.py`)

Zmienne konfiguracyjne określają ścieżki do plików, interwały logowania i czułość alarmów:

| Zmienna | Opis | Wartość |
| :--- | :--- | :--- |
| `VIDEO_SOURCE` | Ścieżka do pliku wejściowego. | `'video/test2.mkv'` |
| `DATA_DIR` | Folder wyjściowy dla wszystkich logów. | `"data"` |
| `MEASUREMENT_LOG_INTERVAL` | Interwał zapisu surowego pomiaru do `pomiary.txt` (w sekundach). | `0.5` |
| `TIME_TO_WAIT_FOR_STOP` | Czas (w sekundach) braku ruchu, po którym uruchamiany jest alarm STOP. | `3.0` |

-----

## Opis Modułów

### `main.py` (Kierownik Zmiany)

Odpowiada za inicjalizację, pętlę wideo, detekcję ruchu i zarządzanie logami.

  * **Inicjalizacja:** Uruchamia moduły `TapeMeasurer` i `SectionDetector`. **Automatycznie tworzy folder `data/`**, jeśli nie istnieje.
  * **Logowanie:** Wykorzystuje funkcje `log_data` i `save_section_stats` do zapisu wyników do folderu `data/`.
  * **Alarm STOP:** Porównuje klatki w skali szarości, aby wykryć przestój taśmy.

### `size.py` (Klasa `TapeMeasurer`)

Odpowiada za dokładny pomiar. Implementuje:

1.  **Maskowanie:** Przetwarzanie obrazu na binarną maskę (`get_mask()`).
2.  **Liniowy Skan:** Wykrywanie krawędzi taśmy na stałej wysokości.
3.  **Stabilizacja:** Wygładzanie skoków pomiarowych, aby zapewnić stabilny odczyt szerokości w czasie.

### `sections.py` (Klasa `SectionDetector`)

Odpowiada za segmentację produktu.

  * **Detekcja łączeń:** Analizuje lokalne gradienty (Sobel Y) w celu wykrycia ostrych, poziomych krawędzi symbolizujących koniec sekcji.
  * **Cooldown:** Zapobiega wielokrotnemu wykrywaniu tego samego łączenia, stabilizując proces segmentacji.

-----

## Format Plików Wynikowych (Wszystkie w `data/`)

### 1\. `pomiary.txt` (Surowe dane pomiarowe)

```
[2025-12-07 09:17:01];456
[2025-12-07 09:17:02];458
---section---  <-- Separator po wykryciu nowego bloku
[2025-12-07 09:17:03];312
```

### 2\. `max_min_values.txt` (Statystyki sekcji)

```
[09:17:03] Sekcja 1 -> MIN: 456.00 | MAX: 460.00 | Średnia: 457.92
[09:17:05] Sekcja 2 -> MIN: 310.00 | MAX: 318.00 | Średnia: 314.55
```

### 3\. `warning.txt` (Alarmy systemowe)

```
[2025-12-07 09:20:30] OSTRZEZENIE: Taśma produkcyjna zatrzymana!
```

-----

## Uruchomienie

Uruchom główny skrypt z katalogu głównego projektu:

```bash
python main.py
```


Wyjście z programu następuje po wciśnięciu klawisza **`q`** w oknie podglądu. Po zakończeniu, system zapisze statystyki ostatniej, trwającej sekcji.
