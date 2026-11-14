# Lokalny filtr antyphishingowy — pakiet rozwiązania

**TF-IDF znakowych 3–5-gramów + liniowy SVM**, 20 000 cech, CPU, bez API modeli.
W katalogu znajduje się gotowy model: trening nie jest wymagany do predykcji.
Można skopiować samo `src/` i uruchomić rozwiązanie offline po instalacji pakietów.

## Wynik dostarczonego modelu

| Kryterium | Wynik |
| --- | --- |
| Rozmiar gotowego modelu | **344 471 B / 0,329 MiB** (skompresowany joblib) |
| Czas normalizacji i predykcji | **0,921 ms/próbkę w batchu** |
| Czułość phishingu | **92,14%** |
| Precision / F1 | **92,81% / 0,925** |
| FPR — fałszywe alarmy wśród benign | **5,09%** |
| Odłożony test | **15 351 próbek**, w tym 6390 phishing |

Źródło liczb: [reports/test_metrics.json](reports/test_metrics.json). Czas to mediana
5 powtórzeń na stałej próbie 512 rekordów ze współdzielonego CPU, po rozgrzewce;
obejmuje normalizację, TF-IDF i SVM, bez odczytu plików i ładowania modelu.
Nie jest to opóźnienie pojedynczego żądania. Rozmiar dotyczy artefaktu, bez danych,
notebooka, bibliotek i Pythona. Te wartości nie są bezpośrednio porównywalne
z wcześniejszymi notatnikami na próbie 8000 rekordów i rozmiarem pickle bez kompresji.

## Pliki

```text
src/
  core.py                       wspólna normalizacja, inferencja, metryki
  data.py                       integralność danych, deduplikacja i podział
  train.py                      odtworzenie modelu i progu
  predict.py                    tekst / batch JSONL lub CSV → JSONL
  evaluate.py                   jakość, rozmiar i benchmark
  models/pipeline.joblib        wytrenowany TF-IDF, SVM, próg i konfiguracja
  models/metadata.json          wersje, SHA-256, podziały i wynik walidacji
  examples/test.jsonl           200 próbek z odłożonego testu
  reports/test_metrics.json     ocena pełnego odłożonego testu
  reports/sample_metrics.json   ocena dołączonych 200 próbek
  notebooks/training.ipynb      wykonany notebook, diagram i wyniki
  run_notebook.py               uruchomienie notebooka bez rejestracji kernela
  requirements.txt             zależności skryptów i modelu
  requirements-notebook.txt    dodatkowe zależności notebooka
  tests/test_solution.py        testy brzegowe i integracyjne
```

## Szybki start

Python **3.12**, Linux/macOS; w Windows użyj `.venv\Scripts\python.exe` zamiast
`.venv/bin/python`. Poniższe komendy wykonuj w katalogu `src/`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python predict.py --text "Meeting tomorrow at ten. Please bring the notes."
.venv/bin/python predict.py --input examples/test.jsonl --output /tmp/predictions.jsonl
.venv/bin/python evaluate.py --input examples/test.jsonl --output /tmp/sample-evaluation.json
```

Po instalacji nie potrzeba internetu, kluczy, GPU ani danych spoza `src/`.
Ładuj wyłącznie zaufane pliki joblib; dostarczony artefakt ma SHA-256 w metadanych.

### Format danych i wyjścia

JSONL: jeden obiekt na wiersz, pole `text` wymagane; `id` opcjonalne,
`label` wymagane tylko podczas oceny (`0` benign, `1` phishing):

```json
{"id": "example-1", "text": "Meeting tomorrow at ten.", "label": 0}
```

CSV: nagłówki `text,label` lub `id,text,label`; tekst z przecinkami/nowymi liniami
musi być poprawnie cytowany. `predict.py` ignoruje etykiety wejścia.

Wyjście to JSONL z `id`, `label`, `prediction` (`benign` lub `phishing`), `score`
i `threshold`. `score` jest **marginesem SVM**, nie prawdopodobieństwem;
phishing oznacza `score >= threshold`. Model ładuje się raz na proces.
`--batch-size 256` steruje porcjowaniem predykcji. Bez `--output` wynik trafia na stdout.
Puste/nienapisowe teksty i błędne etykiety powodują błąd; legalna wiadomość
bez słów-kluczy jest nadal klasyfikowana przez model, a nie odrzucana.

## Odtworzenie treningu i pełnego testu

Potrzebne są lokalne części oryginalnego datasetu i `manifest.json` w `../data/`
(opis źródła, etykiet i licencji: [data/README.md](../data/README.md)).
Nie są potrzebne do predykcji i oceny dołączonych próbek. Zalecane 8 GB RAM.
Pełny trening trwał około 107 s w tej maszynie, poza przygotowaniem danych i oceną.

```bash
# Osobny katalog chroni dostarczony model przed nadpisaniem.
.venv/bin/python train.py --data-dir ../data --output-dir experiments/full
.venv/bin/python evaluate.py --model experiments/full/models/pipeline.joblib \
  --data-dir ../data --output experiments/full/reports/rechecked-test.json

# Odtworzenie testu DOSTARCZONEGO modelu:
.venv/bin/python evaluate.py --data-dir ../data --output /tmp/full-test.json
```

Bez `--output-dir` trening zapisuje model i raporty w katalogu pakietu.
`--limit 8000` umożliwia mniejszy eksperyment (domyślnie `0` = całość).
`--max-features`, `--max-chars`, `--max-fpr` i `--seed` zmieniają konfigurację.
Zmiana tych parametrów oznacza nowy eksperyment i wymaga ponownej oceny.

Protokół dostarczonego modelu:

1. Weryfikacja SHA-256 wszystkich części oraz odtworzonego JSON źródłowego.
2. Pierwsze 4000 znaków, scalenie whitespace, lowercase i ponowny limit po lowercase.
   Usunięcie pustych wejść, identycznych znormalizowanych wejść i konfliktów etykiet.
3. Z 77 677 rekordów pozostaje **76 752**: train **46 051**, validation **15 350**,
   test **15 351** (warstwowo, 60/20/20, seed=42). Podziały są rozłączne po
   SHA-256 znormalizowanego tekstu; ich skróty zapisano w metadanych.
4. Słownik i IDF dopasowane tylko do train; SVM `C=1`, `class_weight=balanced`.
5. Próg maksymalizuje czułość przy **FPR ≤5% na walidacji**, z rozstrzyganiem
   remisów na rzecz niższego FPR. Walidacja: FPR **4,98%**, czułość **92,77%**.
6. Zapisany model oceniono na odłożonym teście, bez ponownego doboru progu.
   Limit walidacyjny **nie gwarantuje** identycznego FPR na teście lub nowych danych.

Dołączone 200 przykładów to warstwowa próbka z tego samego testu; nie jest nowym
niezależnym zbiorem. `evaluate.py --data-dir` odtwarza podział zapisany w modelu
i odrzuca niezgodny dataset lub podział.

## Notebook i testy

```bash
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python run_notebook.py
.venv/bin/python -m unittest discover -s tests -v
```

Notebook domyślnie używa gotowego modelu i ponownie ocenia 200 dołączonych próbek;
pełny wynik odczytuje z raportu treningu. Przełącznik `RETRAIN=True` uruchamia
rzeczywisty trening z pełnymi danymi do osobnego katalogu `experiments/`.
Testy obejmują Unicode/limit długości, remisy progów, budżet FPR walidacji,
JSONL/CSV, zgodność predykcji pojedynczych i batchowych oraz dołączony artefakt.
Weryfikacja dostarczonego pakietu: 10 testów zakończonych poprawnie, wykonany
notebook i odtworzony pełny test z identyczną macierzą pomyłek. Sam katalog
`src/` sprawdzono także po skopiowaniu poza repozytorium, w czystym venv
z zależnościami z `requirements.txt` i zablokowanymi połączeniami sieciowymi.

## Ograniczenia

Zbiór miesza URL, SMS, e-maile i HTML; wynik nie opisuje wyłącznie poczty.
Limit długości usuwa końcówki długich wiadomości. Usunięto identyczne wejścia,
ale bliskie duplikaty, domeny i kampanie mogą występować w różnych podziałach.
Wybór architektury opiera się na wcześniejszych eksperymentach na części tego
datasetu, więc test wewnętrzny nie zastępuje nowego, niezależnego zbioru.
Brak dat i identyfikatorów kampanii nie pozwala potwierdzić odporności na Zero-Day.
Model nie wykonuje ani nie renderuje zawartości próbek i nie odwiedza adresów.
Przy FPR około 5% nadal wymagana jest ostrożna polityka blokowania wiadomości.
