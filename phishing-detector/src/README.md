# Filtr antyphishingowy — pakiet rozwiązania

Gotowy model działa lokalnie. Nie wymaga treningu, internetu podczas predykcji ani danych spoza tego katalogu.

## Jak uruchomić

Python 3.12. Komendy z katalogu `phishing-detector/src/`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python predict.py --text "Meeting tomorrow at ten."
.venv/bin/python predict.py --input /sciezka/dane.csv --output /tmp/predictions.jsonl
.venv/bin/python evaluate.py --input examples/test.jsonl
.venv/bin/python train.py --input /sciezka/trening.jsonl --output-dir /tmp/moj-model
```

CSV ma nagłówki `text,label` (`label` tylko do treningu/oceny); JSON to lista obiektów, JSONL — jeden obiekt na wiersz. Wszystkie formaty obsługują `.gz`, UTF-8 i opcjonalne `id`. Etykiety: `0` benign, `1` phishing. Wymagane są niepuste teksty; trening potrzebuje co najmniej 5 różnych wejść każdej klasy po deduplikacji.

Przykład własnych nazw pól i etykiet:

```bash
.venv/bin/python train.py --input /sciezka/dane.csv --text-column message --label-column category \
  --positive-label phishing --negative-label benign --delimiter ';' --output-dir /tmp/moj-model
.venv/bin/python predict.py --model /tmp/moj-model/models/pipeline.joblib --input /sciezka/nowe.jsonl
```

Wynik predykcji: JSONL z `id`, `label`, `prediction`, `score` i `threshold`. Bez `--output` trafia na stdout. `--batch-size` określa wielkość partii. `evaluate.py` przyjmuje te same opcje nazw kolumn i etykiet; `--model` wskazuje własny model. `score` to margines SVM, nie prawdopodobieństwo. Ładuj wyłącznie zaufane pliki joblib.

Notebook i testy:

```bash
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python run_notebook.py
.venv/bin/python -m unittest discover -s tests -v
```

Notebook domyślnie używa gotowego modelu i 200 dołączonych próbek. Do ponownego treningu ustaw w nim `DATASET`, opcjonalnie `DATA_OPTIONS`, i `RETRAIN=True`. Pełny dataset nie jest dołączony.

## Architektura

```mermaid
flowchart LR
    T["Tekst / URL"] --> N["Normalizacja + limit 4000 znaków"]
    N --> F["TF-IDF: znaki 3–5 + słowa 1–2"]
    F --> S["15 000 cech → liniowy SVM"]
    S --> P["Próg → benign / phishing"]
```

Trening usuwa identyczne wejścia i konflikty etykiet; dzieli wskazane dane 60/20/20 (seed=42). Reprezentacje uczone są wyłącznie na train; próg minimalizuje FPR przy walidacyjnej czułości ≥92,8%. `--min-recall` zmienia to minimum; alternatywne `--max-fpr` ustala limit walidacyjnego FPR. Test nie wybiera progu. Gotowy model ma SVM `C=0.25` i 10k cech znakowych + 5k słownych po selekcji z bazowych 90k cech.

## Working tree

```text
models/                  # pipeline.joblib i metadata.json
notebooks/training.ipynb # notebook z wynikami i diagramami
examples/test.jsonl      # 200 rekordów z odłożonego testu
reports/                 # metryki i porównanie przed/po
predict.py, evaluate.py  # predykcja i ocena własnych plików
train.py, optimize.py    # trening i wybór modeli na walidacji
compression.py, compress_model.py, search_compression.py  # kompresja i jej wybór
core.py, data.py, modeling.py, tests/, requirements*.txt, run_notebook.py
```

## Wyniki po kompresji

Porównanie na tych samych **15 351 próbkach testowych**, po wyborze wariantu wyłącznie na walidacji:

| Miara | Przed | Po |
| --- | --- | --- |
| Rozmiar modelu | 0,681 MiB | **0,073 MiB** |
| Czułość | 92,49% | **92,63%** |
| FPR | 2,96% | **3,14%** |
| Precision | 95,71% | 95,47% |
| F1 | 0,9407 | 0,9403 |
| Fałszywe alarmy / przeoczony phishing | 265 / 480 | **281 / 471** |
| Predykcja [ms/próbkę] | 0,892 | **0,788** |
| Ładowanie [ms] | 127,3 | **13,5** |

Model zmniejszono o **89,29% (9,33×)** względem poprzednich 0,681 MiB; ma dokładnie **76 518 bajtów**. To rozmiar pliku modelu, nie całego repozytorium ani zależności Pythona.

Czułość wzrosła o 0,14 punktu procentowego, FPR wzrósł o 0,18 punktu procentowego. W całym teście oznacza to 9 mniej przeoczonych ataków i 16 dodatkowych fałszywych alarmów. Zmieniły się 193 klasyfikacje. Czas to mediana pięciu pomiarów po rozgrzewce na batchu 512 próbek; oba modele zmierzono ponownie na tym samym CPU. Pomiar predykcji nie obejmuje ładowania modelu/danych, a wyniki czasowe zależą od obciążenia.

## Jak uzyskano mniejszy model

1. Z bazowych 60 tys. cech znakowych i 30 tys. słownych zachowano **10 tys. znakowych i 5 tys. słownych** według bezwzględnych wag SVM wyuczonych na train.
2. Ponownie nauczono SVM na train po zmianie normalizacji TF-IDF. IDF i słowniki nie korzystają z walidacji ani testu.
3. 99,9% wag zaokrąglono do wspólnej siatki **6-bitowej**; 15 największych wag zachowano dokładnie. Podczas predykcji wagi nadal są float32, bez dodatkowego dekodera lub własnych klas modeli.
4. Słowniki zapisano jako uporządkowane sekwencje, bez redundantnych indeksów, i zastosowano LZMA (6). Plik działa z joblib oraz standardowym scikit-learn.

Porównano 20 wariantów. Limity walidacyjne ustalono przed wyborem: spadek czułości ≤0,5 p.p., wzrost FPR ≤0,3 p.p. Wybrano najmniejszy dopuszczalny wariant; następnie na walidacji ustawiono próg z zachowaniem pierwotnego minimum czułości **92,8%**. Finalna walidacja: czułość 92,80%, FPR 3,06%. Test wykonano po zamrożeniu konfiguracji.

To ten sam, wcześniej używany losowy test. Wyniki nie zastępują niezależnej walidacji czasowej lub na nowych kampaniach.

[Raport końcowy](reports/test_metrics.json) · [Porównanie kompresji](reports/compression_comparison.json) · [20 wariantów walidacyjnych](reports/compression_candidates.json).

Historyczne etapy: model 1,525 MiB → 0,681 MiB (sam zapis float32), następnie obecne 0,073 MiB (selekcja cech i kwantyzacja). Raporty `optimization.*` i `size_optimization.json` opisują wcześniejsze wersje.

## Odtworzenie kompresji

Domyślne `train.py` odtwarza pełną sekwencję: trening bazowy, selekcja 10k/5k, ponowny trening SVM, kwantyzacja 6-bitowa i wybór progu na walidacji. Opcje `--char-keep`, `--word-keep`, `--weight-bits` pozwalają ją zmienić. `--char-keep 0 --word-keep 0 --weight-bits 32` wyłącza redukcję cech i kwantyzację.

Aby odtworzyć porównanie 20 wariantów, użyj **poprzedniego modelu 90k**, nie obecnego już skompresowanego modelu. Z katalogu repozytorium można go odzyskać przez `git show fcfec16:phishing-detector/src/models/pipeline.joblib > /tmp/reference.joblib`.

Następnie z tego katalogu:

```bash
python search_compression.py --model /tmp/reference.joblib --input /sciezka/dane.json --output-dir /tmp/compression-search
python compress_model.py --model /tmp/reference.joblib --input /sciezka/dane.json --output-dir /tmp/compressed
```

Pierwszy skrypt korzysta tylko z train/validation. Drugi stosuje zamrożony wariant 10k/5k/6-bit, dobiera próg przy minimum recall 92,8% na walidacji i dopiero wtedy ocenia test. Oba wymagają identycznego datasetu i podziału co model bazowy; sprawdzają ich sumy SHA-256. `search_compression.py` przyspiesza porównania przez wybór i ponowną normalizację kolumn macierzy bazowej; wybrany wariant jest odtwarzany z rzeczywistą transformacją TF-IDF przed końcową kalibracją. Niewielkie różnice zaokrągleń i rozmiaru pliku między etapami są oczekiwane.

Pełny trening od początku odtworzył identyczne słowniki, IDF, wagi, próg i macierz pomyłek. [Potwierdzenie odtwarzalności](reports/reproducibility.json).
