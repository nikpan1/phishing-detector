# phishing-detector

Trzy uruchomione notatniki przedstawiają lokalne metody detekcji phishingu
na zbiorze `ealvaradob/phishing-dataset`, wariant `combined_reduced`:

1. [Reguły i metryki](notebooks/01_reguly_i_metryki.ipynb): regex,
   Levenshtein, Jaro–Winkler i Jaccard.
2. [Clustering i unsupervised](notebooks/02_clustering_i_unsupervised.ipynb):
   DBSCAN, HDBSCAN, K-Means na TF-IDF i embeddingach LSA, Isolation Forest,
   symulacja nowej kampanii.
3. [Statystyczny ML](notebooks/03_statystyczny_ml.ipynb): TF-IDF słowny
   i znakowy, n-gramy, SVM, Naive Bayes i Random Forest (9 konfiguracji).

Każdy kończy się wspólną tabelą porównawczą swoich metod i omówieniem ograniczeń.
Wyniki wykonania są zapisane w notatnikach oraz w [results](results/).
Notatniki zawierają również diagramy przepływu, wykresy porównawcze i mapy
wyników zapisane jako obrazy wewnątrz plików — są widoczne także na GitHubie.

[Zwięzły raport porównawczy z wizualizacjami](docs/raport.md) podsumowuje
wszystkie przeanalizowane metody, ich wyniki i ograniczenia.

## Uruchomienie

Python 3.12, CPU, zalecane 8 GB RAM. Nie potrzeba GPU, kluczy API,
zdalnych modeli ani serwisów. Sieć jest potrzebna tylko do instalacji pakietów.
Pełny dataset jest w repozytorium: [opis i sumy kontrolne](data/README.md).

Z katalogu głównego repozytorium:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_notebooks.py
```

Skrypt uruchamia wszystkie komórki w tym samym interpreterze przez tymczasowy
kernel Jupyter. Nie wymaga instalowania kernela globalnie. Zatrzymuje się przy
błędzie i zapisuje wyniki tylko poprawnie ukończonego notatnika. Można wybrać jeden:

```bash
.venv/bin/python scripts/run_notebooks.py --notebook 02
```

Do edycji interaktywnej wybierz interpreter `.venv` w VS Code/Jupyter.
Opcjonalnie zainstaluj JupyterLab w tym środowisku. Otwórz notatnik z katalogu
`notebooks` albo z katalogu głównego repozytorium i wykonaj komórki od góry.
Wszystkie obliczenia działają offline po instalacji zależności.

## Jak interpretować wyniki

Pełne dane: 77 677 rekordów, `0` benign / `1` phishing, mieszanina e-maili,
SMS, URL i HTML. Eksperymenty korzystają ze stałej próby 8000 rekordów po
deduplikacji pierwszych 4000 znaków, podzielonej warstwowo 60/20/20 z seed=42.
`notebooks/common.py` sprawdza integralność całego zbioru i rozłączność podziałów.
Zmienna `limit` w `load_splits` pozwala zwiększyć próbę; `None` wybiera całość,
co znacznie zwiększy koszt DBSCAN/HDBSCAN i metryk odległości.

Klasyfikatory są porównywane przez czułość phishingu, precision, F1, FPR,
ROC-AUC, average precision, czas i rozmiar. Progi wybierane są na validation,
a ocena końcowa na test. Czas oznacza medianę ms/próbkę w batchu (3 powtórzenia,
po rozgrzewce), z transformacją tekstu i bez odczytu danych. `model_mib` to
nieskompresowany pickle parametrów predykcji, bez runtime'u i całego datasetu.
Nie zapisujemy plików modeli; tabele przedstawiają oszacowanie ich rozmiaru.
Eksperymenty mają ustalone hiperparametry, nie są pełnym tuningiem ani CV.

Klastry nie są klasyfikatorami: mają osobne metryki ARI/NMI/silhouette oraz
udział szumu. HDBSCAN oceniany jest batchowo, pozostałe czasy predykcji zgodnie
z opisem w tabeli. Isolation Forest oceniamy jako detektor anomalii, a nie
automatyczne rozpoznanie phishingu. Embeddingi LSA uczą się lokalnie z TF-IDF.
Symulacja kampanii nie dowodzi odporności na rzeczywisty Zero-Day. Brak dat
i identyfikatorów kampanii uniemożliwia wiarygodną walidację czasową; bliskie
duplikaty mogą zawyżać wyniki mimo usunięcia identycznych wejść.

Nie odwiedzaj linków ani nie renderuj HTML zawartego w danych. Notatniki
analizują tekst, wyświetlają jedynie agregaty i nie wykonują zawartości próbek.

Dokumentacja zadania: [docs/zadanie/zadanie.md](docs/zadanie/zadanie.md).
