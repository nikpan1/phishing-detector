# Wyniki dołączonego wykonania

- `01_reguly_i_metryki.csv`: wszystkie 4 metody regułowe/wzorcowe.
- `02_clustering_i_unsupervised.csv`: 4 warianty clusteringu i Isolation Forest.
- `02_symulacja_kampanii.csv`: agregaty kontrolowanej symulacji kampanii.
- `03_statystyczny_ml.csv`: 9 kombinacji reprezentacji i klasyfikatora.

CSV są generowane przez notatniki i odpowiadają ich dołączonym wynikom.
Wykonano wszystkie komórki na CPU, Python 3.12, z wersjami głównych pakietów
z `requirements.txt`, limitem 2 wątków BLAS/OpenMP; Random Forest używa 2 zadań.
Czasy pochodzą ze współdzielonego środowiska cloud, bez izolacji obciążenia,
więc są poglądowe i mogą się zmieniać przy kolejnym wykonaniu.

Wspólna próbka: 8000 rekordów po deduplikacji, train/validation/test 4800/1600/1600.
Podział i sumy SHA-256 jego kluczy są wypisane w każdym notatniku.
Miary klasyfikacji dotyczą klasy `1` (phishing). `ms_per_sample` jest medianą
czasu batcha podzielonego przez jego liczność, a nie percentylem opóźnienia
pojedynczych wiadomości. `model_mib` to rozmiar pickle w MiB, bez runtime'u.

Puste pola w tabeli clusteringu oznaczają **nie dotyczy**, nie zero.
ARI/NMI/silhouette opisują grupy train, a F1/recall/FPR Isolation Forest dotyczą
testu. `timing_mode` rozróżnia predykcję i batchowe ponowne dopasowanie HDBSCAN.
Nie należy rangować wszystkich wierszy tej tabeli jednym wskaźnikiem jakości.

Dołączone wykonanie pokazuje także porażki: próg regex wybrany pod F1 oznaczył
wszystkie próbki jako phishing; symulowana kampania nie utworzyła osobnej
gęstej grupy bez punktów train. Notatniki omawiają te ograniczenia.
