# Dane

Pełny `combined_reduced.json` wskazany w `docs/zadanie/phishing_detector.ipynb`.
Źródło: https://huggingface.co/datasets/ealvaradob/phishing-dataset

Plik źródłowy przekracza limit 100 MiB na plik GitHuba, dlatego jest zapisany
bezstratnie jako strumień gzip podzielony na kolejne części po maksymalnie 40 MiB.
Nie usunięto żadnych rekordów. `manifest.json` zawiera sumy SHA-256 oryginału
i części, liczby rekordów i etykiet oraz rewizję źródła odczytaną przy pobraniu.

Odtworzenie oryginału z katalogu głównego repozytorium (około 498 MiB):

```bash
cat data/combined_reduced.json.gz.part* | gzip -d > /tmp/combined_reduced.json
sha256sum /tmp/combined_reduced.json
```

Notatniki wczytują części automatycznie i weryfikują sumy kontrolne.
`0` oznacza benign, `1` phishing. Dane obejmują URL, SMS, e-maile i HTML;
wyników nie należy interpretować jako benchmarku samych e-maili.
Karta źródła deklaruje Apache-2.0; jej kopia znajduje się w `SOURCE_README.md`.
Nie odwiedzaj adresów ani nie renderuj HTML z próbek. Notatniki operują wyłącznie na tekście.
