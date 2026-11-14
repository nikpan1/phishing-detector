# Dołączone próbki testowe

`test.jsonl` zawiera 200 rekordów: 83 phishing i 117 benign. Jest to warstwowa
próbka (seed=42) z odłożonego testu dostarczonego modelu, bez udziału w treningu
lub doborze jego progu. Teksty są już znormalizowane i ograniczone do 4000 znaków;
pole `id` jest SHA-256 tego wejścia. `predict.py` normalizuje również surowe wejścia.

Źródło: [ealvaradob/phishing-dataset](https://huggingface.co/datasets/ealvaradob/phishing-dataset),
plik `combined_reduced.json`, rewizja `55caf9af01ae498d68ff44886ff69a4b165eb54b`.
Karta źródła deklaruje licencję Apache-2.0 i wskazuje autora Estebana Alvarado (2024).
Oryginalny plik ma SHA-256
`222fd5f841a0565841f8ede2e21b9b48cfea933a6b5c3a7da3e3a2cbc156d3d5`.

Etykiety: `0` = benign, `1` = phishing. Zbiór miesza URL, SMS, e-maile i HTML.
Nie otwieraj adresów ani nie renderuj HTML z rekordów. Pakiet operuje na tekście.
Wynik na tych 200 próbkach jest w `../reports/sample_metrics.json`; przy mniejszej
liczności różni się od wyniku pełnego testu i nie stanowi niezależnej walidacji.
