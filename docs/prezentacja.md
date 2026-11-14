# Deep learning: BERT

- **Akwizycja kontekstu:** mechanizm self-attention analizuje całe zdania dwukierunkowo.
- **Detekcja subtelnej manipulacji:** rozpoznaje inżynierię społeczną bez oczywistych słów-kluczy.
- **Fine-tuning:** pretrenowane modele dostrajane do korpusów maili.

## Phishing generatywny

**70%+** klasycznych filtrów spamowych jest omijanych przez wiadomości wygenerowane przez LLM z poprawną gramatyką.

## Nowe architektury

### Model hybrydowy ML + LLM

Tani ML filtruje prosty spam, LLM weryfikuje tylko niepewne przypadki.

### Stylometria tekstu

Weryfikacja unikalnego profilu pisarskiego nadawcy na bazie cech składni.

### Wyjaśnialne AI (XAI / LIME)

Wyjaśnienia w języku naturalnym, dlaczego mail uznano za phishing.

### Grafowe sieci neuronowe (GNN)

Mapowanie relacji domen, serwerów DNS i odbiorców w postaci grafu.

## Pipeline kaskadowy

1. **Krok 1 — Szybki filtr ML / clustering:** skanowanie TF-IDF / DBSCAN w czasie <10 ms.
2. **Krok 2 — Progowanie pewności:** pewne klasyfikacje są zatwierdzane od razu.
3. **Krok 3 — Eskalacja do LLM:** podejrzane wiadomości analizuje model LLM.

## Porównanie metod

| Metoda | Opóźnienie | Odporność na Zero-Day | Główne zastosowanie |
| --- | --- | --- | --- |
| Regex / Levenshtein | <1 ms | Bardzo niska | Szybka filtracja wstępna |
| Clustering (DBSCAN/HDBSCAN) | ~20–50 ms | Wysoka | Wykrywanie masowych kampanii |
| TF-IDF + SVM | ~5 ms | Średnia | Masowy odsiew spamu |
| BERT (Deep Learning) | ~100 ms | Wysoka | Analiza kontekstowa maila |
| Hybryda ML + LLM | Zmienne | Bardzo wysoka | Enterprise Security System |

## Wnioski

1. **Clustering wykrywa przed etykietami:** metody unsupervised są niezastąpione w pierwszych minutach zmasowanych ataków Zero-Day.
2. **Konieczna weryfikacja semantyczna:** GenAI wymusza przejście od analizy pojedynczych słów do badania intencji i kontekstu.
3. **Kaskada optymalizuje koszt:** łączenie filtrów statystycznych i clusteringu z LLM obniża opóźnienia i koszty obliczeń.
