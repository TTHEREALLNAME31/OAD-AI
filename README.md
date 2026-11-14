# Od wersji A do B – prompty dla AI (analizator postaci RPG, NumPy)

| Plik | Co to jest |
|---|---|
| `prompty.md` | **Główny plik:** analiza zmian A → B, 6 promptów (+ prompt 0), 2 korekty, tabela wymaganie → działa → dowód |
| `03_kod_ai_koncowy.py` | **Końcowy kod uzyskany od AI** |
| `raport_postaci.csv` | CSV wygenerowany przez kod AI |
| `rozmowa_ai.md` | Zapis rozmowy: co AI zrobiło po każdym prompcie i wynik testu |
| `kroki/` | Kod od AI po kolejnych promptach |
| `01_kod_poczatkowy.py` | Wersja A (jedyny kod pokazany AI) |
| `02_kod_docelowy.py`, `raport_postaci_B.csv`, `wynik_konsola_B.txt` | Wersja B i jej wynik – tylko do porównania |
| `wynik_konsola_ai.txt` | Wynik w konsoli kodu AI |

Uruchomienie: `pip install numpy`, potem `python 03_kod_ai_koncowy.py`.

**Wynik:** `raport_postaci.csv` z kodu AI jest identyczny bajt w bajt z CSV wersji B; na 500 losowych zestawach danych wyniki obu programów są zgodne.
