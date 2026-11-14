# Od wersji A do wersji B – prompty dla AI

**Materiały:** `01_kod_poczatkowy.py` (wersja A) i `02_kod_docelowy.py` (wersja B, tylko do analizy i porównania; **nie była pokazywana modelowi**).

## 1. Analiza: zmiany zauważone między A i B

| # | Obszar | Wersja A | Wersja B |
|---|---|---|---|
| 1 | Format danych | lista słowników | `NAZWY` – tablica NumPy z nazwami, `STATYSTYKI` – macierz 8×4 `float` (HP, atak, obrona, szybkość) |
| 2 | Wagi | wpisane w pętli (0.1 / 2 / 1.5 / 1) | wektor `WAGI = [0.20, 0.40, 0.25, 0.15]` (suma 1) |
| 3 | Skala | surowe wartości (HP dominuje skalą) | normalizacja min–max każdej kolumny do [0, 1] |
| 4 | Stała kolumna | – | gdy max = min, kolumna daje zera, bez dzielenia przez zero (`np.divide(..., where=...)`) |
| 5 | Wynik siły | pętla `for` | wektorowo: macierz znormalizowana `@ WAGI` |
| 6 | Ranking | tylko najlepsza postać | pełny ranking malejąco, sortowanie stabilne (`argsort(-sila, kind="stable")`) |
| 7 | Role | brak | `Szturmowiec` (atak ≥ 35 i szybkość ≥ 12), `Obrońca` (HP ≥ 120 i obrona ≥ 18), inaczej `Wszechstronny`; Szturmowiec ma pierwszeństwo (`np.select`) |
| 8 | Filtrowanie | brak | maska logiczna: HP ≥ 100 i szybkość ≥ 10 i siła ≥ 0.45; wynik „Brak”, gdy pusto |
| 9 | Statystyki grupy | tylko średnia siła | min / średnia / max dla każdej z 4 kolumn |
| 10 | Pojedynki | brak | macierz N×N: P(i wygrywa z j) = 1 / (1 + e^(−8·(siła_i − siła_j))), przekątna = 0.5 (broadcasting) |
| 11 | Średnia szansa | brak | średnia z wiersza macierzy **bez** walki z samym sobą: (suma − 0.5) / (N − 1) |
| 12 | Walidacja | brak | macierz 2D o kształcie (N, 4), N ≥ 2, unikatowe nazwy, liczby skończone i nieujemne → `ValueError` |
| 13 | Testy | brak | funkcja `testy()` z `np.testing.assert_allclose` / `assert`: normalizacja (w tym stała kolumna), przekątna 0.5, P + Pᵀ = 1 |
| 14 | Eksport | brak | `raport_postaci.csv`: separator `;`, kodowanie `utf-8-sig`, kolumny Miejsce;Postać;Rola;Siła;Średnia szansa wygranej, siła z 4 miejscami, szansa jako procent z 2 miejscami |
| 15 | Struktura | kod na poziomie modułu | funkcje + `main()` + `if __name__ == "__main__":` |
| 16 | Raport w konsoli | lista „nazwa: siła” | sekcje: RANKING (miejsce, nazwa, rola, siła 3 miejsca, szansa 1 miejsce %), POSTACIE SPEŁNIAJĄCE WYMAGANIA, STATYSTYKI GRUPY |

## 2. Prompty

Wszystkie prompty wysłane w **nowej rozmowie**. Na początku przekazano tylko wersję A.

### Prompt 0 – kontekst
```
Oto mój program w Pythonie (wersja A). Będę go przebudowywał krok po kroku.
Na razie nic nie zmieniaj, tylko potwierdź, że rozumiesz, co robi.

<tu wklejony cały 01_kod_poczatkowy.py>
```

### Prompt 1 – dane jako tablice NumPy + struktura
**Zmiany:** 1, 15
```
Przepisz dane na NumPy:
- NAZWY: np.array z 8 nazwami postaci w tej samej kolejności co w kodzie,
- STATYSTYKI: macierz np.array o kształcie (8, 4), dtype=float,
  kolumny w kolejności: HP, atak, obrona, szybkość,
- KOLUMNY = ("HP", "Atak", "Obrona", "Szybkość") jako etykiety.
Podziel program na funkcje i dodaj main() oraz if __name__ == "__main__":.
Nie używaj już listy słowników ani pętli for do obliczeń na statystykach.
Weryfikacja: STATYSTYKI.shape == (8, 4) i STATYSTYKI.dtype == float64.
```

### Prompt 2 – normalizacja i ważona siła
**Zmiany:** 2, 3, 4, 5
```
Zmień sposób liczenia siły:
1. Funkcja normalizuj(dane): normalizacja min-max każdej kolumny osobno
   (axis=0) do zakresu [0, 1]: (x - min) / (max - min).
2. Jeżeli kolumna jest stała (max == min), wynik w tej kolumnie ma być 0
   dla wszystkich postaci, BEZ dzielenia przez zero i bez ostrzeżeń NumPy
   (użyj np.divide z parametrami out= i where=).
3. WAGI = np.array([0.20, 0.40, 0.25, 0.15]) dla HP, ataku, obrony, szybkości.
4. Funkcja oblicz_sile(dane) zwraca wektor sił jako iloczyn macierzowy
   znormalizowanych danych i wag (operator @), bez pętli.
Weryfikacja: dla [[10,20,30,40],[10,50,60,40]] normalizacja daje
kolumnę 0 = [0,0], kolumnę 1 = [0,1], kolumnę 3 = [0,0].
Siła każdej postaci musi być w przedziale [0, 1].
```

### Prompt 3 – ranking, role, filtrowanie, statystyki grupy
**Zmiany:** 6, 7, 8, 9, 16
```
Dodaj:
1. Ranking: indeksy postaci posortowane malejąco wg siły,
   sortowanie stabilne (np.argsort(-sila, kind="stable")).
2. Funkcja okresl_role(dane) zwracająca tablicę ról, wektorowo przez np.select:
   - "Szturmowiec": atak >= 35 ORAZ szybkość >= 12 (ten warunek ma pierwszeństwo),
   - "Obrońca": HP >= 120 ORAZ obrona >= 18,
   - w pozostałych przypadkach "Wszechstronny".
3. Filtrowanie maską logiczną NumPy: postacie z HP >= 100 ORAZ szybkością >= 10
   ORAZ siłą >= 0.45. Wypisz ich nazwy po przecinku, a gdy nie ma żadnej – "Brak".
4. Statystyki grupy: dla każdej z 4 kolumn min, średnia i max (1 miejsce po przecinku).
Raport w konsoli w sekcjach:
=== RANKING POSTACI RPG ===   (miejsce, nazwa, rola, siła z 3 miejscami)
=== POSTACIE SPEŁNIAJĄCE WYMAGANIA ===
=== STATYSTYKI GRUPY ===
```

### Prompt 4 – macierz pojedynków i średnia szansa
**Zmiany:** 10, 11
```
Dodaj symulację pojedynków:
1. Funkcja macierz_pojedynkow(sila) zwraca macierz N×N, gdzie element [i, j]
   to szansa, że postać i wygra z j: 1 / (1 + exp(-8 * (sila_i - sila_j))).
   Licz wektorowo przez broadcasting (sila[:, None] - sila[None, :]), bez pętli.
2. Przekątna (walka z samym sobą) ma wynosić dokładnie 0.5 (np.fill_diagonal).
3. Średnia szansa wygranej postaci = średnia z jej wiersza, ale BEZ walki
   z samą sobą, czyli (suma wiersza - 0.5) / (N - 1).
4. Dopisz średnią szansę do każdej linii rankingu jako procent z 1 miejscem
   po przecinku, np. "| średnia szansa: 64.0%".
Weryfikacja: macierz + macierz.T == 1 wszędzie, przekątna == 0.5.
```

### Prompt 5 – walidacja danych i testy
**Zmiany:** 12, 13
```
Dodaj:
1. Funkcję waliduj(nazwy, dane) wywoływaną na początku main(), która rzuca
   ValueError z czytelnym komunikatem po polsku, gdy:
   - dane nie są macierzą 2D o kształcie (liczba_nazw, 4),
   - jest mniej niż 2 postacie,
   - nazwy się powtarzają,
   - jakakolwiek statystyka jest ujemna, NaN lub inf (np.isfinite).
2. Funkcję testy() wywoływaną w main() przed obliczeniami, używającą
   np.testing.assert_allclose i assert, sprawdzającą:
   - normalizację przykładu [[10,20,30,40],[10,50,60,40]] (w tym stałe kolumny = 0),
   - że macierz pojedynków dla sił [0.1, 0.6, 0.9] ma przekątną 0.5
     i spełnia P + P.T == 1.
```

### Prompt 6 – eksport CSV
**Zmiana:** 14
```
Dodaj funkcję zapisz_csv, która na końcu main() zapisuje raport_postaci.csv
modułem csv:
- separator ";", kodowanie "utf-8-sig" (żeby Excel poprawnie pokazał polskie znaki),
  newline="",
- nagłówek: Miejsce;Postać;Rola;Siła;Średnia szansa wygranej,
- wiersze w kolejności rankingu, miejsce od 1,
- siła z 4 miejscami po przecinku (np. 0.5349),
- średnia szansa jako procent z 2 miejscami (np. 63.98%).
Po zapisie wypisz "Zapisano: raport_postaci.csv".
Na koniec pokaż cały gotowy plik.
```

## 3. Korekty promptów

Prompty 1–6 dały poprawny wynik za pierwszym razem (CSV identyczny z B). Zgodnie z poleceniem przetestowałem więc przypadki brzegowe i dla dwóch nieudanych dopisałem prompty korygujące.

### Korekta 1 – dane nieliczbowe (po prompcie 5)
**Problem:** `waliduj(["A","B"], [["a",1,1,1],[1,1,1,1]])` kończyło się `ValueError` z angielskim komunikatem NumPy `could not convert string to float` – niezgodnie z wymaganiem „czytelny komunikat po polsku”.
```
Uruchomiłem program i przetestowałem walidację. Problem: gdy w danych jest tekst,
np. waliduj(np.array(["A","B"]), np.array([["a",1,1,1],[1,1,1,1]])), program kończy się
ValueError z angielskim komunikatem NumPy "could not convert string to float".
Popraw waliduj() tak, żeby:
- konwersję danych na float objąć try/except (ValueError, TypeError),
- w takim przypadku rzucić ValueError z polskim komunikatem "Statystyki muszą być liczbami.",
- pozostałe sprawdzenia i ich kolejność zostały bez zmian.
Dopisz też do testy() sprawdzenie, że waliduj rzuca ValueError dla danych z tekstem,
dla jednej postaci i dla wartości ujemnej (użyj try/except albo pomocniczej funkcji, bez pytest).
```
**Wynik:** `ValueError: Statystyki muszą być liczbami.`; `testy()` ma funkcję `oczekuj_value_error` i 3 nowe przypadki. ✅

### Korekta 2 – przepełnienie `exp` w macierzy pojedynków (po prompcie 6)
**Problem:** `macierz_pojedynkow([0.0, 200.0])` → `RuntimeWarning: overflow encountered in exp`; przy `python -W error` program się wywalał. (Wersja B ma ten sam problem.)
```
Drugi problem z testów brzegowych: macierz_pojedynkow(np.array([0.0, 200.0])) daje
RuntimeWarning "overflow encountered in exp" (przy uruchomieniu z python -W error program
się wywala). Dla normalnych danych siły są w [0, 1], ale funkcja ma być odporna na dowolne
skończone siły. Popraw ją tak, żeby:
- nie było żadnego ostrzeżenia overflow (np. przytnij wykładnik przez np.clip do rozsądnego
  zakresu, np. [-500, 500], albo użyj stabilnej postaci sigmoidy),
- wyniki dla sił z [0, 1] pozostały identyczne jak teraz (ten sam wzór 1/(1+exp(-8·różnica))),
- przekątna nadal = 0.5, a P + P.T = 1.
Dopisz do testy() przypadek [0.0, 200.0]: wynik ma być skończony, P[1,0] ≈ 1, P[0,1] ≈ 0.
```
**Wynik:** wykładnik przycinany `np.clip(..., -500, 500)`, test w `np.errstate(over="raise")`. Dla danych z zadania CSV nadal identyczny z B. ✅

## 4. Weryfikacja: wymaganie → działa / nie działa → dowód

Kod końcowy: `03_kod_ai_koncowy.py`. Uruchamiany z `python -W error` (każde ostrzeżenie NumPy traktowane jako błąd).

| # | Wymaganie (z wersji B) | Działa? | Dowód / wynik testu |
|---|---|---|---|
| 1 | Dane jako tablice NumPy (nazwy + macierz 8×4 float) | ✅ | `STATYSTYKI.shape == (8, 4)`, `dtype float64` (assert w `main`) |
| 2 | Normalizacja min-max po kolumnach | ✅ | `[[10,20,30,40],[10,50,60,40]]` → `[[0,0,0,0],[0,1,1,0]]` (test w `testy()`) |
| 3 | Stała kolumna → 0, bez dzielenia przez 0 | ✅ | macierz w całości stała → same zera, brak ostrzeżeń przy `-W error` |
| 4 | Ważona siła `normalizuj @ WAGI` | ✅ | siły identyczne z B: Berserker 0.5349 … Druid 0.3950 |
| 5 | Ranking malejąco, stabilny | ✅ | kolejność = B; przy remisie `[0.5,0.7,0.5,0.7]` → `[1,3,0,2]` |
| 6 | Role (`np.select`, Szturmowiec ma pierwszeństwo) | ✅ | role = B; postać `[130,40,20,15]` spełnia oba warunki → „Szturmowiec” |
| 7 | Filtrowanie maską + „Brak” | ✅ | wynik „Berserker” = B; pusta maska → „Brak” |
| 8 | Statystyki grupy min/średnia/max | ✅ | HP 75.0/110.0/155.0, Atak 18.0/32.0/44.0, Obrona 6.0/14.0/26.0, Szybkość 5.0/12.2/21.0 = B |
| 9 | Macierz pojedynków (sigmoida ×8, broadcasting, przekątna 0.5) | ✅ | `P + P.T == 1`, `diag == 0.5` (asserty i `testy()`) |
| 10 | Średnia szansa bez walki z samym sobą | ✅ | 64.0%, 58.8%, 54.0%, 50.9%, 47.3%, 46.2%, 45.3%, 33.5% = B |
| 11 | Walidacja (kształt, ≥2 postacie, unikatowe nazwy, NaN/inf, ujemne) | ✅ | 6 złych zestawów: B i AI rzucają `ValueError` w tych samych przypadkach; dodatkowo tekst → polski komunikat (korekta 1) |
| 12 | Testy (`np.testing.assert_allclose`) | ✅ | `testy()` przechodzi przy każdym uruchomieniu; zawiera też walidację i przypadek skrajny (korekta 2) |
| 13 | Eksport CSV (`;`, utf-8-sig, format liczb) | ✅ | `cmp raport_postaci.csv raport_postaci_B.csv` → **pliki identyczne bajt w bajt** |
| 14 | Struktura: funkcje + `main()` + `__main__` | ✅ | kod końcowy |
| 15 | Raport w konsoli w 3 sekcjach | ✅ | `wynik_konsola_ai.txt`; różni się od B tylko wyglądem tabeli (nagłówki kolumn) – formatowanie nie jest oceniane |
| 16 | Zgodność funkcjonalna na innych danych | ✅ | 500 losowych zestawów (2–14 postaci, stałe kolumny, remisy): siła, role, ranking, macierz pojedynków i filtr **identyczne z B (0 różnic)** |

**Różnice względem B (świadome, nie wpływają na wynik):** AI ma nagłówki kolumn w rankingu, stałe `HP/ATAK/OBRONA/SZYBKOSC`, `STROMOSC`, polski komunikat dla tekstu w danych i ochronę przed przepełnieniem `exp`.
