# Zapis rozmowy z AI

Nowa rozmowa, model: Claude (osobna, czysta sesja). Modelowi przekazano **tylko wersję A** i kolejne prompty z `prompty.md`. Wersji B nie pokazywano.

## Prompt 0 → odpowiedź AI (skrót)

AI potwierdziło, że rozumie wersję A: lista 8 słowników, siła liczona wzorem `hp·0.1 + atak·2 + obrona·1.5 + szybkosc` w pętli, wyszukiwanie maksimum (`>` – przy remisie wygrywa wcześniejsza postać), średnia siła. Ręcznie policzyło: najsilniejszy Łotrzyk (119.00), średnia 108.25 – zgodne z uruchomieniem A. Zauważyło też `ZeroDivisionError` przy pustej liście i zapowiedź przejścia na NumPy.

## Prompt 1 → odpowiedź AI
AI utworzyło `NAZWY`, macierz `STATYSTYKI` (8, 4) `float`, `KOLUMNY`, wektor wag i liczenie `STATYSTYKI @ WAGI`; funkcje `oblicz_sile`, `najsilniejsza` (`np.argmax`), `srednia_sila`, `wypisz_sily`, `main()` z `assert shape/dtype`.
**Test:** uruchomione – wynik jak w A (Łotrzyk, średnia 108.25). ✅ (kod: `krok1.py`)

## Prompt 2 → odpowiedź AI
AI dodało `normalizuj()` (min-max po `axis=0`, `np.divide(..., out=np.zeros_like, where=zakres != 0)`), nowe `WAGI = [0.20, 0.40, 0.25, 0.15]`, `oblicz_sile = normalizuj(dane) @ WAGI` i `assert` na zakres [0, 1]. Samo zauważyło, że zwycięzcą będzie teraz Berserker.
**Test:** uruchomione z `-W error` (każde ostrzeżenie NumPy = błąd) – brak ostrzeżeń. Przykład `[[10,20,30,40],[10,50,60,40]]` → `[[0,0,0,0],[0,1,1,0]]`; macierz w całości stała → same zera. Siły zgodne z B (Berserker 0.53, Druid 0.39). ✅ (kod: `krok2.py`)

## Prompt 3 → odpowiedź AI
AI dodało `ranking()` (`argsort(-sila, kind="stable")`), `okresl_role()` przez `np.select` (Szturmowiec pierwszy), `filtruj()` (maska `&`), `statystyki_grupy()` i raport w trzech sekcjach; stare funkcje z wersji A usunęło.
**Test:** uruchomione – kolejność rankingu, role, filtr (tylko Berserker) i statystyki grupy identyczne z B. ✅ Różnica tylko w wyglądzie tabeli (nagłówki kolumn, brak średniej szansy – to dopiero w prompcie 4). (kod: `krok3.py`)

## Prompt 4 → odpowiedź AI
AI dodało `macierz_pojedynkow()` (broadcasting `sila[:, None] - sila[None, :]`, sigmoida ze współczynnikiem 8 jako stała `STROMOSC`, `np.fill_diagonal(…, 0.5)`), `srednia_szansa()` = (suma − 0.5)/(N − 1) i `assert` na P + Pᵀ = 1 (przez `np.allclose`, z wyjaśnieniem błędów zaokrągleń).
**Test:** ranking z szansami identyczny z B (64.0%, 58.8%, 54.0%, 50.9%, 47.3%, 46.2%, 45.3%, 33.5%). ✅
**Przypadek brzegowy:** dla jednej postaci `srednia_szansa` daje `nan` i `RuntimeWarning` (dzielenie przez N − 1 = 0) → musi to wyłapać walidacja (prompt 5). Priorytet ról sprawdzony: postać spełniająca oba warunki dostaje „Szturmowiec”. (kod: `krok4.py`)

## Prompt 5 → odpowiedź AI
AI dodało `waliduj()` (kształt 2D (N, 4), N ≥ 2, unikatowe nazwy przez `np.unique(return_counts=True)`, `np.isfinite`, brak ujemnych) – sprawdzanie NaN celowo przed ujemnymi, bo `NaN < 0` daje `False` – oraz `testy()` z `np.testing.assert_allclose`.
**Test:** program działa jak wcześniej. Walidacja sprawdzona na 9 złych zestawach danych: 1 postać, 3 kolumny, dane 1D, duplikat, wartość ujemna, NaN, inf → każdy daje `ValueError` z polskim komunikatem. ✅
**Przypadek brzegowy:** dane z tekstem (`"a"` zamiast liczby) też kończą się `ValueError`, ale z angielskim komunikatem NumPy `could not convert string to float` → korekta 1. (kod: `krok5.py`)

## Prompt 6 → odpowiedź AI
AI dodało `zapisz_csv()` (moduł `csv`, `;`, `utf-8-sig`, `newline=""`, siła `.4f`, szansa `*100:.2f%`).
**Test:** `raport_postaci.csv` od AI jest **identyczny bajt w bajt** z plikiem z wersji B. ✅

## Korekta 1 → odpowiedź AI
Konwersja w `waliduj()` objęta `try/except (ValueError, TypeError)` → `ValueError("Statystyki muszą być liczbami.") from None`; nowa funkcja `oczekuj_value_error()` i 3 testy walidacji w `testy()`. AI samo zauważyło, że tekst `"5"` przejdzie (NumPy go zamieni na liczbę) i że nierówne wiersze dostaną ten sam komunikat.
**Test:** tekst w danych → `Statystyki muszą być liczbami.` ✅ (kod: `krok7.py`)

## Korekta 2 → odpowiedź AI
Stała `LIMIT_WYKLADNIKA = 500`, `np.clip(-STROMOSC * roznice, -500, 500)` przed `np.exp`; test `[0.0, 200.0]` w `np.errstate(over="raise")`. AI wyjaśniło, że dla sił z [0, 1] wykładnik jest w [-8, 8], więc wynik się nie zmienia.
**Test:** brak ostrzeżeń przy `-W error`, CSV nadal identyczny z B, 500 losowych zestawów – 0 różnic względem B. ✅ (kod końcowy: `03_kod_ai_koncowy.py`)
