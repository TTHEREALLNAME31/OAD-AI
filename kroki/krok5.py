"""Szósta wersja systemu porównywania postaci RPG (NumPy: walidacja, testy, pojedynki)."""

import numpy as np

NAZWY = np.array([
    "Wojownik",
    "Mag",
    "Łucznik",
    "Paladyn",
    "Łotrzyk",
    "Berserker",
    "Druid",
    "Strażnik",
])

KOLUMNY = ("HP", "Atak", "Obrona", "Szybkość")

# Wiersze: postacie (kolejność jak w NAZWY), kolumny: HP, atak, obrona, szybkość
STATYSTYKI = np.array([
    [120, 28, 18, 9],
    [75, 44, 6, 13],
    [95, 34, 10, 17],
    [140, 25, 22, 7],
    [80, 39, 8, 21],
    [110, 42, 7, 12],
    [105, 26, 15, 14],
    [155, 18, 26, 5],
], dtype=float)

# Wagi dla HP, ataku, obrony i szybkości (sumują się do 1)
WAGI = np.array([0.20, 0.40, 0.25, 0.15])

# Indeksy kolumn
HP, ATAK, OBRONA, SZYBKOSC = 0, 1, 2, 3

# Współczynnik "ostrości" krzywej szans w pojedynku
STROMOSC = 8


def waliduj(nazwy, dane):
    """Sprawdza poprawność danych wejściowych; przy błędzie rzuca ValueError."""
    nazwy = np.asarray(nazwy)
    dane = np.asarray(dane, dtype=float)
    liczba_kolumn = len(KOLUMNY)

    if dane.ndim != 2 or dane.shape != (nazwy.size, liczba_kolumn):
        raise ValueError(
            f"Dane muszą być macierzą 2D o kształcie ({nazwy.size}, {liczba_kolumn}), "
            f"a mają kształt {dane.shape}."
        )
    if nazwy.size < 2:
        raise ValueError(
            f"Potrzebne są co najmniej 2 postacie, a jest {nazwy.size}."
        )
    unikalne, liczniki = np.unique(nazwy, return_counts=True)
    if np.any(liczniki > 1):
        powtorzone = ", ".join(unikalne[liczniki > 1])
        raise ValueError(f"Nazwy postaci się powtarzają: {powtorzone}.")
    if not np.all(np.isfinite(dane)):
        raise ValueError("Statystyki zawierają wartości NaN lub nieskończone (inf).")
    if np.any(dane < 0):
        raise ValueError("Statystyki nie mogą być ujemne.")


def normalizuj(dane):
    """Normalizacja min-max każdej kolumny osobno do zakresu [0, 1].

    Kolumna stała (max == min) daje same zera, bez dzielenia przez zero.
    """
    dane = np.asarray(dane, dtype=float)
    minimum = dane.min(axis=0)
    zakres = dane.max(axis=0) - minimum
    wynik = np.zeros_like(dane)
    np.divide(dane - minimum, zakres, out=wynik, where=zakres != 0)
    return wynik


def oblicz_sile(dane):
    """Zwraca wektor sił: znormalizowane dane @ WAGI."""
    return normalizuj(dane) @ WAGI


def ranking(sila):
    """Zwraca indeksy postaci posortowane malejąco wg siły (sortowanie stabilne)."""
    return np.argsort(-sila, kind="stable")


def okresl_role(dane):
    """Zwraca tablicę ról postaci (wektorowo, przez np.select).

    Kolejność warunków decyduje o pierwszeństwie: Szturmowiec przed Obrońcą.
    """
    warunki = [
        (dane[:, ATAK] >= 35) & (dane[:, SZYBKOSC] >= 12),
        (dane[:, HP] >= 120) & (dane[:, OBRONA] >= 18),
    ]
    role = ["Szturmowiec", "Obrońca"]
    return np.select(warunki, role, default="Wszechstronny")


def filtruj(dane, sila):
    """Zwraca maskę logiczną: HP >= 100 ORAZ szybkość >= 10 ORAZ siła >= 0.45."""
    return (dane[:, HP] >= 100) & (dane[:, SZYBKOSC] >= 10) & (sila >= 0.45)


def statystyki_grupy(dane):
    """Zwraca (min, średnia, max) dla każdej kolumny."""
    return dane.min(axis=0), dane.mean(axis=0), dane.max(axis=0)


def macierz_pojedynkow(sila):
    """Zwraca macierz N×N: element [i, j] to szansa, że postać i wygra z j.

    Liczona przez broadcasting; przekątna ustawiona dokładnie na 0.5.
    """
    sila = np.asarray(sila, dtype=float)
    roznice = sila[:, None] - sila[None, :]
    macierz = 1 / (1 + np.exp(-STROMOSC * roznice))
    np.fill_diagonal(macierz, 0.5)
    return macierz


def srednia_szansa(macierz):
    """Średnia szansa wygranej każdej postaci, bez walki z samą sobą."""
    n = macierz.shape[0]
    return (macierz.sum(axis=1) - 0.5) / (n - 1)


def testy():
    """Proste testy jednostkowe kluczowych funkcji."""
    # Normalizacja: kolumny 0 i 3 są stałe -> same zera
    przyklad = np.array([[10, 20, 30, 40], [10, 50, 60, 40]], dtype=float)
    oczekiwane = np.array([[0, 0, 0, 0], [0, 1, 1, 0]], dtype=float)
    np.testing.assert_allclose(normalizuj(przyklad), oczekiwane)
    np.testing.assert_allclose(normalizuj(przyklad)[:, 0], [0, 0])
    np.testing.assert_allclose(normalizuj(przyklad)[:, 3], [0, 0])

    # Macierz pojedynków: przekątna 0.5 i P + P.T == 1
    p = macierz_pojedynkow([0.1, 0.6, 0.9])
    assert p.shape == (3, 3)
    np.testing.assert_allclose(np.diag(p), [0.5, 0.5, 0.5])
    np.testing.assert_allclose(p + p.T, np.ones((3, 3)))


def wypisz_ranking(nazwy, role, sila, szanse, kolejnosc):
    print("=== RANKING POSTACI RPG ===")
    print(f"{'Miejsce':<8} {'Nazwa':<10} {'Rola':<14} {'Siła':>6}")
    for miejsce, i in enumerate(kolejnosc, start=1):
        print(
            f"{miejsce:<8} {nazwy[i]:<10} {role[i]:<14} {sila[i]:>6.3f}"
            f" | średnia szansa: {szanse[i] * 100:.1f}%"
        )


def wypisz_wymagania(nazwy, maska):
    print("\n=== POSTACIE SPEŁNIAJĄCE WYMAGANIA ===")
    wybrane = nazwy[maska]
    print(", ".join(wybrane) if wybrane.size > 0 else "Brak")


def wypisz_statystyki(minima, srednie, maksima):
    print("\n=== STATYSTYKI GRUPY ===")
    print(f"{'Cecha':<10} {'Min':>7} {'Średnia':>8} {'Max':>7}")
    for kolumna, mn, sr, mx in zip(KOLUMNY, minima, srednie, maksima):
        print(f"{kolumna:<10} {mn:>7.1f} {sr:>8.1f} {mx:>7.1f}")


def main():
    waliduj(NAZWY, STATYSTYKI)
    assert STATYSTYKI.dtype == np.float64
    testy()

    sila = oblicz_sile(STATYSTYKI)
    assert np.all((sila >= 0) & (sila <= 1))

    pojedynki = macierz_pojedynkow(sila)
    assert np.allclose(pojedynki + pojedynki.T, 1)
    assert np.all(np.diag(pojedynki) == 0.5)
    szanse = srednia_szansa(pojedynki)

    kolejnosc = ranking(sila)
    role = okresl_role(STATYSTYKI)
    maska = filtruj(STATYSTYKI, sila)

    wypisz_ranking(NAZWY, role, sila, szanse, kolejnosc)
    wypisz_wymagania(NAZWY, maska)
    wypisz_statystyki(*statystyki_grupy(STATYSTYKI))


if __name__ == "__main__":
    main()
