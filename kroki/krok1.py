"""Druga wersja systemu porównywania postaci RPG (z NumPy)."""

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

# Wagi do wzoru na siłę, w tej samej kolejności co KOLUMNY
WAGI = np.array([0.1, 2.0, 1.5, 1.0])


def oblicz_sile(statystyki, wagi):
    """Zwraca wektor sił postaci (suma ważona statystyk w każdym wierszu)."""
    return statystyki @ wagi


def najsilniejsza(nazwy, sily):
    """Zwraca nazwę i siłę najsilniejszej postaci (przy remisie: pierwsza)."""
    indeks = np.argmax(sily)
    return nazwy[indeks], sily[indeks]


def srednia_sila(sily):
    """Zwraca średnią siłę postaci."""
    return sily.mean()


def wypisz_sily(nazwy, sily):
    """Wypisuje siłę każdej postaci."""
    print("=== SIŁA POSTACI ===")
    for nazwa, sila in zip(nazwy, sily):
        print(f"{nazwa}: {sila:.2f}")


def main():
    assert STATYSTYKI.shape == (8, 4)
    assert STATYSTYKI.dtype == np.float64

    sily = oblicz_sile(STATYSTYKI, WAGI)
    wypisz_sily(NAZWY, sily)

    nazwa, _ = najsilniejsza(NAZWY, sily)
    print(f"\nNajsilniejsza postać: {nazwa}")
    print(f"Średnia siła: {srednia_sila(sily):.2f}")


if __name__ == "__main__":
    main()
