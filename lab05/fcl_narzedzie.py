"""
Podstawy AI – narzędzie do plików FCL (Fuzzy Control Language, norma IEC 61131-7).

Czyta plik FCL i sam liczy wyjście sterownika rozmytego. Potrzebuje tylko numpy i matplotlib
(w Google Colab są zainstalowane – nic nie trzeba instalować).

    from fcl_narzedzie import wynik, tabela, rysuj_zbiory, krzywa, wariant, rozszerz, krzywe

    wynik("wentylator.fcl", temperature=25)                       # jedna liczba
    tabela(["a.fcl", "b.fcl"], "temperature", [0, 5, 10, 15])     # tabela testów
    rysuj_zbiory("wentylator.fcl")                                # wykresy funkcji przynależności
    krzywa(["a.fcl", "b.fcl"], "temperature")                     # wyjście w funkcji wejścia
    wariant("a.fcl", "a_prod.fcl", ACT="PROD")                    # kopia pliku ze zmienionym ustawieniem
    rozszerz("a.fcl", "b.fcl", wejscie=..., fuzzify=..., reguly=...)  # dopisanie nowego wejścia i reguł
    krzywe("b.fcl", "temperature", "humidity", [20, 90])          # linia dla każdej wartości drugiego wejścia

Obsługiwane: TERM jako punkty (x, μ) albo trian/trape, RANGE, METHOD (COG, COA, MM, LM, RM),
DEFAULT, AND (MIN, PROD, BDIF), OR (MAX, ASUM, BSUM), ACT (MIN, PROD), ACCU (MAX, ASUM, BSUM, NSUM),
reguły z AND, OR i NOT (AND wiąże silniej niż OR).
"""
import re
import numpy as np
import matplotlib.pyplot as plt

ROZDZIELCZOSC = 1001          # liczba punktów na osi wyjścia przy wyostrzaniu

_T = {"MIN": np.minimum, "PROD": np.multiply, "BDIF": lambda a, b: np.maximum(0.0, a + b - 1)}
_S = {"MAX": np.maximum, "ASUM": lambda a, b: a + b - a * b, "PROBOR": lambda a, b: a + b - a * b,
      "BSUM": lambda a, b: np.minimum(1.0, a + b), "NSUM": lambda a, b: a + b}


# --------------------------------------------------------------------------- funkcje przynależności
class Termin:
    def __init__(self, nazwa, xs, ys):
        self.nazwa, self.xs, self.ys = nazwa, np.asarray(xs, float), np.asarray(ys, float)

    def __call__(self, x):
        return np.interp(x, self.xs, self.ys)          # łamana; poza punktami – wartość skrajna


def _termin(nazwa, opis):
    opis = opis.strip()
    m = re.match(r"(trian|trape)\s+(.*)", opis, re.I)
    if m:
        p = [float(v) for v in m.group(2).split()]
        if m.group(1).lower() == "trian":
            return Termin(nazwa, [p[0], p[1], p[2]], [0, 1, 0])
        return Termin(nazwa, [p[0], p[1], p[2], p[3]], [0, 1, 1, 0])
    punkty = re.findall(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)", opis)
    if not punkty:
        raise ValueError(f"Nie rozumiem definicji terminu {nazwa!r}: {opis!r}")
    xs, ys = zip(*[(float(a), float(b)) for a, b in punkty])
    return Termin(nazwa, xs, ys)


class Zmienna:
    def __init__(self, nazwa):
        self.nazwa, self.terminy, self.min, self.max = nazwa, {}, None, None


# --------------------------------------------------------------------------- wczytanie pliku FCL
class Sterownik:
    def __init__(self, plik):
        tekst = open(plik, encoding="utf-8").read()
        tekst = re.sub(r"\(\*.*?\*\)", " ", tekst, flags=re.S)   # komentarze (* ... *)
        tekst = re.sub(r"//[^\n]*", " ", tekst)
        self.plik = plik
        self.wejscia, self.wyjscia = {}, {}
        for typ, slownik in (("VAR_INPUT", self.wejscia), ("VAR_OUTPUT", self.wyjscia)):
            for tresc in re.findall(rf"{typ}(.*?)END_VAR", tekst, flags=re.S | re.I):
                for n in re.findall(r"(\w+)\s*:\s*REAL", tresc, flags=re.I):
                    slownik[n] = Zmienna(n)
        wszystkie = {**self.wejscia, **self.wyjscia}

        bloki = re.findall(r"(FUZZIFY|DEFUZZIFY)\s+(\w+)(.*?)END_\1", tekst, flags=re.S | re.I)
        for _, n, tresc in bloki:
            if n not in wszystkie:
                raise ValueError(f"Blok dla zmiennej {n!r}, której nie ma w VAR_INPUT / VAR_OUTPUT")
            v = wszystkie[n]
            for tn, to in re.findall(r"TERM\s+(\w+)\s*:=\s*(.*?);", tresc, flags=re.S | re.I):
                v.terminy[tn] = _termin(tn, to)
            m = re.search(r"RANGE\s*:=\s*\(\s*([-\d.]+)\s*\.\.\s*([-\d.]+)\s*\)", tresc, re.I)
            if m:
                v.min, v.max = float(m.group(1)), float(m.group(2))
            else:
                xs = np.concatenate([t.xs for t in v.terminy.values()])
                v.min, v.max = float(xs.min()), float(xs.max())
            if n in self.wyjscia:
                m = re.search(r"METHOD\s*:\s*(\w+)", tresc, re.I)
                v.metoda = (m.group(1) if m else "COG").upper()
                m = re.search(r"DEFAULT\s*:=\s*([-\d.]+)", tresc, re.I)
                v.domyslna = float(m.group(1)) if m else None

        blok = re.search(r"RULEBLOCK\s+\w+(.*?)END_RULEBLOCK", tekst, flags=re.S | re.I)
        if not blok:
            raise ValueError("Brak bloku RULEBLOCK")
        tresc = blok.group(1)

        def ustaw(klucz, domyslnie):
            m = re.search(rf"\b{klucz}\s*:\s*(\w+)", tresc, re.I)
            return (m.group(1) if m else domyslnie).upper()

        self.AND, self.OR = _T[ustaw("AND", "MIN")], _S[ustaw("OR", "MAX")]
        self.ACT, self.ACCU_nazwa = ustaw("ACT", "MIN"), ustaw("ACCU", "MAX")
        self.ACCU = _S[self.ACCU_nazwa]
        self.reguly = [self._regula(r) for r in re.findall(r"RULE\s+\w+\s*:\s*(.*?);", tresc, flags=re.S | re.I)]

    def _regula(self, tekst):
        tekst = " ".join(tekst.split())
        m = re.match(r"IF\s+(.*)\s+THEN\s+(\w+)\s+IS\s+(\w+)$", tekst, re.I)
        if not m:
            raise ValueError(f"Nie rozumiem reguły: {tekst!r}")
        warunek, wy, term = m.groups()
        if wy not in self.wyjscia or term not in self.wyjscia[wy].terminy:
            raise ValueError(f"Reguła {tekst!r}: nie ma wyjścia/stanu {wy} IS {term}")
        lub = []
        for grupa in re.split(r"\s+OR\s+", warunek, flags=re.I):
            i = []
            for czesc in re.split(r"\s+AND\s+", grupa, flags=re.I):
                c = re.match(r"(\w+)\s+IS\s+(NOT\s+)?(\w+)$", czesc.strip(), re.I)
                if not c:
                    raise ValueError(f"Reguła {tekst!r}: nie rozumiem fragmentu {czesc!r}")
                zm, nie, st = c.group(1), bool(c.group(2)), c.group(3)
                if zm not in self.wejscia or st not in self.wejscia[zm].terminy:
                    raise ValueError(f"Reguła {tekst!r}: nie ma wejścia/stanu {zm} IS {st}")
                i.append((zm, nie, st))
            lub.append(i)
        return lub, wy, term

    # ----------------------------------------------------------------------- obliczenia
    def oblicz(self, **wejscia):
        for n in self.wejscia:
            if n not in wejscia:
                raise KeyError(f"Brakuje wejścia {n!r}. Sterownik ma wejścia: {list(self.wejscia)}")
        wy = next(iter(self.wyjscia.values()))
        x = np.linspace(wy.min, wy.max, ROZDZIELCZOSC)
        ksztalt = np.zeros_like(x)
        for lub, _, term in self.reguly:
            sila = None
            for grupa in lub:                                   # 1–2. rozmycie i siła reguły
                s_grupy = None
                for zm, nie, st in grupa:
                    mu = float(self.wejscia[zm].terminy[st](wejscia[zm]))
                    mu = 1 - mu if nie else mu
                    s_grupy = mu if s_grupy is None else float(self.AND(s_grupy, mu))
                sila = s_grupy if sila is None else float(self.OR(sila, s_grupy))
            zbior = wy.terminy[term](x)                         # 3. implikacja
            wniosek = np.minimum(sila, zbior) if self.ACT == "MIN" else sila * zbior
            ksztalt = self.ACCU(ksztalt, wniosek)               # 4. suma zbiorów
        if self.ACCU_nazwa == "NSUM" and ksztalt.max() > 1:
            ksztalt = ksztalt / ksztalt.max()
        if ksztalt.max() <= 0:                                  # żadna reguła nie działa
            return wy.domyslna
        return float(_wyostrz(x, ksztalt, wy.metoda))          # 5. wyostrzanie


def _wyostrz(x, mu, metoda):
    if metoda == "COG":
        return np.sum(x * mu) / np.sum(mu)
    if metoda == "COA":
        c = np.cumsum(mu)
        return x[np.searchsorted(c, c[-1] / 2)]
    maks = x[np.isclose(mu, mu.max())]
    return {"MM": maks.mean(), "LM": maks.min(), "RM": maks.max()}[metoda]


def wczytaj_fcl(plik):
    return Sterownik(plik)


# --------------------------------------------------------------------------- funkcje dla studentów
def wynik(plik, **wejscia):
    """Wyjście sterownika dla podanych wejść, np. wynik("wentylator.fcl", temperature=25)."""
    w = Sterownik(plik).oblicz(**wejscia)
    return None if w is None else round(w, 2)


def tabela(pliki, wejscie, wartosci, **stale):
    """Tabela testów: wiersz = wartość wejścia, kolumna = plik FCL. Inne wejścia podaj jako stale=..."""
    pliki = [pliki] if isinstance(pliki, str) else pliki
    nazwy = [p.rsplit("/", 1)[-1].replace(".fcl", "") for p in pliki]
    szer = [max(len(n), 8) + 2 for n in nazwy]
    sterowniki = [Sterownik(p) for p in pliki]
    print(f"{wejscie:>12}" + "".join(f"{n:>{w}}" for n, w in zip(nazwy, szer)))
    for x in wartosci:
        wiersz = []
        for s in sterowniki:
            w = s.oblicz(**{wejscie: x}, **stale)
            wiersz.append("brak" if w is None else f"{w:.2f}")
        print(f"{x:>12}" + "".join(f"{v:>{w}}" for v, w in zip(wiersz, szer)))


def rysuj_zbiory(plik):
    """Wykresy funkcji przynależności wszystkich zmiennych z pliku FCL."""
    s = Sterownik(plik)
    zmienne = [("wejście: ", v) for v in s.wejscia.values()] + [("wyjście: ", v) for v in s.wyjscia.values()]
    fig, osie = plt.subplots(1, len(zmienne), figsize=(6 * len(zmienne), 3))
    for os_, (rodzaj, v) in zip(np.atleast_1d(osie), zmienne):
        xs = np.linspace(v.min, v.max, 501)
        for t in v.terminy.values():
            os_.plot(xs, t(xs), lw=2, label=t.nazwa)
        os_.set_title(rodzaj + v.nazwa); os_.set_ylim(0, 1.1); os_.grid(alpha=.3); os_.legend(fontsize=8)
    plt.suptitle(plik.rsplit("/", 1)[-1]); plt.tight_layout(); plt.show()


def krzywa(pliki, wejscie, **stale):
    """Wykres: wyjście sterownika w funkcji jednego wejścia, dla kilku plików FCL naraz."""
    pliki = [pliki] if isinstance(pliki, str) else pliki
    s0 = Sterownik(pliki[0])
    v = s0.wejscia[wejscie]
    xs = np.linspace(v.min, v.max, 201)
    plt.figure(figsize=(8, 3.5))
    for p in pliki:
        s = Sterownik(p)
        ys = [s.oblicz(**{wejscie: x}, **stale) for x in xs]
        plt.plot(xs, [np.nan if y is None else y for y in ys], lw=2, label=p.rsplit("/", 1)[-1])
    plt.xlabel(wejscie); plt.ylabel(next(iter(s0.wyjscia)))
    plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout(); plt.show()


def krzywe(plik, wejscie, parametr, wartosci):
    """Wyjście w funkcji jednego wejścia, osobna linia dla każdej wartości drugiego wejścia."""
    s = Sterownik(plik)
    v = s.wejscia[wejscie]
    xs = np.linspace(v.min, v.max, 201)
    plt.figure(figsize=(8, 3.5))
    for w in wartosci:
        ys = [s.oblicz(**{wejscie: x, parametr: w}) for x in xs]
        plt.plot(xs, [np.nan if y is None else y for y in ys], lw=2, label=f"{parametr} = {w}")
    plt.xlabel(wejscie); plt.ylabel(next(iter(s.wyjscia))); plt.title(plik.rsplit("/", 1)[-1])
    plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout(); plt.show()


def wariant(plik, nowy_plik, **zmiany):
    """Kopia pliku FCL ze zmienionymi ustawieniami, np. wariant("a.fcl", "b.fcl", ACT="PROD", METHOD="MM")."""
    tekst = open(plik, encoding="utf-8").read()
    for klucz, wartosc in zmiany.items():
        tekst, n = re.subn(rf"(\b{klucz}\s*:\s*)\w+", rf"\g<1>{wartosc}", tekst)
        if n == 0:
            raise ValueError(f"W pliku {plik} nie ma ustawienia {klucz}")
    open(nowy_plik, "w", encoding="utf-8").write(tekst)
    print(f"Zapisano {nowy_plik}: " + ", ".join(f"{k} = {v}" for k, v in zmiany.items()))
    return nowy_plik


def rozszerz(plik, nowy_plik, wejscie="", fuzzify="", reguly=""):
    """Kopia pliku FCL z dopisanymi fragmentami:
    wejscie  – linia(e) do VAR_INPUT, np. "humidity : REAL;"
    fuzzify  – cały blok FUZZIFY ... END_FUZZIFY dla nowego wejścia
    reguly   – nowe linie RULE ... (wstawione przed END_RULEBLOCK)"""
    tekst = open(plik, encoding="utf-8").read()
    if wejscie:
        i = tekst.index("END_VAR")                      # pierwszy END_VAR zamyka VAR_INPUT
        tekst = tekst[:i] + "    " + wejscie.strip() + "\n" + tekst[i:]
    if fuzzify:
        i = tekst.index("DEFUZZIFY")
        tekst = tekst[:i] + fuzzify.strip() + "\n\n" + tekst[i:]
    if reguly:
        i = tekst.index("END_RULEBLOCK")
        tekst = tekst[:i] + "\n".join("    " + r.strip() for r in reguly.strip().splitlines()) + "\n" + tekst[i:]
    open(nowy_plik, "w", encoding="utf-8").write(tekst)
    try:
        Sterownik(nowy_plik)                            # sprawdzenie, czy plik jest poprawny
    except Exception as e:
        raise ValueError(f"Plik {nowy_plik} zapisano, ale ma błąd – popraw fragmenty i uruchom komórkę ponownie. "
                         f"Szczegóły: {e}") from None
    print(f"Zapisano {nowy_plik} – plik wczytuje się poprawnie.")
    return nowy_plik
