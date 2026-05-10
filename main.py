# ==============================================================
# HLAVNÝ SÚBOR – Meteorologická analýza meteorologických dát
# ==============================================================

import os               # Práca so súbormi a adresármi
import warnings         # Knižnica na správu varovných hlásení
warnings.filterwarnings('ignore')   # Potlačíme všetky varovania

# ---- Načítanie našich modulov ----
from nacitanie_dat     import nacitaj_data                          # Načítanie CSV súboru
from analyza_chyb      import (analyza_chybajucich_dni,             # Analýza chýbajúcich dní
                                analyza_chybajucich_vzoriek,         # Analýza NaN hodnôt
                                detekcia_poskodennych_vzoriek,       # Detekcia poškodených vzoriek
                                histogramy_dlzok_chyb,               # Histogramy dĺžok skupín chýb
                                vycisti_a_uloz)                      # Čistenie dát → cleaned.csv
from analyza_ziarenia  import rocna_analyza_ziarenia                # Ročná analýza žiarenia
from analyza_ziarenia  import denna_analyza_pre_mesiace             # Denné profily pre každý mesiac
from analyza_korelacie import suhrne_statistiky                     # Súhrnné štatistiky parametrov
from analyza_korelacie import korelacie_parametrov                  # Korelačná analýza
from generuj_html      import generuj_html_report                   # Generovanie HTML reportu


# ==============================================================
# NASTAVENIA – stačí zmeniť tieto premenné
# ==============================================================

#CSV    = 'output_analyza_train_raw/cleaned.csv'   # Vstupný súbor s meteorologickými dátami
#OUTPUT = 'output_cleaned'                         # Adresár, kam sa uložia výstupy

CSV    = 'meteo_data/train/meteo_data_raw.csv'   # Vstupný súbor s meteorologickými dátami
OUTPUT = 'output_analyza_train_raw'              # Adresár, kam sa uložia výstupy

# ==============================================================
# SPUSTENIE ANALÝZY
# ==============================================================

os.makedirs(OUTPUT, exist_ok=True)   # Vytvoríme výstupný adresár ak ešte neexistuje

# ---- 1. Načítanie dát ----
df = nacitaj_data(CSV)   # Načítame CSV a pripravíme DataFrame
print(f"Načítaných {len(df):,} vzoriek z {df['Date'].nunique()} dní.")

# ---- 2. Súhrnné štatistiky ----
suhrne_statistiky(df, OUTPUT)   # Vypočítame min, max, priemer, std pre každý parameter

# ---- 3. Analýza chýb ----
analyza_chybajucich_dni(df, OUTPUT)                          # Nájdeme dni bez akýchkoľvek meraní
analyza_chybajucich_vzoriek(df)                              # Spočítame NaN hodnoty v každom stĺpci
poskodene_maska, _ = detekcia_poskodennych_vzoriek(df)       # Detekujeme hodnoty mimo fyzikálnych medzi
histogramy_dlzok_chyb(df, poskodene_maska, OUTPUT)           # Histogramy dĺžok skupín chýb

# ---- 3b. Čistenie dát ----
df_clean = vycisti_a_uloz(df, OUTPUT)   # Odstráni chybné riadky a uloží cleaned.csv do OUTPUT adresára

# ---- 4. Analýza žiarenia ----
rocna_analyza_ziarenia(df, OUTPUT)       # Ročný priebeh žiarenia s disperziou
denna_analyza_pre_mesiace(df, OUTPUT)    # Denné profily žiarenia pre každý mesiac

# ---- 5. Korelačná analýza ----
korelacie_parametrov(df, OUTPUT)   # Pearsonova korelačná matica a scatter ploty

# ---- 6. Generovanie HTML reportu ----
generuj_html_report(CSV, OUTPUT)   # Zostavíme jeden prenosný HTML súbor so všetkými výsledkami


# ---- Záverečný výpis vygenerovaných súborov ----
print(f"\n{'=' * 50}")
print(f"  HOTOVO – výstup v adresári: {OUTPUT}/")
print(f"{'=' * 50}")
for subor in sorted(os.listdir(OUTPUT)):   # Iterujeme cez všetky súbory vo výstupnom adresári
    velkost = os.path.getsize(os.path.join(OUTPUT, subor)) / 1024   # Veľkosť v kB
    print(f"  {subor:45s} ({velkost:.0f} kB)")   # Vypíšeme názov a veľkosť
