# ==============================================================
# MODUL: Analýza chýb – chýbajúce dni, chýbajúce a poškodené vzorky
# ==============================================================

import os               # Práca so súborovým systémom (cesty, adresáre)
import numpy as np      # Numerické výpočty (priemer skupín)
import pandas as pd     # Práca s tabuľkovými dátami
import matplotlib.pyplot as plt    # Kreslenie grafov
import matplotlib.dates as mdates  # Formátovanie dátumov na osiach


# ==============================================================
# FYZIKÁLNE MEDZE – jediné miesto definície, importujú ostatné moduly
# ==============================================================

FYZIKALNE_MEDZE = {
    'Irradiance':               (0,     1500),  # W/m²  – záporné žiarenie fyzikálne nemožné; max ~1400 W/m² pri povrchu
    'IrradianceNotCompensated': (0,     1500),  # W/m²  – rovnaké medze ako Irradiance (nekompenzovaná hodnota)
    'BodyTemperature':          (-40,   80),    # °C    – operačný rozsah Kipp & Zonen SMP10: -40 až +70 °C
    'RelativeHumidity':         (0,     100),   # %     – fyzikálna definícia relatívnej vlhkosti
    'HumidityTemp':             (-40,   60),    # °C    – rozsah snímača vlhkosti (Campbell EE181: -40 až +60 °C)
    'Pressure':                 (870,   1085),  # hPa   – pozemné extrémy redukované na more: min 870 (cyklón), max 1084.8 (Mongolsko)
    'PressureAvg':              (870,   1085),  # hPa   – rovnaké medze ako Pressure
    'PressureTemp':             (-40,   80),    # °C    – teplota tlakového senzora (rovnaký rozsah ako BodyTemperature)
    'PressureTempAvg':          (-40,   80),    # °C    – priemer teploty tlakového senzora
    'TiltAngle':                (-10,   10),    # °     – pyranometer sa inštaluje blízko horizontály; ±10° pokrýva aj chyby montáže
    'TiltAngleAvg':             (-10,   10),    # °     – priemer uhla naklonenia
    'FanSpeed':                 (0,     15000), # RPM   – typický DC ventilátor; v dátach: 8779–9490 RPM
    'HeaterCurrent':            (0,     1.0),   # A     – CVF4 celkový odber 0.9 A pri 12 V (ventilátor + ohrievač)
    'FanCurrent':               (0,     0.5),   # A     – ventilátor CVF4 ~5 W pri 12 V → ~0.42 A; v dátach: 0.064–0.073 A
    'SunLatitude':              (-90,   90),    # °     – zemepisná šírka stanice; WGS84: ±90°
    'SunLongitude':             (-180,  180),   # °     – zemepisná dĺžka stanice; WGS84: ±180°
    'SunAzimuth':               (0,     360),   # °     – kompasový smer slnka (0° = sever, 180° = juh)
    'SunZenith':                (0,     90),    # °     – 0° = slnko v nadhlavníku, 90° = horizont; na 46.94°N (Alpnach) záporné hodnoty fyzikálne neexistujú
}


# --------------------------------------------------------------
# 1. Chýbajúce dni
# --------------------------------------------------------------

def analyza_chybajucich_dni(df, output_dir):
    """
    Nájde dni, v ktorých chýbajú VŠETKY merania (celé vynechané dni).
    Uloží histogramy.
    """
    print("\n=== ANALÝZA CHÝBAJÚCICH DNÍ ===")   # Nadpis sekcie

    # Nájdenie najstaršieho a najnovšieho dátumu v dátach
    min_datum = df['DateTime'].min().date()   # Prvý deň merania
    max_datum = df['DateTime'].max().date()   # Posledný deň merania

    # Generovanie zoznamu každého dňa od začiatku do konca merania
    vsetky_dni = pd.date_range(start=min_datum, end=max_datum, freq='D').date   # Všetky dni v rozsahu

    # Zistenie, ktoré dni sa skutočne nachádzajú v dátach
    pritomne_dni = set(df['Date'].unique())   # Množina dní, pre ktoré existujú záznamy

    # Odčítaním množín nájdeme chýbajúce dni
    chybajuce_dni = sorted(set(vsetky_dni) - pritomne_dni)   # Dni v rozsahu, ale bez dát

    # Výpis štatistík
    print(f"  Rozsah merania:              {min_datum} až {max_datum}")   # Časový rozsah
    print(f"  Celkový počet dní v rozsahu: {len(vsetky_dni)}")            # Koľko dní malo byť
    print(f"  Počet dní s dátami:          {len(pritomne_dni)}")          # Koľko dní skutočne je
    print(f"  Počet chýbajúcich dní:       {len(chybajuce_dni)}")         # Rozdiel
    if chybajuce_dni:                                                      # Ak nejaké chýbajú
        print(f"  Chýbajúce dni:               {chybajuce_dni}")          # Vypíšeme ich zoznam

    # --- Graf: Histogram chýbajúcich dní podľa mesiaca ---
    if chybajuce_dni:   # Graf kreslíme len ak niečo chýba
        mesiace_chyb = [d.month for d in chybajuce_dni]   # Extrakcia mesiaca z každého chýbajúceho dňa
        fig, ax = plt.subplots(figsize=(12, 5))            # Nová figúra
        ax.hist(mesiace_chyb, bins=range(1, 14), align='left', rwidth=0.8,
                color='tomato', edgecolor='black')          # Histogram: každý mesiac = jeden stĺpec
        ax.set_xlabel('Mesiac')                            # Popis osi x
        ax.set_ylabel('Počet chýbajúcich dní')             # Popis osi y
        ax.set_title('Histogram chýbajúcich dní podľa mesiaca')   # Nadpis grafu
        ax.set_xticks(range(1, 13))                        # Značky na osi x pre každý mesiac
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Máj', 'Jún',
                            'Júl', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'])   # Popisky mesiacov
        plt.tight_layout()                                 # Automatické prispôsobenie okrajov
        plt.savefig(os.path.join(output_dir, 'histogram_chybajucich_dni.png'), dpi=150)   # Uloženie
        plt.close()                                        # Zatvoríme graf, uvoľníme pamäť
        print(f"  Graf uložený: histogram_chybajucich_dni.png")   # Potvrdenie

    # --- Graf: Počet vzoriek za každý deň (histogram distribúcie) ---
    vzorky_za_den = df.groupby('Date').size()   # Spočítanie vzoriek pre každý deň

    fig, ax = plt.subplots(figsize=(12, 5))   # Nová figúra
    ax.hist(vzorky_za_den.values, bins=30, color='steelblue', edgecolor='black')   # Histogram počtov
    ax.axvline(vzorky_za_den.mean(), color='red', linestyle='--',
               label=f'Priemer: {vzorky_za_den.mean():.1f}')   # Zvislá čiara priemeru
    ax.set_xlabel('Počet vzoriek za deň')    # Popis osi x
    ax.set_ylabel('Počet dní')               # Popis osi y
    ax.set_title('Distribúcia počtu vzoriek na deň')   # Nadpis grafu
    ax.legend()                              # Zobrazenie legendy
    plt.tight_layout()                       # Prispôsobenie okrajov
    plt.savefig(os.path.join(output_dir, 'histogram_vzoriek_za_den.png'), dpi=150)   # Uloženie
    plt.close()                              # Zatvoríme graf
    print(f"  Graf uložený: histogram_vzoriek_za_den.png")              # Potvrdenie
    print(f"  Priemerný počet vzoriek za deň: {vzorky_za_den.mean():.1f}")   # Štatistika
    print(f"  Min / Max vzoriek za deň:       {vzorky_za_den.min()} / {vzorky_za_den.max()}")

    # --- Graf: Časový priebeh počtu vzoriek za deň ---
    vzorky_ts = vzorky_za_den.copy()                      # Kópia série pre časový graf
    vzorky_ts.index = pd.to_datetime(vzorky_ts.index)     # Konverzia indexu na datetime

    fig, ax = plt.subplots(figsize=(16, 4))               # Nová figúra (široký formát)
    ax.plot(vzorky_ts.index, vzorky_ts.values, color='steelblue', linewidth=0.8)   # Čiarový graf
    ax.set_xlabel('Dátum')                                # Popis osi x
    ax.set_ylabel('Počet vzoriek za deň')                 # Popis osi y
    ax.set_title('Počet vzoriek za každý deň v čase')     # Nadpis grafu
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))       # Formát dátumu: Jan 2018
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))       # Značka každé 2 mesiace
    plt.xticks(rotation=45)                               # Popisky osi x natočené o 45°
    plt.tight_layout()                                    # Prispôsobenie okrajov
    plt.savefig(os.path.join(output_dir, 'vzorky_za_den_casovy_priebeh.png'), dpi=150)   # Uloženie
    plt.close()                                           # Zatvoríme graf
    print(f"  Graf uložený: vzorky_za_den_casovy_priebeh.png")        # Potvrdenie

    return chybajuce_dni   # Vrátime zoznam chýbajúcich dní (môže využiť HTML report)


# --------------------------------------------------------------
# 2. Chýbajúce vzorky (NaN hodnoty)
# --------------------------------------------------------------

def analyza_chybajucich_vzoriek(df):
    """
    Spočíta chýbajúce (NaN) hodnoty pre každý numerický stĺpec.
    """
    print("\n=== ANALÝZA CHÝBAJÚCICH VZORIEK (NaN HODNOTY) ===")   # Nadpis sekcie

    # Zoznam stĺpcov, ktoré chceme skontrolovať (všetky numerické merané stĺpce)
    stlpce = ['Irradiance', 'IrradianceNotCompensated',
              'BodyTemperature', 'RelativeHumidity', 'HumidityTemp',
              'Pressure', 'PressureAvg', 'PressureTemp', 'PressureTempAvg',
              'TiltAngle', 'TiltAngleAvg', 'FanSpeed', 'HeaterCurrent', 'FanCurrent',
              'SunLatitude', 'SunLongitude', 'SunAzimuth', 'SunZenith']

    # Výpočet počtu NaN hodnôt pre každý stĺpec
    chybajuce = df[stlpce].isnull().sum()              # Počet NaN v každom stĺpci
    print(f"  Celkový počet vzoriek: {len(df)}")        # Celkový počet riadkov
    for stlpec, pocet in chybajuce.items():             # Iterujeme cez každý stĺpec
        # Výpis aj stĺpcov bez chýb, aby bolo jasné, že boli skontrolované
        stav = f"{pocet} ({100 * pocet / len(df):.2f}%)" if pocet > 0 else "0 chýbajúcich"
        print(f"    {stlpec:30s}: {stav}")              # Vypíšeme stav pre daný stĺpec

    # Celkový počet riadkov s aspoň jednou NaN hodnotou
    riadky_s_nan = df[stlpce].isnull().any(axis=1).sum()   # any(axis=1) = True ak aspoň jedna NaN v riadku
    print(f"  Riadkov s aspoň jednou chýbajúcou hodnotou: {riadky_s_nan}")   # Výpis výsledku


# --------------------------------------------------------------
# 3. Poškodené vzorky (fyzikálne medze)
# --------------------------------------------------------------

def detekcia_poskodennych_vzoriek(df):
    """
    Detekuje vzorky, kde hodnota je mimo fyzikálne možného rozsahu.
    Vracia: boolovskú masku poškodených riadkov a pravdepodobnosť chybnej vzorky.
    """
    print("\n=== DETEKCIA POŠKODENÝCH VZORIEK (FYZIKÁLNE MEDZE) ===")   # Nadpis sekcie

    # Začíname s maskou, kde všetko je False (žiadna vzorka nie je poškodená)
    poskodene_maska = pd.Series(False, index=df.index)   # Inicializácia prázdnej masky

    for stlpec, (min_val, max_val) in FYZIKALNE_MEDZE.items():   # Pre každý parameter
        if stlpec not in df.columns:   # Preskočíme ak stĺpec v dátach neexistuje
            continue
        # Vzorky mimo medzi, ktoré zároveň nie sú NaN
        mimo_medze = ((df[stlpec] < min_val) | (df[stlpec] > max_val)) & df[stlpec].notna()
        pocet = mimo_medze.sum()   # Počet poškodených vzoriek v tomto stĺpci
        stav = f"{pocet} ({100 * pocet / len(df):.3f}%)" if pocet > 0 else "0 poškodených"
        print(f"    {stlpec:30s}: {stav}")   # Výpis stavu pre daný stĺpec
        # Pridáme do celkovej masky (OR – stačí jedna poškodená hodnota v riadku)
        poskodene_maska = poskodene_maska | mimo_medze   # Aktualizujeme celkovú masku

    celkove = poskodene_maska.sum()            # Celkový počet poškodených riadkov
    pravdepodobnost = celkove / len(df)        # Pravdepodobnosť chybnej vzorky

    print(f"\n  Celkový počet poškodených vzoriek: {celkove}")                                     # Výpis
    print(f"  Pravdepodobnosť chybnej vzorky:    {pravdepodobnost:.6f} ({100 * pravdepodobnost:.4f}%)")

    return poskodene_maska, pravdepodobnost   # Vrátime masku a pravdepodobnosť


# --------------------------------------------------------------
# 4. Histogramy dĺžok za sebou idúcich chýb
# --------------------------------------------------------------

def histogramy_dlzok_chyb(df, poskodene_maska, output_dir):
    """
    Pre chýbajúce (NaN) aj poškodené vzorky vypočíta dĺžky súvislých skupín
    a zobrazí ich ako histogram.
    """
    print("\n=== HISTOGRAMY DĹŽOK ZA SEBOU IDÚCICH CHÝB ===")   # Nadpis sekcie

    # Spustíme analýzu pre oba typy chýb
    chyby = {
        'chybajucich_nan': (df['Irradiance'].isnull(), 'chýbajúcich (NaN) vzoriek'),   # Maska NaN hodnôt
        'poskodennych':    (poskodene_maska,            'poškodených vzoriek'),          # Maska poškodených
    }

    for nazov_suboru, (maska, popis) in chyby.items():   # Pre každý typ chyby
        # Výpočet dĺžok súvislých skupín True hodnôt
        skupiny = []   # Zoznam dĺžok nájdených skupín
        dlzka = 0      # Aktuálna dĺžka práve sledovanej skupiny
        for hodnota in maska:         # Prechádzame masku vzorku po vzorke
            if hodnota:               # Ak je vzorka chybná
                dlzka += 1            # Rozširujeme aktuálnu skupinu
            else:                     # Ak je vzorka v poriadku
                if dlzka > 0:         # Ak sme práve ukončili skupinu
                    skupiny.append(dlzka)   # Uložíme dĺžku skončenej skupiny
                dlzka = 0             # Resetujeme počítadlo
        if dlzka > 0:                 # Ak séria končí chybou
            skupiny.append(dlzka)     # Uložíme poslednú skupinu

        print(f"  --- {popis} ---")   # Oddeľovač pre každý typ chyby
        if skupiny:                   # Ak boli nájdené nejaké skupiny
            print(f"    Počet skupín:         {len(skupiny)}")       # Počet skupín
            print(f"    Max dĺžka skupiny:    {max(skupiny)}")       # Najdlhšia skupina
            print(f"    Priemerná dĺžka:      {np.mean(skupiny):.2f}")   # Priemerná dĺžka
            fig, ax = plt.subplots(figsize=(10, 5))   # Nová figúra pre histogram
            # Half-integer hrany: hodnota 1 → bin [0.5, 1.5], hodnota 2 → [1.5, 2.5], ...
            # Tak je každé celé číslo vycentrované vo vlastnom stĺpci
            max_dlzka = min(max(skupiny), 30)                           # Obmedzíme na max 30 stĺpcov
            bin_edges = [i + 0.5 for i in range(0, max_dlzka + 1)]     # Hrany binov: [0.5, 1.5, 2.5, ...]
            ax.hist(skupiny, bins=bin_edges, color='coral', edgecolor='black')   # Histogram dĺžok
            ax.set_xticks(range(1, max_dlzka + 1))   # Popisky osi x len na celých číslach
            ax.set_xlabel('Dĺžka súvislej skupiny (počet vzoriek)')   # Popis osi x
            ax.set_ylabel('Počet skupín')                              # Popis osi y
            ax.set_title(f'Histogram dĺžok za sebou idúcich {popis}') # Nadpis grafu
            plt.tight_layout()                                         # Prispôsobenie okrajov
            plt.savefig(os.path.join(output_dir, f'histogram_dlzok_{nazov_suboru}.png'), dpi=150)   # Uloženie
            plt.close()                                                # Zatvoríme graf
            print(f"    Graf uložený: histogram_dlzok_{nazov_suboru}.png")   # Potvrdenie
        else:
            print(f"    Žiadne za sebou idúce {popis} neboli nájdené.")   # Informácia ak nič nenájdené


# --------------------------------------------------------------
# 5. Čistenie dát – odstránenie chybných riadkov a uloženie CSV
# --------------------------------------------------------------

def vycisti_a_uloz(df, output_dir):
    """
    Odstráni všetky riadky s chýbajúcimi (NaN) alebo fyzikálne neplatnými hodnotami
    a uloží výsledok ako cleaned.csv do výstupného adresára (OUTPUT).

    Kritériá mazania (OR – stačí jedna podmienka):
      1. Aspoň jedna NaN hodnota v sledovaných stĺpcoch
      2. Aspoň jedna hodnota mimo fyzikálnych medzi (FYZIKALNE_MEDZE)

    Vracia: vyčistený DataFrame
    """
    print("\n=== ČISTENIE DÁT A ULOŽENIE CLEANED.CSV ===")   # Nadpis sekcie pre čitateľnosť výstupu

    sledovane = list(FYZIKALNE_MEDZE.keys())   # Zoznam všetkých sledovaných parametrov (kľúče zo slovníka fyzikálnych medzi)

    # --- Maska chýbajúcich hodnôt (NaN) ---
    sledovane_pritomne = [s for s in sledovane if s in df.columns]   # Len stĺpce, ktoré reálne existujú v DataFrame (obrana pred chýbajúcimi stĺpcami)
    maska_nan = df[sledovane_pritomne].isnull().any(axis=1)   # True ak má riadok aspoň jednu NaN; .any(axis=1) = porovnanie po riadkoch

    # --- Maska hodnôt mimo fyzikálnych medzi ---
    maska_poskodene = pd.Series(False, index=df.index)   # Inicializácia masky poškodených riadkov: začíname s False pre každý riadok
    for stlpec, (min_val, max_val) in FYZIKALNE_MEDZE.items():   # Iterujeme: stlpec = názov parametra, min_val/max_val = povolený rozsah
        if stlpec not in df.columns:   # Ak tento stĺpec v dátach neexistuje (napr. chýba v CSV)
            continue                   # Preskočíme ho a pokračujeme na ďalší parameter
        mimo = ((df[stlpec] < min_val) | (df[stlpec] > max_val)) & df[stlpec].notna()   # True: hodnota je pod min ALEBO nad max, a zároveň nie je NaN
        maska_poskodene = maska_poskodene | mimo   # OR: riadok je poškodený ak má aspoň jednu hodnotu mimo medzi v akomkoľvek stĺpci

    # --- Kombinovaná maska a štatistiky ---
    maska_chybnych = maska_nan | maska_poskodene   # OR: riadok je chybný ak má NaN (maska_nan) ALEBO hodnotu mimo medzi (maska_poskodene)
    print(f"  Riadkov pred čistením:                     {len(df):,}")                   # Výpis: celkový počet riadkov pred čistením (:, = formátovanie s oddeľovačom tisícov)
    print(f"  Riadky s chýbajúcimi hodnotami (NaN):      {maska_nan.sum():,}")           # Výpis: počet riadkov s aspoň jednou NaN hodnotou
    print(f"  Riadky s hodnotami mimo fyzikálnych medzi: {maska_poskodene.sum():,}")     # Výpis: počet riadkov s hodnotou mimo fyzikálnych medzi
    print(f"  Celkovo chybných riadkov (union):           {maska_chybnych.sum():,}")     # Výpis: celkový počet chybných riadkov (OR oboch masiek; môže byť menej než súčet)

    # --- Odstránenie chybných riadkov ---
    df_clean = df[~maska_chybnych].reset_index(drop=True)   # ~ = negácia masky; vyberieme len dobré riadky; reset_index = nové poradové čísla od 0
    print(f"  Riadkov po čistení:                        {len(df_clean):,}")   # Výpis: počet zostávajúcich riadkov po čistení

    # --- Uloženie do cleaned.csv do výstupného adresára (OUTPUT) ---
    cesta_cleaned = os.path.join(output_dir, 'cleaned.csv')   # Zostavíme cestu: output_dir + 'cleaned.csv' (bezpečné na Windows aj Linux)

    # Ukladáme len pôvodné stĺpce z CSV (bez pomocných stĺpcov Date, Hour, Month, DayOfYear, ktoré sme pridali pri načítaní)
    pomocne_stlpce = ['Date', 'Hour', 'Month', 'DayOfYear']   # Zoznam pomocných stĺpcov pridaných funkciou nacitaj_data
    stlpce_csv = [s for s in df_clean.columns if s not in pomocne_stlpce]   # Všetky stĺpce okrem pomocných = originálne stĺpce z CSV
    df_clean[stlpce_csv].to_csv(cesta_cleaned, index=False)   # Uložíme do CSV; index=False = neukladáme číselný index riadkov

    print(f"  Uložené: {cesta_cleaned}  ({os.path.getsize(cesta_cleaned) / 1024:.0f} kB)")   # Potvrdenie uloženia s veľkosťou súboru v kB
    return df_clean   # Vrátime vyčistený DataFrame pre použitie v ďalšej analýze (ročná analýza, korelácie)
