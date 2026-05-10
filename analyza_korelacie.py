# ==============================================================
# MODUL: Korelačná analýza a súhrnné štatistiky parametrov
# ==============================================================

import os                        # Práca so súborovým systémom (ukladanie grafov)
import torch                     # PyTorch – overenie výpočtov pomocou tenzorov (GPU/CPU)
import matplotlib.pyplot as plt  # Kreslenie grafov (heatmapa, scatter ploty)
import seaborn as sns            # Nadstavba matplotlib – heatmapa korelačnej matice


# Slovník slovenských názvov stĺpcov pre popisky osí a nadpisy grafov.
# Kľúč = anglický názov stĺpca v CSV, hodnota = slovenský popis s fyzikálnou jednotkou.
NAZVY_SK = {
    'Irradiance':               'Žiarenie [W/m²]',               # Merané slnečné žiarenie v W na m²
    'IrradianceNotCompensated': 'Žiarenie nekompenz. [W/m²]',    # Žiarenie bez teplotnej kompenzácie
    'BodyTemperature':          'Teplota senzora [°C]',           # Teplota tela pyranometra (krytu senzora)
    'RelativeHumidity':         'Relatívna vlhkosť [%]',          # Vlhkosť vzduchu v percentách (0–100)
    'HumidityTemp':             'Teplota vlhkomera [°C]',         # Teplota snímača vlhkosti
    'Pressure':                 'Atmosf. tlak [hPa]',             # Okamžitý atmosferický tlak v hektopascaloch
    'PressureAvg':              'Atmosf. tlak – priemer [hPa]',   # Priemer atmosferického tlaku za interval
    'PressureTemp':             'Teplota tlakomera [°C]',         # Teplota interného senzora tlakometra
    'PressureTempAvg':          'Teplota tlakomera – priemer [°C]', # Priemer teploty tlakometra
    'TiltAngle':                'Uhol naklonenia [°]',            # Uhol naklonenia pyranometra od horizontály
    'TiltAngleAvg':             'Uhol naklonenia – priemer [°]',  # Priemer uhla naklonenia za interval
    'FanSpeed':                 'Rýchlosť ventilátora [RPM]',     # Otáčky ventilátora v otáčkach za minútu
    'HeaterCurrent':            'Prúd ohrievača [A]',             # Elektrický prúd ohrievača ventilátora v ampéroch
    'FanCurrent':               'Prúd ventilátora [A]',           # Elektrický prúd motora ventilátora v ampéroch
    'SunLatitude':              'Zem. šírka stanice [°]',         # Zemepisná šírka meracej stanice (WGS84)
    'SunLongitude':             'Zem. dĺžka stanice [°]',         # Zemepisná dĺžka meracej stanice (WGS84)
    'SunAzimuth':               'Azimut slnka [°]',               # Kompasový smer slnka (0° = sever, 180° = juh)
    'SunZenith':                'Zenitový uhol slnka [°]',        # Uhol slnka od nadhlavníka (0° = kolmé, 90° = horizont)
}


# Zoznam parametrov pre korelačnú analýzu.
# SunLatitude a SunLongitude sú vynechané – sú konštantné (rovnaká hodnota v celom datasete),
# korelácia konštanty s čímkoľvek je matematicky nedefinovaná (štandardná odchýlka = 0, delenie nulou).
PARAMETRE = [
    'Irradiance', 'IrradianceNotCompensated',               # Merania slnečného žiarenia
    'BodyTemperature', 'RelativeHumidity', 'HumidityTemp',  # Teplota senzora a vlhkosť vzduchu
    'Pressure', 'PressureAvg', 'PressureTemp', 'PressureTempAvg',  # Tlak a teplota tlakometra
    'TiltAngle', 'TiltAngleAvg', 'FanSpeed',                # Mechanické veličiny pyranometra
    'HeaterCurrent', 'FanCurrent',                          # Elektrické prúdy ohrievača a ventilátora
    'SunAzimuth', 'SunZenith',                              # Geometrická poloha slnka na oblohe
]


# --------------------------------------------------------------
# 1. Korelačná analýza
# --------------------------------------------------------------

def korelacie_parametrov(df, output_dir):
    """
    Vypočíta Pearsonove korelačné koeficienty medzi všetkými parametrami.
    Overí výsledky pomocou PyTorch. Uloží heatmapu korelačnej matice a scatter ploty.

    Parametre:
        df         – pandas DataFrame s meteorologickými dátami (výstup nacitaj_data)
        output_dir – reťazec, cesta k adresáru pre ukladanie grafov (napr. 'output/')
    """
    print("\n=== KORELAČNÁ ANALÝZA PARAMETROV ===")   # Nadpis sekcie pre čitateľnosť výstupu

    # Výpočet Pearsonovej korelačnej matice pomocou pandas.
    # Výsledok je DataFrame rozmerov N×N (N = počet parametrov).
    # Každá bunka [i, j] obsahuje Pearsonov r medzi parametrami i a j.
    # Hlavná uhlopriečka je vždy 1.0 (parameter je dokonale korelovaný sám so sebou).
    kor_matica = df[PARAMETRE].corr(method='pearson')   # method='pearson' = štandardná lineárna korelácia

    # Výpis korelácie každého parametra s Irradiance (žiarením) zoradený podľa absolútnej hodnoty.
    # Najsilnejšie korelácie (kladné aj záporné) sú na vrchu zoznamu.
    print("  Korelácie ostatných parametrov s Irradiance (žiarením):")   # Nadpis výpisu
    kor_s_irr = kor_matica['Irradiance'].drop('Irradiance').sort_values(key=abs, ascending=False)   # Vyberieme stĺpec Irradiance, odoberieme vlastnú koreláciu (=1.0), zoradíme
    for param, r in kor_s_irr.items():   # Iterujeme: param = názov parametra (str), r = korelačný koeficient (float)
        sila = "silná" if abs(r) > 0.5 else ("stredná" if abs(r) > 0.3 else "slabá")   # Slovná klasifikácia sily (|r|>0.5=silná, |r|>0.3=stredná)
        smer = "pozitívna" if r > 0 else "negatívna"                                    # Smer: kladný r = oba rastú spolu, záporný = opačný trend
        print(f"    {param:30s}: r = {r:+.4f}  ({sila} {smer})")                        # Zarovnaný výpis: +znamienko vždy zobrazené

    # --- Overenie korelačného výpočtu pomocou PyTorch tenzorov ---
    # Počítame ten istý Pearsonov r pre pár Irradiance/SunZenith priamo cez tenzorové operácie.
    # Slúži ako overenie, že pandas a PyTorch dávajú rovnaký výsledok.
    df_c = df[PARAMETRE].dropna()   # Odstránime riadky s akoukoľvek NaN (PyTorch tensor nepodporuje NaN hodnoty)
    if len(df_c) > 1:               # Pokračujeme len ak máme aspoň 2 platné riadky (1 riadok → štandardná odchýlka = 0)
        T = torch.tensor(df_c.values, dtype=torch.float32)   # Konverzia celej tabuľky (numpy array) na 2D float32 tenzor [vzorky × parametre]

        # Vyberieme dva stĺpce pre overenie korelácie
        irr = T[:, 0]                                   # Všetky riadky, stĺpec 0 = Irradiance (prvý v zozname PARAMETRE)
        zen = T[:, PARAMETRE.index('SunZenith')]         # Všetky riadky, stĺpec SunZenith – index hľadáme v zozname PARAMETRE

        # Výpočet Pearsonovho r manuálne:
        # r = Σ[(xi - x̄)(yi - ȳ)] / (||x - x̄|| · ||y - ȳ||)
        irr_c = irr - irr.mean()                         # Odchýlky Irradiance od priemeru (centrovanie vektora)
        zen_c = zen - zen.mean()                         # Odchýlky SunZenith od priemeru (centrovanie vektora)
        r = (irr_c * zen_c).sum() / (irr_c.norm() * zen_c.norm())   # Bodový súčin / súčin noriem = Pearsonov r

        print(f"\n  PyTorch – Irradiance vs SunZenith: {r.item():.4f}")   # .item() = konverzia jednoprvkového tenzora na Python float

    # --- Graf: Heatmapa korelačnej matice ---
    # Premenujeme indexy a stĺpce matice zo slovenských názvov s jednotkami na kratšie verzie bez jednotiek.
    # Tým sa nadpisy zmestia do buniek heatmapy bez prekrývania.
    nazvy_kratke = {k: v.split(' [')[0] for k, v in NAZVY_SK.items()}   # split(' [')[0] = odoberieme časť za '[' (jednotku)
    kor_matica_sk = kor_matica.rename(index=nazvy_kratke, columns=nazvy_kratke)   # Premenujeme riadkové aj stĺpcové indexy matice

    fig, ax = plt.subplots(figsize=(14, 11))   # Nová figúra 14×11 palcov – väčšia kvôli dlhším slovenským názvom
    sns.heatmap(kor_matica_sk,
                annot=True,        # Zobrazí číselnú hodnotu korelácie v každej bunke
                fmt='.2f',         # Formát čísel: 2 desatinné miesta (napr. "0.87")
                cmap='coolwarm',   # Farebnáschéma: modrá = silná negatívna, červená = silná pozitívna korelácia
                center=0,          # Stred (biela farba) = 0 (žiadna korelácia)
                vmin=-1, vmax=1,   # Rozsah farebnej škály: plný možný rozsah Pearsonovho r (-1 až +1)
                square=True,       # Každá bunka má rovnakú šírku aj výšku (štvorcová mriežka)
                linewidths=0.5,    # Hrúbka čiar medzi bunkami (tenká oddeľovacia mriežka)
                ax=ax)             # Kreslíme do nášho subplotu ax, nie do automaticky vytvoreného okna
    ax.set_title('Korelačná matica meteorologických parametrov (Pearson)')   # Nadpis grafu
    plt.tight_layout()             # Automatické prispôsobenie vonkajších okrajov, aby sa popisky nerezali
    plt.savefig(os.path.join(output_dir, 'heatmapa_korelacie.png'), dpi=150)   # Uloženie ako PNG s rozlíšením 150 DPI
    plt.close()                    # Zatvoríme figúru a uvoľníme pamäť (dôležité pri generovaní veľa grafov)
    print(f"  Graf uložený: heatmapa_korelacie.png")   # Potvrdenie uloženia pre používateľa

    # --- Graf: Scatter ploty vybraných párov parametrov ---
    # Zobrazujeme 12 pevne zvolených párov usporiadaných do trojuholníkov:
    #   Trojuholník 1: Žiarenie / Atmosf. tlak / Teplota senzora     (sezónny efekt)
    #   Trojuholník 2: Žiarenie / Zenitový uhol slnka / Azimut slnka (geometria slnka)
    #   Trojuholník 3: Atmosf. tlak / Rýchlosť ventilátora / Teplota tlakomera
    #   Trojuholník 4: Teploty troch senzorových modulov (Body, Pressure, Humidity)
    PEVNE_PARY = [
        ('Irradiance', 'Pressure'),          # Pár 1:  Žiarenie vs Atmosf. tlak (sezónna závislosť)
        ('Irradiance', 'BodyTemperature'),   # Pár 2:  Žiarenie vs Teplota senzora (slnečné dni = teplejšie)
        ('Pressure',   'BodyTemperature'),   # Pár 3:  Atmosf. tlak vs Teplota senzora (obe korelujú so sezónou)

        ('Irradiance', 'SunZenith'),         # Pár 4:  Žiarenie vs Zenitový uhol (vyšší uhol = menej žiarenia)
        ('Irradiance', 'SunAzimuth'),        # Pár 5:  Žiarenie vs Azimut slnka (smer závisí od dennej doby)
        ('SunZenith',  'SunAzimuth'),        # Pár 6:  Zenitový uhol vs Azimut (oba popisujú polohu slnka)

        ('Pressure',        'FanSpeed'),        # Pár 7:  Atmosf. tlak vs Rýchlosť ventilátora
        ('Pressure',        'PressureTemp'),    # Pár 8:  Atmosf. tlak vs Teplota tlakometra
        ('FanSpeed',        'PressureTemp'),    # Pár 9:  Rýchlosť ventilátora vs Teplota tlakometra

        ('BodyTemperature', 'PressureTemp'),    # Pár 10: Teplota senzora vs Teplota tlakometra (obe merajú okolitú teplotu)
        ('BodyTemperature', 'HumidityTemp'),    # Pár 11: Teplota senzora vs Teplota vlhkomera
        ('PressureTemp',    'HumidityTemp'),    # Pár 12: Teplota tlakometra vs Teplota vlhkomera
    ]

    # Vytvoríme zoznam tuplov (absolútna_korelácia, parameter1, parameter2) pre každý pár.
    # Absolútna hodnota sa použije pre prípadné zoradenie; znamienko zachovávame separátne.
    top_pary = [(abs(kor_matica.loc[p1, p2]), p1, p2) for p1, p2 in PEVNE_PARY]   # List comprehension: pre každý pár vyhľadáme r v matici

    print(f"\n  Vybrané páry pre scatter ploty ({len(top_pary)} celkovo):")   # Informácia o počte párov
    for r_val, p1, p2 in top_pary:   # Iterujeme: r_val = |r|, p1 a p2 = názvy parametrov
        smer = "pozitívna" if kor_matica.loc[p1, p2] > 0 else "negatívna"   # Smer korelácie pre výpis
        print(f"    {p1} vs {p2}: r = {kor_matica.loc[p1, p2]:+.4f}  ({smer})")   # Výpis s kladným/záporným znamienkom

    # Nastavenie mriežky subplotov pre scatter ploty
    n = len(top_pary)                            # Celkový počet párov = počet scatter plotov
    ncols = 3                                    # Pevný počet stĺpcov v mriežke grafov
    nrows = (n + ncols - 1) // ncols             # Počet riadkov – stropné delenie (ceil): napr. 12 párov / 3 stĺpce = 4 riadky
    paleta = ['darkorange', 'crimson', 'seagreen', 'steelblue',
              'mediumpurple', 'saddlebrown', 'teal', 'tomato', 'slategray',
              'goldenrod', 'indianred', 'cadetblue']   # 12 rôznych farieb – každý scatter plot má svoju farbu
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))   # Mriežka subplotov; výška 5 palcov × počet riadkov
    axes = axes.flatten()   # Sploštíme 2D numpy pole subplotov (nrows×ncols) na 1D zoznam pre jednoduchý prístup indexom

    for ax, (r_val, p1, p2), farba in zip(axes, top_pary, paleta):   # zip spáruje: subplot, pár parametrov, farba
        df_sc = df[[p1, p2]].dropna()                          # Vyberieme len dva stĺpce; dropna() odstráni riadky kde aspoň jedna hodnota chýba
        ax.scatter(df_sc[p2], df_sc[p1],
                   alpha=0.1, s=2, color=farba)                # Scatter: p2 na osi x, p1 na osi y; alpha=0.1 = priesvitnosť redukuje cluttering; s=2 = malé body
        ax.set_xlabel(NAZVY_SK.get(p2, p2))                    # Popis osi x: slovenský názov, fallback = anglický ak nie je v slovníku
        ax.set_ylabel(NAZVY_SK.get(p1, p1))                    # Popis osi y: slovenský názov, fallback = anglický ak nie je v slovníku
        r_sign = kor_matica.loc[p1, p2]                        # Korelačný koeficient so znamienkom (nie absolútna hodnota) pre nadpis
        nazov1 = NAZVY_SK.get(p1, p1).split(' [')[0]          # Skrátený slovenský názov 1 – odoberieme jednotku v zátvorkách
        nazov2 = NAZVY_SK.get(p2, p2).split(' [')[0]          # Skrátený slovenský názov 2 – odoberieme jednotku v zátvorkách
        ax.set_title(f'{nazov1}\nvs {nazov2}\nr = {r_sign:+.3f}')   # Trojriadkový nadpis: param1, "vs param2", hodnota r
        ax.grid(True, alpha=0.3)                               # Jemná priesvitná mriežka pre lepšiu čitateľnosť

    # Skryjeme prázdne subploty (vzniknú ak počet párov nie je násobkom ncols)
    for ax in axes[n:]:          # axes[n:] = všetky subploty za posledným skutočným grafom
        ax.set_visible(False)    # Skryjeme prázdny subplot (inak by sa zobrazoval prázdny rám)

    plt.tight_layout()   # Automatické prispôsobenie okrajov a medzier medzi grafmi
    plt.savefig(os.path.join(output_dir, 'scatter_korelacie.png'), dpi=150)   # Uloženie PNG s rozlíšením 150 DPI
    plt.close()          # Zatvoríme figúru a uvoľníme pamäť
    print(f"  Graf uložený: scatter_korelacie.png")   # Potvrdenie uloženia pre používateľa


# --------------------------------------------------------------
# 2. Súhrnné štatistiky
# --------------------------------------------------------------

def suhrne_statistiky(df, output_dir):
    """
    Vypíše základné opisné štatistiky (count, min, max, priemer, std, percentily)
    pre všetky parametre. Uloží ich do CSV súboru. Overí výsledky pomocou PyTorch.

    Parametre:
        df         – pandas DataFrame s meteorologickými dátami (výstup nacitaj_data)
        output_dir – reťazec, cesta k adresáru pre ukladanie výstupov (napr. 'output/')
    """
    print("\n=== SÚHRNNÉ ŠTATISTIKY ===")   # Nadpis sekcie pre čitateľnosť výstupu

    # pandas .describe() automaticky vypočíta tieto štatistiky pre každý stĺpec:
    # count (počet platných hodnôt), mean (priemer), std (smerodajná odchýlka),
    # min, 25% percentil, 50% percentil (medián), 75% percentil, max
    stats = df[PARAMETRE].describe()    # Výsledok je DataFrame: riadky = štatistiky, stĺpce = parametre
    print(stats.to_string())            # to_string() vypíše celú tabuľku bez skrátenia (pandas inak skracuje pri veľa stĺpcoch)

    # Uloženie tabuľky štatistík do CSV súboru pre použitie v HTML reporte
    stats.to_csv(os.path.join(output_dir, 'suhrne_statistiky.csv'))   # os.path.join = bezpečné spájanie ciest (funguje na Windows aj Linux)
    print(f"\n  Uložené do: suhrne_statistiky.csv")   # Potvrdenie uloženia pre používateľa

    # --- PyTorch overenie štatistík ---
    # Overíme, že pandas štatistiky sa zhodujú s výpočtami cez tenzorové operácie.
    # Toto je pedagogická ukážka toho, že PyTorch dokáže robiť rovnaké výpočty ako pandas.
    print("\n  PyTorch – min | max | priemer | std:")   # Nadpis sekcie s PyTorch výsledkami
    df_c = df[PARAMETRE].dropna()                                       # Odstránime riadky s NaN (PyTorch tenzor nepodporuje NaN hodnoty)
    T = torch.tensor(df_c.values, dtype=torch.float32)                  # Konverzia pandas DataFrame na 2D PyTorch tenzor [riadky × stĺpce] typu float32
    for i, param in enumerate(PARAMETRE):                               # enumerate dáva dvojicu (index, hodnota): i = číslo stĺpca, param = jeho názov
        col = T[:, i]                                                    # Vyberieme i-tý stĺpec tenzora: ':' = všetky riadky, 'i' = i-tý stĺpec
        print(f"    {param:30s}: "
              f"min={col.min().item():9.3f} | max={col.max().item():9.3f} | "
              f"mean={col.mean().item():9.3f} | std={col.std().item():9.3f}")   # .item() konvertuje jednoprvkový PyTorch tenzor na Python float pre výpis
