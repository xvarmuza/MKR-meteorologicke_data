# ==============================================================
# MODUL: Generovanie HTML reportu zo všetkých analýz
# ==============================================================

import os            # Práca so súborovým systémom (cesty, kontrola existencie súborov, veľkosť)
import io            # In-memory textový buffer (StringIO) pre zachytávanie výstupu funkcií
import base64        # Kódovanie binárnych obrázkov do base64 textu pre vloženie priamo do HTML
import contextlib    # Nástroje pre správu kontextu – redirect_stdout na presmerovanie výstupu
import pandas as pd  # Načítanie CSV so štatistikami a konverzia DataFrame na HTML tabuľku

# Naše analytické moduly – importujeme funkcie, ktoré spustíme v reporte
from nacitanie_dat     import nacitaj_data                    # Načítanie a príprava DataFrame zo CSV
from analyza_chyb      import (analyza_chybajucich_dni,       # Analýza chýbajúcich dní
                                analyza_chybajucich_vzoriek,  # Počítanie NaN hodnôt
                                detekcia_poskodennych_vzoriek, # Detekcia hodnôt mimo fyzikálnych medzi
                                histogramy_dlzok_chyb)         # Histogramy dĺžok skupín chýb
from analyza_ziarenia  import rocna_analyza_ziarenia, denna_analyza_pre_mesiace   # Analýzy žiarenia
from analyza_korelacie import korelacie_parametrov, suhrne_statistiky             # Korelácie a štatistiky

# Fyzikálne medze importujeme z analyza_chyb – jediné miesto definície (Single Source of Truth)
from analyza_chyb import FYZIKALNE_MEDZE   # Slovník: stĺpec → (min, max) povolený rozsah

# Stĺpce zobrazované v tabuľke poškodených vzoriek v HTML reporte:
# DateTime (časová značka) + všetky stĺpce, pre ktoré máme fyzikálne medze
ZOBRAZIT_STLPCE = ['DateTime'] + list(FYZIKALNE_MEDZE.keys())   # list() konvertuje dict_keys na zoznam


# ==============================================================
# POMOCNÉ FUNKCIE
# ==============================================================

def zachyt_vystup(funkcia, *args, **kwargs):
    """
    Spustí ľubovoľnú funkciu a zachytí celý jej textový výstup (print) do reťazca.
    Tým sa výpisy nevypíšu do terminálu, ale uložia sa do premennej pre HTML report.

    Parametre:
        funkcia  – funkcia, ktorú chceme spustiť (napr. suhrne_statistiky)
        *args    – pozicinálne argumenty, ktoré sa odovzdajú funkcii
        **kwargs – kľúčové argumenty, ktoré sa odovzdajú funkcii

    Vracia: dvojicu (návratová_hodnota_funkcie, zachytený_text_ako_string)
    """
    buffer = io.StringIO()                          # Vytvoríme prázdny in-memory textový buffer (ako súbor v pamäti)
    with contextlib.redirect_stdout(buffer):        # Presmerujeme štandardný výstup (print) do buffra namiesto terminálu
        vysledok = funkcia(*args, **kwargs)         # Spustíme funkciu – všetky jej print() sa zapíšu do buffra
    return vysledok, buffer.getvalue()              # Vrátime: návratovú hodnotu funkcie + celý zachytený text


def tabulka_poskodennych(df):
    """
    Nájde všetky poškodené vzorky (hodnoty mimo fyzikálnych medzi) v DataFrame
    a vráti HTML reťazec s tabuľkou. Chybné bunky sú červeno zvýraznené.
    Pri podržaní myši (tooltip) sa zobrazí povolený rozsah hodnôt.

    Parametre:
        df – pandas DataFrame s meteorologickými dátami (celý, nefiltrovaný)

    Vracia: HTML reťazec (str) s tabuľkou alebo správou o absencii chýb
    """
    # Inicializácia celkovej masky (True = riadok je poškodený) a slovníka masiek pre každý stĺpec.
    riadky_maska  = pd.Series(False, index=df.index)   # Začíname: žiadny riadok nie je poškodený (všetko False)
    stlpce_chyby  = {s: pd.Series(False, index=df.index) for s in FYZIKALNE_MEDZE}   # Pre každý stĺpec vlastná maska chýb

    for stlpec, (mn, mx) in FYZIKALNE_MEDZE.items():   # Iterujeme: stlpec = názov stĺpca, mn = minimum, mx = maximum
        if stlpec not in df.columns:   # Ak stĺpec v dátach neexistuje (napr. chýba v CSV), preskočíme
            continue                   # Pokračujeme na ďalší parameter v slovníku
        mimo = ((df[stlpec] < mn) | (df[stlpec] > mx)) & df[stlpec].notna()   # Hodnota je mimo medzi A zároveň nie je NaN
        stlpce_chyby[stlpec] = mimo    # Uložíme masku chýb pre tento konkrétny stĺpec (pre zvýraznenie bunky)
        riadky_maska = riadky_maska | mimo   # OR: riadok je chybný, ak má aspoň jednu hodnotu mimo medzi

    # Vyberieme len poškodené riadky a len stĺpce, ktoré chceme zobraziť v tabuľke
    dostupne = [s for s in ZOBRAZIT_STLPCE if s in df.columns]   # Stĺpce, ktoré reálne existujú v dátach (obrana pred chýbajúcimi stĺpcami)
    df_chyby = df[riadky_maska][dostupne].copy()   # Vyfiltrujeme poškodené riadky; .copy() zabraňuje SettingWithCopyWarning

    if df_chyby.empty:   # Ak DataFrame neobsahuje žiadne poškodené riadky
        return '<p style="color:#2e7d32; font-weight:600;">&#10003; Žiadne poškodené vzorky nenájdené.</p>'   # Zelená správa o úspechu

    # Zostavenie HTML tabuľky ručne (nie cez pandas .to_html()) – potrebujeme zvýrazniť konkrétne bunky
    hlavicka = ''.join(f'<th>{s}</th>' for s in dostupne)   # HTML hlavička: pre každý stĺpec jeden <th> element
    riadky_html = []   # Zoznam HTML reťazcov (jeden prvok = jeden <tr>...</tr> riadok tabuľky)

    for idx, row in df_chyby.iterrows():   # Iterujeme cez poškodené riadky: idx = index v pôvodnom DataFrame, row = riadok
        bunky = []   # Zoznam HTML buniek (<td>...</td>) pre aktuálny riadok
        for stlpec in dostupne:   # Pre každý zobrazovaný stĺpec
            hodnota = row[stlpec]   # Hodnota konkrétnej bunky (môže byť float, str, datetime)
            # Skontrolujeme, či je práve táto bunka (stĺpec × riadok) mimo fyzikálnych medzi
            je_chybna = (stlpec in stlpce_chyby) and stlpce_chyby[stlpec].get(idx, False)   # .get(idx, False) = bezpečné čítanie (ak idx chýba, vráti False)

            if je_chybna:   # Bunka obsahuje hodnotu mimo fyzikálneho rozsahu – zvýrazníme červenou
                mn, mx = FYZIKALNE_MEDZE[stlpec]   # Načítame povolené medze pre tento stĺpec (pre tooltip)
                bunky.append(
                    f'<td class="chybna-bunka" title="Povolený rozsah: [{mn}, {mx}]">'
                    f'{hodnota}</td>'   # Bunka: trieda CSS "chybna-bunka" = červené pozadie; title = tooltip pri hover
                )
            else:   # Bunka je v poriadku – normálne zobrazenie
                if isinstance(hodnota, float):   # Ak je hodnota desatinné číslo (float)
                    bunky.append(f'<td>{hodnota:.4f}</td>')   # Zaokrúhlíme na 4 desatinné miesta pre prehľadnosť
                else:
                    bunky.append(f'<td>{hodnota}</td>')   # Ostatné typy (str, datetime) zobrazíme bez formátovania

        riadky_html.append(f'<tr>{"".join(bunky)}</tr>')   # Zostavíme celý riadok tabuľky zo zoznamu buniek

    # Vrátime kompletný HTML blok s tabuľkou poškodených vzoriek
    return f"""
    <p style="margin-bottom:12px; color:#555;">
      Zobrazených <strong>{len(df_chyby)}</strong> poškodených vzoriek.
      <span style="color:#c62828;">&#9632;</span> Červená bunka = hodnota mimo fyzikálneho rozsahu
      (podržte myš pre zobrazenie povoleného rozsahu).
    </p>
    <div style="overflow-x:auto;">
    <table class="chyby-tabulka">
      <thead><tr>{hlavicka}</tr></thead>
      <tbody>{''.join(riadky_html)}</tbody>
    </table>
    </div>"""


def obrazok_na_base64(cesta):
    """
    Načíta PNG obrázok zo súborového systému a zakóduje ho do base64 reťazca.
    Výsledok sa vloží priamo do HTML ako: <img src="data:image/png;base64,...">
    Výhodou je self-contained HTML súbor – nevyžaduje externé obrázky.

    Parametre:
        cesta – reťazec, absolútna alebo relatívna cesta k PNG súboru

    Vracia: base64 reťazec (str) alebo None ak súbor neexistuje
    """
    if not os.path.exists(cesta):   # Skontrolujeme, či súbor skutočne existuje na disku
        return None                 # Ak nie, vrátime None (volajúci kód to ošetrí)
    with open(cesta, 'rb') as f:   # Otvoríme súbor v binárnom móde ('rb' = read binary) – PNG je binárny formát
        data = f.read()            # Načítame celý obsah súboru ako bytes objekt
    return base64.b64encode(data).decode('utf-8')   # b64encode() = zakódujeme bytes na base64 bytes; .decode() = konverzia na Python str


def text_na_html(text):
    """
    Konvertuje zachytený textový výstup (print výstupy analýz) na formátovaný HTML blok.
    Rôzne typy riadkov dostanú rôzne CSS triedy pre vizuálne odlíšenie.

    Parametre:
        text – reťazec (str) so zachyteným textovým výstupom funkcie

    Vracia: HTML reťazec (str) pripravený na vloženie do <div class="vystup-text">
    """
    riadky = []   # Zoznam HTML reťazcov – jeden prvok za každý vstupný riadok
    for riadok in text.strip().split('\n'):   # .strip() odstraňuje prázdne riadky na začiatku/konci; .split('\n') rozdelí na riadky
        if not riadok.strip():   # Ak je riadok prázdny (len biele znaky alebo prázdny reťazec)
            riadky.append('<br>')   # Vložíme HTML zalomenie riadku pre zachovanie vizuálnych medzier
            continue                # Preskočíme zvyšok cyklu, pokračujeme na ďalší riadok
        if riadok.strip().startswith('==='):   # Riadok začína '===' = ide o nadpis sekcie (napr. "=== ŠTATISTIKY ===")
            obsah = riadok.strip().strip('=').strip()   # Odoberieme znaky '=' z oboch strán a biele znaky
            riadky.append(f'<div class="sekcia-nadpis">{obsah}</div>')   # Zobrazíme ako tučný modrý nadpis
        elif set(riadok.strip()) <= set('=-'):   # Riadok pozostáva VÝHRADNE zo znakov '=' a '-' = čistý oddeľovač
            continue   # Preskočíme – vizuálne oddeľovače nepotrebujeme v HTML (máme CSS)
        elif ':' in riadok and any(c.isdigit() for c in riadok):   # Riadok obsahuje ':' (kľúč: hodnota) A aspoň jednu číslicu (štatistika)
            cast = riadok.split(':', 1)   # Rozdelíme len pri prvej dvojbodke (maxsplit=1), napr. ["  Priemer", " 345.6 W/m²"]
            riadky.append(
                f'<div class="hodnota-riadok">'
                f'<span class="kluc">{cast[0]}:</span>'   # Ľavá časť = parameter/kľúč (šedý)
                f'<span class="hodnota">{cast[1]}</span>' # Pravá časť = číslo/hodnota (modrá, tučná)
                f'</div>'
            )
        else:   # Obyčajný informatívny riadok (napr. "Graf uložený: ..." alebo "---")
            escaped = riadok.replace('<', '&lt;').replace('>', '&gt;')   # Nahradíme HTML znaky '<' a '>' entitami (bezpečnosť – XSS prevencia)
            riadky.append(f'<div class="info-riadok">{escaped}</div>')   # Zobrazíme ako bežný šedý riadok
    return '\n'.join(riadky)   # Spojíme všetky HTML riadky do jedného reťazca oddelenými novým riadkom


# ==============================================================
# HLAVNÁ FUNKCIA – zostavenie HTML reportu
# ==============================================================

def generuj_html_report(csv_subor, output_dir):
    """
    Spustí celú analýzu, zachytí textový výstup a vygeneruje self-contained HTML report.
    Obrázky sú vložené priamo do HTML ako base64 → výsledok je jeden prenosný .html súbor.

    Parametre:
        csv_subor  – reťazec, cesta k vstupnému CSV súboru s meteorologickými dátami
        output_dir – reťazec, cesta k adresáru kde sú uložené grafy a kde sa uloží report.html
    """
    print("\n=== GENEROVANIE HTML REPORTU ===")   # Nadpis sekcie

    # --- Načítanie dát ---
    print("  Načítavam dáta...")   # Informujeme používateľa o začatí načítavania
    df = nacitaj_data(csv_subor)   # Načítame CSV a vrátime pripravený DataFrame (s DateTime, Date, Hour, Month, DayOfYear)

    # ----------------------------------------------------------
    # Spustenie všetkých analýz so zachytením textového výstupu
    # Každá analýza sa spustí znova – výsledné texty idú do HTML.
    # Grafy sa neukladajú znova (súbory PNG už existujú z main.py).
    # ----------------------------------------------------------
    print("  Spúšťam analýzy...")   # Informujeme o začatí

    _, txt_statistiky   = zachyt_vystup(suhrne_statistiky,           df, output_dir)   # Súhrnné štatistiky: vracia (None, text)
    _, txt_dni          = zachyt_vystup(analyza_chybajucich_dni,      df, output_dir)   # Chýbajúce dni: vracia (zoznam_dni, text)
    _, txt_nan          = zachyt_vystup(analyza_chybajucich_vzoriek,  df)               # NaN analýza: vracia (None, text)
    (poskodene, pravd), txt_poskodene = zachyt_vystup(                                  # Poškodené vzorky: vracia ((maska, pravd.), text)
        detekcia_poskodennych_vzoriek, df                                               # Rozbalíme dvojicu (poskodene, pravd) z návratovej hodnoty
    )
    _, txt_dlzky        = zachyt_vystup(histogramy_dlzok_chyb,        df, poskodene, output_dir)   # Histogramy dĺžok skupín chýb
    _, txt_rocna        = zachyt_vystup(rocna_analyza_ziarenia,        df, output_dir)             # Ročná analýza žiarenia
    _, txt_mesiacna     = zachyt_vystup(denna_analyza_pre_mesiace,     df, output_dir)             # Mesačné denné profily žiarenia
    _, txt_korelacie    = zachyt_vystup(korelacie_parametrov,          df, output_dir)             # Korelačná matica a scatter ploty

    # ----------------------------------------------------------
    # Načítanie obrázkov ako base64 reťazcov
    # ----------------------------------------------------------
    print("  Načítavam obrázky...")   # Informujeme o začatí kódovania

    def img(nazov):
        """Pomocná funkcia: zostaví cestu k obrázku v output_dir a zakóduje ho do base64."""
        return obrazok_na_base64(os.path.join(output_dir, nazov))   # os.path.join bezpečne spája adresár a názov súboru

    obrazky = {   # Slovník: kľúč (str) → base64 reťazec alebo None (ak súbor neexistuje)
        'statistiky_csv':         os.path.join(output_dir, 'suhrne_statistiky.csv'),   # Cesta k CSV (nie base64 – načítame inak)
        'histogram_dni':          img('histogram_chybajucich_dni.png'),                # Histogram chýbajúcich dní podľa mesiaca
        'histogram_vzoriek':      img('histogram_vzoriek_za_den.png'),                 # Distribúcia počtu vzoriek za deň
        'vzorky_casovy':          img('vzorky_za_den_casovy_priebeh.png'),             # Časový priebeh počtu vzoriek
        'histogram_poskodene':    img('histogram_dlzok_poskodennych.png'),             # Histogram dĺžok skupín poškodených vzoriek
        'rocna_analyza':          img('rocna_analyza_ziarenia.png'),                   # Ročný priebeh žiarenia s pásom ±1σ
        'mesiacna_analyza':       img('denna_analyza_mesiacov.png'),                   # Denné profily žiarenia pre každý mesiac
        'heatmapa':               img('heatmapa_korelacie.png'),                       # Korelačná heatmapa (Pearson)
        'scatter':                img('scatter_korelacie.png'),                        # Scatter ploty vybraných párov
    }

    # Načítanie CSV štatistík a konverzia na HTML tabuľku
    stats_csv = os.path.join(output_dir, 'suhrne_statistiky.csv')   # Cesta k CSV so súhrnnými štatistikami
    if os.path.exists(stats_csv):   # Skontrolujeme, či CSV súbor existuje (mohol byť vygenerovaný skôr)
        df_stats = pd.read_csv(stats_csv, index_col=0)               # Načítame CSV; index_col=0 = prvý stĺpec je index (count, mean, std, ...)
        tabulka_html = df_stats.round(3).to_html(classes='stats-tabulka', border=0)   # round(3) = 3 desatinné miesta; classes = CSS trieda pre štýlovanie
    else:
        tabulka_html = '<p>Štatistiky nie sú dostupné.</p>'   # Náhradný text ak CSV chýba

    # ----------------------------------------------------------
    # Pomocná funkcia pre vkladanie obrázkov do HTML
    # ----------------------------------------------------------
    def img_tag(kluc, alt='', sirka='100%'):
        """
        Vráti HTML <img> tag s obrázkom zakódovaným v base64, alebo chybovú správu ak obrázok chýba.

        Parametre:
            kluc  – kľúč do slovníka obrazky (str)
            alt   – alternatívny text obrázku pre prístupnosť (str)
            sirka – CSS šírka obrázku (str, napr. '100%' alebo '50%')
        """
        if obrazky.get(kluc):   # Skontrolujeme, či pre daný kľúč existuje base64 reťazec (nie None)
            return (f'<img src="data:image/png;base64,{obrazky[kluc]}" '   # Vložíme base64 dáta priamo do atribútu src (Data URI)
                    f'alt="{alt}" style="width:{sirka}; max-width:100%;">')  # alt = popis pre screen readery; max-width = responsívnosť
        return f'<p class="chyba">Obrázok "{alt}" nie je dostupný.</p>'   # Fallback: zobraziť text ak obrázok chýba

    # ----------------------------------------------------------
    # Zostavenie HTML dokumentu pomocou f-string šablóny
    # ----------------------------------------------------------
    print("  Zostavujem HTML...")   # Informujeme o začatí zostavovania HTML

    html = f"""<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Štatistická analýza meteorologických dát</title>
  <style>
    /* ---- Základné štýly ---- */
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f0f2f5;
      color: #222;
      line-height: 1.6;
    }}

    /* ---- Hlavička ---- */
    header {{
      background: linear-gradient(135deg, #1a237e, #283593);
      color: white;
      padding: 40px 60px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 8px; }}
    header p  {{ font-size: 1rem; opacity: 0.85; }}

    /* ---- Navigácia ---- */
    nav {{
      background: #283593;
      padding: 12px 60px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }}
    nav a {{
      color: #c5cae9;
      text-decoration: none;
      font-size: 0.85rem;
      padding: 4px 10px;
      border-radius: 12px;
      transition: background 0.2s;
    }}
    nav a:hover {{ background: rgba(255,255,255,0.15); color: white; }}

    /* ---- Obsah ---- */
    main {{ max-width: 1300px; margin: 30px auto; padding: 0 30px; }}

    /* ---- Sekcie ---- */
    .sekcia {{
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
      margin-bottom: 30px;
      overflow: hidden;
    }}
    .sekcia-hlavicka {{
      background: #e8eaf6;
      padding: 18px 28px;
      border-left: 5px solid #3949ab;
    }}
    .sekcia-hlavicka h2 {{
      font-size: 1.2rem;
      color: #1a237e;
      font-weight: 600;
    }}
    .sekcia-telo {{ padding: 24px 28px; }}

    /* ---- Karta metriky ---- */
    .karty {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
    .karta {{
      background: #f5f5f5;
      border-radius: 10px;
      padding: 16px 22px;
      min-width: 160px;
      flex: 1;
      border-top: 4px solid #3949ab;
    }}
    .karta-nadpis {{ font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
    .karta-hodnota {{ font-size: 1.6rem; font-weight: 700; color: #1a237e; margin-top: 4px; }}
    .karta-jednotka {{ font-size: 0.8rem; color: #888; }}

    /* ---- Výstupný text ---- */
    .vystup-text {{
      background: #f8f9ff;
      border: 1px solid #e0e3ff;
      border-radius: 8px;
      padding: 16px 20px;
      font-family: 'Consolas', 'Courier New', monospace;
      font-size: 0.82rem;
      margin-bottom: 20px;
      overflow-x: auto;
    }}
    .sekcia-nadpis {{
      font-weight: 700;
      color: #283593;
      margin: 10px 0 6px;
      font-size: 0.9rem;
    }}
    .hodnota-riadok {{ display: flex; gap: 8px; padding: 2px 0; }}
    .kluc  {{ color: #555; min-width: 260px; }}
    .hodnota {{ color: #1a237e; font-weight: 600; }}
    .info-riadok {{ color: #444; padding: 1px 0; }}

    /* ---- Tabuľka štatistík ---- */
    .stats-tabulka {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      margin-bottom: 20px;
      overflow-x: auto;
      display: block;
    }}
    .stats-tabulka th {{
      background: #3949ab;
      color: white;
      padding: 10px 14px;
      text-align: right;
      white-space: nowrap;
    }}
    .stats-tabulka th:first-child {{ text-align: left; }}
    .stats-tabulka td {{
      padding: 8px 14px;
      text-align: right;
      border-bottom: 1px solid #eee;
    }}
    .stats-tabulka td:first-child {{ text-align: left; font-weight: 600; color: #3949ab; }}
    .stats-tabulka tr:hover {{ background: #f0f4ff; }}

    /* ---- Obrázky ---- */
    .obrazok-kontainer {{ margin: 16px 0; }}
    .obrazok-kontainer img {{
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      display: block;
    }}
    .obrazok-popis {{
      font-size: 0.8rem;
      color: #888;
      text-align: center;
      margin-top: 6px;
    }}
    .dvojica {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 800px) {{ .dvojica {{ grid-template-columns: 1fr; }} }}

    /* ---- Päta ---- */
    footer {{
      text-align: center;
      padding: 30px;
      color: #888;
      font-size: 0.8rem;
    }}

    .chyba {{ color: #aaa; font-style: italic; font-size: 0.85rem; }}

    /* ---- Tabuľka chybných vzoriek ---- */
    .chyby-tabulka {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
      font-family: 'Consolas', monospace;
    }}
    .chyby-tabulka th {{
      background: #b71c1c;
      color: white;
      padding: 9px 12px;
      text-align: left;
      white-space: nowrap;
      position: sticky;
      top: 0;
    }}
    .chyby-tabulka td {{
      padding: 7px 12px;
      border-bottom: 1px solid #eee;
      white-space: nowrap;
    }}
    .chyby-tabulka tr:hover td {{ background: #fff8f8; }}
    .chybna-bunka {{
      background: #ffebee !important;
      color: #c62828;
      font-weight: 700;
      border: 1px solid #ef9a9a !important;
      cursor: help;
    }}
  </style>
</head>
<body>

<!-- HLAVIČKA -->
<header>
  <h1>Štatistická analýza meteorologických dát</h1>
  <p>Vstupný súbor: <strong>{csv_subor}</strong> &nbsp;|&nbsp; Výstupný adresár: <strong>{output_dir}/</strong></p>
</header>

<!-- NAVIGÁCIA -->
<nav>
  <a href="#statistiky">Štatistiky</a>
  <a href="#chybajuce-dni">Chýbajúce dni</a>
  <a href="#chybajuce-vzorky">Chýbajúce vzorky</a>
  <a href="#poskodene">Poškodené vzorky</a>
  <a href="#rocna">Ročná analýza</a>
  <a href="#mesiacna">Mesačná analýza</a>
  <a href="#korelacie">Korelácie</a>
</nav>

<main>

<!-- ============================================================ -->
<!-- 1. SÚHRNNÉ ŠTATISTIKY -->
<!-- ============================================================ -->
<div class="sekcia" id="statistiky">
  <div class="sekcia-hlavicka"><h2>1. Súhrnné štatistiky parametrov</h2></div>
  <div class="sekcia-telo">
    <div class="karty">
      <div class="karta">
        <div class="karta-nadpis">Celkový počet vzoriek</div>
        <div class="karta-hodnota">{len(df):,}</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Počet dní s dátami</div>
        <div class="karta-hodnota">{df['Date'].nunique()}</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Priemerné žiarenie</div>
        <div class="karta-hodnota">{df['Irradiance'].mean():.1f}</div>
        <div class="karta-jednotka">W/m²</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Maximum žiarenia</div>
        <div class="karta-hodnota">{df['Irradiance'].max():.1f}</div>
        <div class="karta-jednotka">W/m²</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Priemerný tlak</div>
        <div class="karta-hodnota">{df['Pressure'].mean():.0f}</div>
        <div class="karta-jednotka">hPa</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Priemerná teplota</div>
        <div class="karta-hodnota">{df['BodyTemperature'].mean():.1f}</div>
        <div class="karta-jednotka">°C</div>
      </div>
    </div>
    {tabulka_html}
    <div class="vystup-text">{text_na_html(txt_statistiky)}</div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 2. CHÝBAJÚCE DNI -->
<!-- ============================================================ -->
<div class="sekcia" id="chybajuce-dni">
  <div class="sekcia-hlavicka"><h2>2. Analýza chýbajúcich dní a počtu vzoriek</h2></div>
  <div class="sekcia-telo">
    <div class="karty">
      <div class="karta">
        <div class="karta-nadpis">Chýbajúce dni</div>
        <div class="karta-hodnota" style="color:#c62828;">
          {len(set(pd.date_range(df['DateTime'].min().date(), df['DateTime'].max().date(), freq='D').date) - set(df['Date'].unique()))}
        </div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Priem. vzoriek / deň</div>
        <div class="karta-hodnota">{df.groupby('Date').size().mean():.1f}</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Min vzoriek / deň</div>
        <div class="karta-hodnota">{df.groupby('Date').size().min()}</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Max vzoriek / deň</div>
        <div class="karta-hodnota">{df.groupby('Date').size().max()}</div>
      </div>
    </div>
    <div class="vystup-text">{text_na_html(txt_dni)}</div>
    <div class="dvojica">
      <div class="obrazok-kontainer">
        {img_tag('histogram_dni', 'Histogram chýbajúcich dní')}
        <div class="obrazok-popis">Histogram chýbajúcich dní podľa mesiaca</div>
      </div>
      <div class="obrazok-kontainer">
        {img_tag('histogram_vzoriek', 'Histogram vzoriek za deň')}
        <div class="obrazok-popis">Distribúcia počtu vzoriek na deň</div>
      </div>
    </div>
    <div class="obrazok-kontainer" style="margin-top:20px;">
      {img_tag('vzorky_casovy', 'Vzorky v čase')}
      <div class="obrazok-popis">Počet vzoriek za každý deň v čase</div>
    </div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 3. CHÝBAJÚCE VZORKY (NaN) -->
<!-- ============================================================ -->
<div class="sekcia" id="chybajuce-vzorky">
  <div class="sekcia-hlavicka"><h2>3. Chýbajúce vzorky (NaN hodnoty)</h2></div>
  <div class="sekcia-telo">
    <div class="vystup-text">{text_na_html(txt_nan)}</div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 4. POŠKODENÉ VZORKY -->
<!-- ============================================================ -->
<div class="sekcia" id="poskodene">
  <div class="sekcia-hlavicka"><h2>4. Poškodené vzorky a dĺžky súvislých skupín chýb</h2></div>
  <div class="sekcia-telo">
    <div class="karty">
      <div class="karta" style="border-top-color:#c62828;">
        <div class="karta-nadpis">Pravdepodobnosť chybnej vzorky</div>
        <div class="karta-hodnota" style="color:#c62828;">{pravd*100:.4f} %</div>
      </div>
    </div>
    <div class="vystup-text">{text_na_html(txt_poskodene + txt_dlzky)}</div>
    <div class="obrazok-kontainer">
      {img_tag('histogram_poskodene', 'Histogram dĺžok poškodených vzoriek')}
      <div class="obrazok-popis">Histogram dĺžok za sebou idúcich poškodených vzoriek</div>
    </div>
    <h3 style="margin: 24px 0 12px; color:#b71c1c; font-size:1rem;">
      Zoznam poškodených vzoriek
    </h3>
    {tabulka_poskodennych(df)}
  </div>
</div>

<!-- ============================================================ -->
<!-- 5. ROČNÁ ANALÝZA ŽIARENIA -->
<!-- ============================================================ -->
<div class="sekcia" id="rocna">
  <div class="sekcia-hlavicka"><h2>5. Ročná analýza slnečného žiarenia</h2></div>
  <div class="sekcia-telo">
    <div class="vystup-text">{text_na_html(txt_rocna)}</div>
    <div class="obrazok-kontainer">
      {img_tag('rocna_analyza', 'Ročná analýza žiarenia')}
      <div class="obrazok-popis">Priemerné žiarenie počas roka s disperziou ±1σ</div>
    </div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 6. MESAČNÁ DENNÁ ANALÝZA -->
<!-- ============================================================ -->
<div class="sekcia" id="mesiacna">
  <div class="sekcia-hlavicka"><h2>6. Denné profily žiarenia pre každý mesiac</h2></div>
  <div class="sekcia-telo">
    <div class="vystup-text">{text_na_html(txt_mesiacna)}</div>
    <div class="obrazok-kontainer">
      {img_tag('mesiacna_analyza', 'Mesačná denná analýza')}
      <div class="obrazok-popis">Priemerné žiarenie počas dňa pre každý mesiac s disperziou ±1σ</div>
    </div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 7. KORELAČNÁ ANALÝZA -->
<!-- ============================================================ -->
<div class="sekcia" id="korelacie">
  <div class="sekcia-hlavicka"><h2>7. Korelačná analýza parametrov</h2></div>
  <div class="sekcia-telo">
    <div class="vystup-text">{text_na_html(txt_korelacie)}</div>
    <div class="obrazok-kontainer">
      {img_tag('heatmapa', 'Korelačná matica')}
      <div class="obrazok-popis">Pearsonova korelačná matica</div>
    </div>
    <div class="obrazok-kontainer" style="margin-top:20px;">
      {img_tag('scatter', 'Scatter plot korelácie')}
      <div class="obrazok-popis">Scatter ploty vybraných dvojíc parametrov</div>
    </div>
  </div>
</div>


</main>

<!-- PÄTA -->
<footer>
  Vygenerované automaticky &nbsp;|&nbsp; Vstupný súbor: {csv_subor}
</footer>

</body>
</html>"""

    # ----------------------------------------------------------
    # Uloženie hotového HTML súboru na disk
    # ----------------------------------------------------------
    cesta_html = os.path.join(output_dir, 'report.html')   # Zostavíme cestu: output_dir + 'report.html'
    with open(cesta_html, 'w', encoding='utf-8') as f:     # Otvoríme súbor pre zápis; UTF-8 je nutné pre slovenské znaky (á, é, č, ...)
        f.write(html)   # Zapíšeme celý HTML reťazec do súboru

    velkost_kb = os.path.getsize(cesta_html) / 1024   # os.path.getsize vráti veľkosť v bajtoch; delíme 1024 pre kB
    print(f"  HTML report uložený: {cesta_html}  ({velkost_kb:.0f} kB)")   # Informujeme o úspešnom uložení a veľkosti súboru
    return cesta_html   # Vrátime cestu k vygenerovanému HTML súboru (môže využiť volajúci kód)
