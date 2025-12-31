# -*- coding: utf-8 -*-
"""
Batterikalkylator för hushåll med solceller (timbaserat spotpris).

Antaganden (kan ändras i gränssnittet):
- Timupplöst modell för 1 år (8760 h).
- Förbrukning över dygnet följer spotprisets dygnskurva (normaliserad).
- Solproduktion har en enkel säsongs- + dygnsprofil (0 nattetid, max mitt på dagen, mer på sommaren).
- Batteriet laddas i första hand av solöverskott, och urladdas för att täcka last.
- Grid-arbitrage (ladda från nätet när priset är lågt) är avstängt som standard (kan aktiveras).
- Exportintäkt antas vara spotpris * exportfaktor (t.ex. 100% av spot). Inga nätavgifter/energiskatt på export.
- Importkostnad = spot + påslag + elöverföring + energiskatt (inkl moms), enligt dina indata.

Kör:
    streamlit run batteri_kalkylator.py
"""
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Batterikalkylator", layout="wide")

# ----------------------------
# Profiler
# ----------------------------
def daily_spot_shape(hours=np.arange(24), peak_amp=1.0, peak_sigma=2.5, dip_amp=-0.5, dip_sigma=3.5, sharpness=1.0):
    """
    Bygger en normaliserad dygnskurva (medel=1) med två toppar (morgon/kväll) och en svacka mitt på dagen.
    'sharpness' > 1 ger större dygnsspridning, <1 ger flatare kurva.
    """
    h = hours
    shape = (
        1
        + peak_amp * np.exp(-(h - 8) ** 2 / (2 * peak_sigma ** 2))
        + peak_amp * np.exp(-(h - 18) ** 2 / (2 * peak_sigma ** 2))
        + dip_amp  * np.exp(-(h - 13) ** 2 / (2 * dip_sigma ** 2))
    )
    shape = np.clip(shape, 0.2, None)

    # Justera spridning utan att ändra medelvärde
    if sharpness != 1.0:
        shape = shape ** sharpness

    shape = shape / shape.mean()
    return shape

def build_year_profiles(load_kwh_yr, pv_kwh_yr, spot_mean_kr, sharpness=1.0, seed=0):
    """
    Returnerar spot (kr/kWh), load (kWh/h), pv (kWh/h) för ett "typår".
    """
    hours = pd.date_range("2025-01-01", "2026-01-01", freq="h", inclusive="left")
    hod = hours.hour.to_numpy()

    shape = daily_spot_shape(sharpness=sharpness)
    spot = spot_mean_kr * np.take(shape, hod)

    # Last: proportionell mot spot-dygnsprofil (per antagandet)
    load_w = np.take(shape, hod)
    load = load_kwh_yr * load_w / load_w.sum()

    # Sol: enkel säsong + dygn (ingen molnrandomness)
    doy = hours.dayofyear.to_numpy()
    season = np.sin(np.pi * (doy - 80) / 365.0)
    season = np.clip(season, 0, None)  # 0..1

    # daglängd proxy: 8..16 h
    sun_hours = 8 + 8 * season
    sunrise = 12 - sun_hours / 2
    t = hod + 0.5
    pv_diurnal = np.sin(np.pi * (t - sunrise) / sun_hours)
    pv_diurnal = np.clip(pv_diurnal, 0, None)

    pv_raw = (season ** 1.3) * pv_diurnal
    pv = pv_raw * (pv_kwh_yr / pv_raw.sum()) if pv_raw.sum() > 0 else np.zeros_like(pv_raw)

    return hours, spot, load, pv

# ----------------------------
# Simulering
# ----------------------------
def simulate_battery(
    cap_kwh,
    load_kwh_yr,
    pv_kwh_yr,
    spot_mean_kr,
    import_add_kr,
    export_factor=1.0,
    eta_rt=0.90,
    c_rate=1.0,
    allow_grid_arbitrage=False,
    arbitrage_threshold_kr=0.10,
    sharpness=1.0
):
    hours, spot, load, pv = build_year_profiles(load_kwh_yr, pv_kwh_yr, spot_mean_kr, sharpness=sharpness)

    eta_ch = math.sqrt(eta_rt)
    eta_dis = math.sqrt(eta_rt)
    max_power = cap_kwh * c_rate  # kWh per timme

    soc = 0.0
    imports = np.zeros_like(load)
    exports = np.zeros_like(load)
    soc_series = np.zeros_like(load)

    for i in range(len(load)):
        g = pv[i]
        l = load[i]

        # 1) PV direkt till last
        use_pv = min(g, l)
        l_rem = l - use_pv
        surplus = g - use_pv

        # 2) Ladda av PV-överskott
        if cap_kwh > 0:
            charge_in = min(surplus, max_power, (cap_kwh - soc) / eta_ch)
            soc += charge_in * eta_ch
            surplus -= charge_in

        # 3) Urladda till last
        if cap_kwh > 0:
            deliverable = min(l_rem, max_power, soc * eta_dis)
            soc -= deliverable / eta_dis
            l_rem -= deliverable

        # 4) Enkel grid-arbitrage (valfritt): om priset "lågt" i denna timme jämfört med ett rullande framtidsmedel
        #    (Mycket förenklat – syftet är att ge en "knapp" för att testa effekt, inte exakt driftoptimering.)
        if allow_grid_arbitrage and cap_kwh > 0:
            # jämför mot medel av kommande 12 timmar
            j = min(i + 12, len(spot))
            future_avg = float(np.mean(spot[i:j])) if j > i else float(spot[i])
            if future_avg - spot[i] >= arbitrage_threshold_kr:
                # ladda från nätet upp till kapacitet
                grid_charge = min(max_power, (cap_kwh - soc) / eta_ch)
                soc += grid_charge * eta_ch
                imports[i] += grid_charge  # detta är extra import för laddning

        # 5) Export/import efter batteri
        exports[i] += surplus
        imports[i] += l_rem

        soc_series[i] = soc

    # Basfall utan batteri
    net = load - pv
    imp0 = np.clip(net, 0, None)
    exp0 = np.clip(-net, 0, None)

    # Kostnader: import betalar spot + fasta påslag; export ger spot*export_factor
    export_price = spot * export_factor
    cost0 = float(np.sum(imp0 * (spot + import_add_kr) - exp0 * export_price))
    cost1 = float(np.sum(imports * (spot + import_add_kr) - exports * export_price))
    savings = cost0 - cost1

    out = {
        "hours": hours,
        "spot": spot,
        "load": load,
        "pv": pv,
        "imports0": imp0, "exports0": exp0,
        "imports": imports, "exports": exports,
        "soc": soc_series,
        "cost0": cost0, "cost1": cost1, "savings": savings
    }
    return out

def npv(annual_savings, capex, r, years):
    pv_factor = (1 - (1 + r) ** (-years)) / r if r > 0 else years
    return annual_savings * pv_factor - capex

# ----------------------------
# UI
# ----------------------------
st.title("Batterikalkylator (spotpris + sol + batteri)")

with st.sidebar:
    st.header("Indata")
    load_mwh = st.slider("Årlig elanvändning (MWh/år)", 0.0, 30.0, 10.0, 0.5)
    pv_mwh = st.slider("Årlig solproduktion (MWh/år)", 0.0, 30.0, 11.0, 0.5)
    cap_kwh = st.slider("Batteristorlek (kWh)", 0.0, 150.0, 25.0, 1.0)

    st.subheader("Priser och avgifter")
    spot_ore = st.slider("Årligt snitt spotpris (öre/kWh)", 0.0, 300.0, 65.0, 1.0)
    markup_ore = st.slider("Påslag (öre/kWh)", 0.0, 50.0, 7.0, 0.5)
    transfer_ore = st.slider("Elöverföring (öre/kWh)", 0.0, 200.0, 87.15, 0.05)
    tax_ore = st.slider("Energiskatt inkl. moms (öre/kWh)", 0.0, 200.0, 54.88, 0.05)
    export_factor = st.slider("Exportintäkt som andel av spot (%)", 0, 120, 100, 5) / 100.0

    st.subheader("Batteri & ekonomi")
    batt_price = st.slider("Batteripris (kr/kWh)", 0, 6000, 2000, 50)
    eta_rt = st.slider("Rundverkansgrad (%)", 50, 100, 90, 1) / 100.0
    c_rate = st.slider("Max ladd/urladdning per timme (C-rate)", 0.1, 2.0, 1.0, 0.1)
    years = st.slider("Livslängd (år)", 1, 30, 20, 1)
    r_pct = st.slider("Investeringsränta (%)", 0.0, 15.0, 3.5, 0.1)
    sharpness = st.slider("Dygnsspridning i spot/förbrukning (1=default)", 0.5, 2.0, 1.0, 0.05)

    st.subheader("Valfritt: grid-arbitrage")
    allow_arb = st.checkbox("Tillåt enkel grid-arbitrage", value=False)
    arb_thr_ore = st.slider("Arbitrage-tröskel (öre/kWh)", 0.0, 200.0, 10.0, 1.0)

# Beräkning
spot_mean_kr = spot_ore / 100.0
import_add_kr = (markup_ore + transfer_ore + tax_ore) / 100.0

sim = simulate_battery(
    cap_kwh=cap_kwh,
    load_kwh_yr=load_mwh * 1000,
    pv_kwh_yr=pv_mwh * 1000,
    spot_mean_kr=spot_mean_kr,
    import_add_kr=import_add_kr,
    export_factor=export_factor,
    eta_rt=eta_rt,
    c_rate=c_rate,
    allow_grid_arbitrage=allow_arb,
    arbitrage_threshold_kr=arb_thr_ore / 100.0,
    sharpness=sharpness
)

capex = cap_kwh * batt_price
annual_savings = sim["savings"]
npv_val = npv(annual_savings, capex, r_pct/100.0, years)
payback = (capex / annual_savings) if annual_savings > 0 else float("inf")

# Sammanfattning
col1, col2, col3, col4 = st.columns(4)
col1.metric("Investering (kr)", f"{capex:,.0f}".replace(",", " "))
col2.metric("Årlig besparing (kr/år)", f"{annual_savings:,.0f}".replace(",", " "))
col3.metric("NPV (kr)", f"{npv_val:,.0f}".replace(",", " "))
col4.metric("Enkel återbetalning (år)", f"{payback:.1f}" if math.isfinite(payback) else "—")

# Energibalans
imp0 = float(np.sum(sim["imports0"]))
exp0 = float(np.sum(sim["exports0"]))
imp1 = float(np.sum(sim["imports"]))
exp1 = float(np.sum(sim["exports"]))

st.write("### Energibalans (kWh/år)")
tbl = pd.DataFrame([{
    "Import utan batteri": imp0,
    "Import med batteri": imp1,
    "Import minskar": imp0-imp1,
    "Export utan batteri": exp0,
    "Export med batteri": exp1,
    "Export minskar": exp0-exp1,
}], index=["kWh/år"]).T
st.dataframe(tbl.style.format("{:,.0f}"))

# Dagexempel (sommar + vinter)
def pick_day(hours_index, month):
    dt0 = pd.Timestamp(f"2025-{month:02d}-15")
    mask = (hours_index >= dt0) & (hours_index < dt0 + pd.Timedelta(days=1))
    return mask

winter_mask = pick_day(sim["hours"], 1)
summer_mask = pick_day(sim["hours"], 6)

def plot_day(mask, title):
    h = sim["hours"][mask]
    spot = sim["spot"][mask]
    load = sim["load"][mask]
    pv = sim["pv"][mask]
    soc = sim["soc"][mask]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(h.hour, spot, label="Spot (kr/kWh)")
    ax1.set_xlabel("Timme")
    ax1.set_ylabel("Spot (kr/kWh)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(h.hour, load, label="Last (kWh)", linestyle="--")
    ax2.plot(h.hour, pv, label="Sol (kWh)", linestyle="-.")
    ax2.plot(h.hour, soc, label="SOC (kWh)", linestyle=":")
    ax2.set_ylabel("Energi (kWh)")

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    st.pyplot(fig, clear_figure=True)

st.write("### Exempel på dygn")
c1, c2 = st.columns(2)
with c1:
    plot_day(winter_mask, "Vinterdag (15 jan)")
with c2:
    plot_day(summer_mask, "Sommardag (15 jun)")

st.caption("Obs: Profilerna är syntetiska/analytiska. För bästa precision – byt gärna till verkliga timserier för spotpris och solproduktion.")
