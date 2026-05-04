# PUT_SKEW_NIVEL_BATMAN_MT Dashboard

Pagina web con grafico interactivo + evidencia estadistica del **PUT SKEW NIVEL**
(percentil expanding del spread IV puts 25-delta vs ATM, DTE 60, snapshot 10:30 ET)
**validado exclusivamente contra Batman MT** (DTE 40-200).

URL publica: https://manumartinb.github.io/PUT_SKEW_NIVEL_BATMAN_MT/

## Que muestra

- Linea principal: **skew_25d_vs50_pct_expanding** (percentil 0-100)
- Bandas coloreadas: FAVORABLE (>=80, Batman MT convention), NEUTRAL (20-80), ADVERSO (<=20)
  - **OJO BWB**: convencion invertida. BWB FAV = pct <=20.
- Selector de rango: 30D / 90D / 1A / 3A / All
- Bloque de reglas operativas en el top (Batman MT entry/durante el trade)
- Seccion de evidencia (7 cards) bajo el grafico, **toda Batman MT exclusivo**:
  1. Concepto + bandas + inversion BWB
  2. Metodologia (Batman MT, sin filtro)
  3. Spearman r vs PnL Batman MT por horizonte d001-d049
  4. Deciles D1-D10 (PnL d020) + spread D10-D1 por horizonte
  5. Year stability 2019-2025
  6. Regime split (FAV/NEU/ADV) en d020 y d050
  7. Window-forward conditioning (HIGH/LOW PUT SKEW durante el trade Batman MT)

## Pipeline

Actualizacion automatica diaria via `V0.[PERMA] MASTER_DAILY_PIPELINE.py` (paralelo a los otros 5 dashboards):

```
V18 -> V8.0 (genera SKEW_PUT_ENRICHED.csv) -> Steps de dashboards en serie
```

`update_dashboard.py` lee SKEW_PUT_ENRICHED.csv filtrado a DTE=60/snapshot=10:30/side=PUT,
regenera `data.json` y hace push a este repo. GitHub Pages sirve el HTML estatico.

## Fuente de datos

- Daily series del grafico: `Skew/SKEW_PUT_ENRICHED.csv`, columna `skew_25d_vs50_pct_expanding`
- Validacion estadistica: `Batman/SPX/LIVE/[MAIN RANKEO MT]_combined_BATMAN_..._OWN_ALLDAYS.csv`
  (49,934 trades 2019-2025, DTE 40-200) joineado con PUT_SKEW por `trade_date`. Sin filtro SPX.

## Seccion de evidencia estadistica

La evidencia es **estatica** (no se regenera con V0 diario). Para regen manual:

```
python "C:\Users\Administrator\Desktop\BULK OPTIONSTRAT\ESTRATEGIAS\Skew\dashboards\PUT_SKEW_NIVEL_BATMAN_MT_DASHBOARD\generate_evidence.py" --push
```

Lo que hace `generate_evidence.py` (Batman MT exclusive):

1. Lee Batman MT dataset (`[MAIN RANKEO MT]_combined_BATMAN_..._OWN_ALLDAYS.csv`).
2. Joinea PUT SKEW NIVEL desde `SKEW_PUT_ENRICHED.csv` (DTE=60) por `trade_date`.
3. **Sin filtro SPX** (consistente con rules block del top y con dashboards hermanos LT/Allantis).
4. Calcula Spearman + bootstrap CI95, deciles a d020, year stability, regime split d020+d050.
5. Genera 5 PNGs propios (matplotlib dark theme matching dashboard).
6. Computa **window-forward in-script** sobre Batman MT (no usa CSV externo cross-strategy).
7. Volca `evidence/evidence.json` con metricas + tablas HTML inline.
8. Si `--push`: hace `git pull --rebase`, commit y push usando `GH_PUT_SKEW_TOKEN`.

**No correr entre 12:55 y 13:10 Madrid** &mdash; coincide con la ventana del push diario de V0
(todos los dashboards en serie) y podria provocar conflictos de rebase.

Sin `--push`: solo genera locales (util para iterar diseno antes de publicar).
