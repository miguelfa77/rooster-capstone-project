"""Cadenas de interfaz en español para Rooster (Streamlit)."""

# —— Página y pestañas ——
PAGE_TITLE = "Rooster – Inmobiliario Valencia"
PAGE_SUBTITLE_INTEL = "Inteligencia"
PAGE_SUBTITLE_CHAT = "Pregunta a Rooster"
TAB_INTEL = "Inteligencia"
TAB_CHAT = "Pregunta a Rooster"

# —— Sidebar ——
SETTINGS = "Ajustes"
LLM_CAPTION = "LLM: **OpenAI** (único proveedor)"

ERR_NO_SQL = "El modelo no devolvió SQL"

POPUP_COLOR = "Escala ({legend}): {v}"
POPUP_IDEALISTA = "Ver en Idealista →"
LEGEND_EUR_M2 = "€/m²"
LEGEND_PRICE = "Precio (€)"
LEGEND_YIELD = "Rentabilidad barrio %"

# —— Mapas (Inteligencia) ——
INTEL_CAPTION = (
    "De arriba abajo: contexto del mercado, mapa, ranking por barrio y el equilibrio "
    "rentabilidad–liquidez. Objetivo: **¿dónde invertir en Valencia y por qué?**"
)
INTEL_S1_TITLE = "### 1 · El mercado en tres cifras"
INTEL_S2_TITLE = "### 2 · Mapa de Valencia"
INTEL_MAP_COLOR_BY = "Colorear barrios por"
INTEL_TOGGLE_BARRIOS = "Barrios"
INTEL_TOGGLE_TRANSPORT = "Transporte"
INTEL_TOGGLE_VUT = "VUT"
INTEL_TOGGLE_LISTINGS = "Anuncios"
INTEL_TOGGLE_PARCELS = "Parcelas"
INTEL_S3_TITLE = "### 3 · Ranking de oportunidades (decidir)"
INTEL_S4_TITLE = "### 4 · Rentabilidad frente a liquidez (riesgo)"
INTEL_METRIC_MEDIAN_YIELD = "Mediana rentabilidad bruta ciudad"
INTEL_METRIC_MEDIAN_YIELD_HELP = (
    "Mediana de la rentabilidad bruta en barrios con datos de alquiler y venta."
)
INTEL_METRIC_SPREAD = "Dispersión de rentabilidad"
INTEL_METRIC_SPREAD_HELP = (
    "Diferencia en puntos porcentuales entre el barrio con mayor y menor rentabilidad bruta."
)
INTEL_METRIC_STRONG_YIELD = "Barrios con rentabilidad >6%"
INTEL_METRIC_STRONG_YIELD_HELP = (
    "Cuenta de barrios por encima del umbral habitual en España para rentabilidad bruta."
)
INTEL_NO_FRAMING = "No hay datos suficientes para enmarcar el mercado (analytics.neighborhood_metrics)."
INTEL_NO_RANKING = "Sin datos para el ranking (necesitas analytics.neighborhood_metrics o neighborhood_profile)."
INTEL_NO_SCATTER = "Sin datos para el gráfico rentabilidad–liquidez."
MAP_LEGEND_INVESTMENT = (
    "Más oscuro = **mejor puntuación de inversión** (rentabilidad + transporte + baja presión turística)."
)
INTEL_RANK_CAPTION = (
    "Ordenado por **puntuación de inversión**. Barras = lectura rápida de rentabilidad, turismo y puntuación."
)
INTEL_SCATTER_CAPTION = (
    "Tamaño del punto = mediana €/m² venta. **Arriba-izquierda** = alta rentabilidad, poca oferta. "
    "**Arriba-derecha** = equilibrio raro: rentabilidad y liquidez."
)
BRIEF_TOP_PICK = "Mejor opción hoy"
BRIEF_BEST_VALUE = "Mejor €/m² (entre barrios por encima de la mediana)"
BRIEF_MARKET_SIGNAL = "Señal de mercado"
BRIEF_NO_TOP_PICK = "Sin candidato claro (revisa datos de analytics / perfil de barrios)."
BRIEF_NO_VALUE = "Sin barrio que cumpla criterios de valor (mediana ciudad y listados)."
BRIEF_NO_SIGNAL = "Sin comparación de precios entre scrapes (hacen falta al menos dos fechas en listing_snapshots)."
BRIEF_TOP_PICK_BODY = (
    "**{name}** — puntuación **{score:.1f}** — rent. bruta **{yld:.1f}%**, "
    "transporte **{tr}**, presión turística **{tp}**, **{vc}** anuncios de venta."
)
BRIEF_VALUE_BODY = (
    "**{name}** — **€{m2:,}/m²** mediana venta — rent. bruta **{yld:.1f}%** (> mediana ciudad **{med:.1f}%**)."
)
BRIEF_SIGNAL_BODY = (
    "**{name}** — precio medio venta **{chg:+.1f}%** entre el último scrape y el anterior — "
    "**{n}** anuncios con precio en ambos periodos."
)
NEIGHBORHOOD_DIVE_TITLE = "Perfil del barrio"
NEIGHBORHOOD_DIVE_SUPPLY = "**Oferta:** {v} venta · {a} alquiler"
NEIGHBORHOOD_DIVE_PRICES = (
    "**Mediana alquiler:** €{rent:,}/mes · **Mediana venta:** €{sale:,}"
)
NEIGHBORHOOD_DIVE_TOURISM = (
    "**VUT:** {n} ({dens:.1f}% densidad) — presión turística **{tp}**"
)
NEIGHBORHOOD_DIVE_TRANSIT = "**Paradas de transporte en el barrio:** {n}"
MAP_VIEW_RADIO = "Vista"
MAP_VIEW_NEIGHBORHOOD = "Vista por barrios"
MAP_VIEW_LISTINGS = "Anuncios individuales"
COLOR_BY = "Colorear por"
SHOW_PARCELS = "Mostrar parcelas"
NO_PARCELS = "No hay parcelas en core.parcels."
PARCEL_LAYER_ERR = "Error en la capa de parcelas: {err}"
GROSS_YIELD_SECTION = "### Rentabilidad bruta por barrio"
GROSS_YIELD_CAPTION = (
    "Rentabilidad = (mediana alquiler mensual × 12) / mediana precio venta × 100. "
    "Requiere anuncios de alquiler y venta en el barrio."
)

# Orden: la primera opción es la predeterminada cuando existe en datos (puntuación de inversión).
CHOROPLETH_METRICS = {
    "Puntuación inversión": "investment_score",
    "Mediana venta €/m²": "median_venta_eur_per_sqm",
    "Mediana alquiler €/m²": "median_alquiler_eur_per_sqm",
    "Rentabilidad bruta %": "gross_rental_yield_pct",
    "Anuncios totales": "total_count",
    "Paradas transporte (barrio)": "transit_stop_count",
    "Densidad viviendas turísticas %": "tourist_density_pct",
}

# Only on analytics.neighborhood_profile (open-data). Not in neighborhood_metrics.
CHOROPLETH_METRICS_PROFILE_ONLY_COLUMNS = frozenset(
    {"transit_stop_count", "tourist_density_pct", "investment_score"}
)

# —— Mapa de puntos (Inteligencia) ——
INTEL_POINT_COLOR = {
    "€/m² (anuncio)": "eur_per_sqm",
    "Precio (€)": "price_int",
    "Rentabilidad barrio %": "yield_pct",
}

MAP_LOAD_ERR = "No se pudo cargar el mapa de barrios: {err}"
MAP_NO_POLYGONS = (
    "No hay polígonos de barrios. Carga **core.neighborhoods** con **geom** "
    "(límites de barrio). El mapa coroplético necesita al menos una geometría válida."
)
PARCEL_CAPTION = "{n:,} límites de parcela (simplificados, reproyectados a WGS84)."
LOADING_PARCELS = "Cargando parcelas catastrales…"
LOADING_LISTINGS = "Cargando anuncios…"

FOLIUM_NEIGHBORHOODS = "Barrios"
FOLIUM_LAYER_LISTINGS = "Anuncios (€/m²)"
FOLIUM_LAYER_TRANSIT = "Paradas de transporte"
FOLIUM_LAYER_TOURISM = "VUT (turístico)"
FOLIUM_PARCELS = "Parcelas catastrales"
TOOLTIP_NEIGHBORHOOD = "Barrio"
TOOLTIP_LISTINGS = "Anuncios"
TOOLTIP_YIELD = "Rent. bruta %"

# —— Tarjetas de rentabilidad ——
NO_YIELD_DATA = "Sin datos de rentabilidad: hacen falta alquiler y venta por barrio (analytics)."
TIER_STRONG = "Rentabilidad alta"
TIER_GOOD = "Buena rentabilidad"
TIER_MODERATE = "Moderada"
CARD_RENT = "Alquiler"
CARD_SALE = "Venta"

# —— Tarjetas de insights ——
INSIGHT_BEST_YIELD = "MEJOR RENTABILIDAD"
INSIGHT_BEST_VALUE = "MEJOR PRECIO/M²"
INSIGHT_MOST_ACTIVE = "MÁS ACTIVIDAD"
INSIGHT_BEST_YIELD_BODY = "{gy:.2f}% rentabilidad bruta — alquiler {ra} vs venta {sa}"
INSIGHT_BEST_VALUE_BODY = "{m2} mediana — {vc} anuncios de venta"
INSIGHT_SUPPLY_BODY = "{tc} anuncios en total — {ac} alquiler, {vc} venta"
NO_YIELD_INSIGHT = "Aún no hay datos de rentabilidad."
NO_AFFORD_INSIGHT = "Aún no hay datos de asequibilidad."
NO_ACTIVITY_INSIGHT = "Aún no hay datos de actividad."

INSIGHT_BEST_COMBINED = "MEJOR PUNTUACIÓN INVERSIÓN"
INSIGHT_COMBINED_BODY = (
    "Puntuación **{score:.2f}** — rent. bruta {yield_p:.2f}%, "
    "transporte **{tr}**, presión turística **{tur}**"
)
NO_COMBINED_INSIGHT = (
    "Sin puntuación combinada. Aplica `sql/open_data_*.sql` y carga transporte / viviendas turísticas."
)

# —— Mapa de anuncios (lista) ——
MAP_NO_LISTINGS = (
    "No hay anuncios en el mapa. Solo se incluyen filas con **neighborhood_id** "
    "(coincidencia con **core.neighborhoods**). Ejecuta geocodificación y "
    "`sql/match_listings_neighborhood_spatial.sql`, o comprueba que los anuncios tengan barrio."
)
MAP_NO_LATLNG = (
    "No hay filas con **lat/lng** — ejecuta la geocodificación (`sql/enrich_geocode.sql` / pipeline) "
    "para poder dibujar puntos."
)
MAP_BAD_COLUMN = "La columna `{col}` no está en los datos de anuncios."
MAP_NO_NUMERIC = "No hay valores numéricos para **{label}**."
MAP_CAPTION_GEO = (
    "Sólido = geocodificación a calle · hueco = centro del barrio. "
    "Escala: **{legend}** (percentiles 5–95)."
)

# —— Chat ——
CHAT_EMPTY = (
    "No hay anuncios que coincidan. Prueba ampliar la búsqueda: por ejemplo, "
    "subir el rango de precio o quitar el filtro de habitaciones."
)
CHAT_INPUT_PLACEHOLDER = "Pregunta por anuncios, rentabilidades, barrios…"
STATUS_THINKING = "Pensando…"
STATUS_QUERY = "Generando consulta…"
STATUS_RUNNING = "Ejecutando consulta…"
STATUS_SUMMARIZING = "Resumiendo…"
STATUS_MEMO = "Redactando memo de inversión…"
STATUS_DONE = "Hecho"
STATUS_ERROR = "Error"
RANKING_METRIC_DEFAULT = "Valor"
CHART_DEFAULT_SUMMARY = "Aquí tienes el gráfico con el último conjunto de anuncios."
MEMO_ERR = "No se pudo generar el memo: {err}"
CHAT_CONVERSATIONAL_TIMEOUT = (
    "No pude terminar la respuesta a tiempo. Inténtalo de nuevo."
)
CHAT_TRANSIT_EMPTY = "No hay paradas de transporte en esta selección."
CHAT_TRANSIT_STOP_DEFAULT = "Parada"
CHAT_TRANSIT_LEGEND_METRO = "🟣 Metro / tren (station, halt)"
CHAT_TRANSIT_LEGEND_BUS = "🟢 Bus (stop_position)"
CHAT_TRANSIT_COUNT = "**{n}** paradas en el área"
CHAT_TOURISM_EMPTY = "No hay viviendas turísticas licenciadas en esta selección."
CHAT_TOURISM_POPUP_TITLE = "VUT licenciada"
CHAT_TOURISM_COUNT = "🟡 **{n}** viviendas turísticas (VUT)"
CHAT_TOURISM_RISK = "Alta concentración = presión Airbnb + riesgo regulatorio"
CHAT_COMBINED_EMPTY = "No hay capas para mostrar en el mapa combinado."
CHAT_COMBINED_CAP_LISTINGS = "🔵 {n} anuncios (color = €/m²)"
CHAT_COMBINED_CAP_TRANSIT = "🟣 {n} paradas de transporte"
CHAT_COMBINED_CAP_TOURISM = "🟡 {n} VUT"
CHAT_NEIGHBORHOOD_HIGHLIGHT_CAPTION = (
    "{n} barrio(s) destacado(s). Haz clic en un barrio verde para ver los detalles."
)
CHAT_NO_COORDS_FALLBACK = (
    "No hay coordenadas en estos resultados; se muestra la tabla. "
    "Pide de nuevo un mapa si necesitas puntos en el plano."
)
CHAT_OUTPUT_CORRECTED = (
    "Consulta ajustada automáticamente para incluir todos los campos necesarios."
)

COMBINED_MAP_ALL_FAILED = "No se pudo cargar ninguna capa del mapa combinado."

# —— Fases timeout (si se usan) ——
TIMEOUT_PHASE_HINTS = {
    "intent_llm": "**Causa probable:** clasificación de intención — red o latencia del modelo.",
    "combined_llm": "**Causa probable:** llamada LLM intención+SQL — modelo lento, red o límites de uso.",
    "llm": "**Causa probable:** API de LLM (OpenAI/Gemini) — no es aún tu SQL.",
    "parse_sql": "**Causa probable:** raro; el LLM respondió pero el análisis tardó.",
    "database": "**Causa probable:** PostgreSQL — consulta pesada, bloqueo o base inaccesible.",
    "summarize_llm": "**Causa probable:** resumen tras el SQL — red o latencia del modelo.",
    "done": "Trabajo terminado en el servidor; sube el tiempo máximo si lo ves a menudo.",
    "unknown": "No se pudo identificar el paso en ejecución.",
}

# —— Analytics aviso ——
ANALYTICS_SETUP_MD = (
    "Faltan las vistas de analytics. Desde la raíz del repo (con `PG*` definido, p. ej. `pipeline/.env`):\n\n"
    "`psql -d rooster -f sql/analytics_views.sql`\n\n"
    "Si la vista falla por columnas antiguas en `core.listings` (p. ej. enriquecimiento o `first_seen_at`/`last_seen_at`), ejecuta antes:\n\n"
    "`psql -d rooster -f sql/migrate_listing_enrichment.sql`"
)

# —— Contexto asistente (LLM) ——
ASSISTANT_EMPTY_SEARCH = "Ninguna fila coincidió con esa búsqueda."
ASSISTANT_ERROR_PREFIX = "No se pudo completar: {err}"
ROWS_RETURNED = "({n} filas devueltas.)"

# —— Suelos (gráficos) ——
FLOOR_GROUND = "Baja"
FLOOR_PENTHOUSE = "Ático"
FLOOR_N = "Planta {n}"

# —— Renderers (chat) ——
GEO_FALLBACK_CAPTION = (
    "Contornos de barrio desde **core.neighborhoods** (instala las vistas **analytics** para métricas en el coroplético). "
    "Ejecuta: `psql -d rooster -f sql/analytics_views.sql`"
)
NO_ROWS_TABLE = "No hay filas que mostrar."
NO_ROWS_MAP = "No hay filas para el mapa."
NO_MAPPABLE_LATLNG = "No hay anuncios mapeables (hacen falta lat/lng)."
NO_MAPPABLE_FOUND = "No se encontraron anuncios mapeables."
NEED_EUR_M2 = "Hace falta **eur_per_sqm** (o precio + superficie) para colorear puntos."
NO_NUMERIC_MAP = "No hay valores numéricos para colorear el mapa."
LISTINGS_SHOWN = "{n} anuncios mostrados"
NO_GEO_LOADED = "No se cargó geometría de barrios (necesitas **core.neighborhoods** o vistas analytics)."
NO_GEO_LOADED_SHORT = "No se cargó geometría de barrios."
MATCH_ROWS_FAIL = (
    "No se pudieron cruzar las filas con los barrios del mapa — comprueba que incluyan **neighborhood_name**."
)
MATCH_GEOM_FAIL = "No se pudieron emparejar los barrios con la geometría."
NOTHING_TO_MAP = "Nada que mostrar en el mapa."
NEED_NEIGHBORHOOD_NAME = "Hace falta **neighborhood_name** para el coroplético."
NO_RANKING = "Sin datos de ranking."
NO_COMPARE = "Sin filas para comparar."
NO_OVERVIEW = "Sin datos de resumen."
NO_UNDERPRICED = "Sin datos de oportunidades."
UNDERPRICED_NEED_COLS = (
    "Hacen falta columnas como **underpriced_pct** (o **below_median_count** + **total_listings**) por barrio."
)
UNDERPRICED_CHORO_CAPTION = "Más verde = mayor parte de anuncios por debajo de la mediana de venta del barrio."
BELOW_MEDIAN_SHARE = "Parte bajo la mediana %"
NO_LISTINGS_DB = "No hay anuncios con precio y superficie en la base de datos."
AMENITY_MISSING = "Faltan columnas de equipamiento en los anuncios."
NO_FLOOR_DATA = "No hay datos de planta en este conjunto."
TREND_BUILDING = (
    "Los datos de tendencia aún se están generando — vuelve tras la próxima captura. "
    "Mientras tanto, la instantánea actual:"
)
AMENITY_PCT_LISTINGS = "% de anuncios"

# Tabla anuncios
COL_LINK = "Enlace"
COL_VIEW = "Ver →"
COL_PRICE = "Precio"
COL_AREA = "Superficie"
COL_NEIGHBORHOOD = "Barrio"
BARRIO_COL = "Barrio"
BELOW_MEDIAN_COL = "Bajo la mediana"
LISTINGS_COUNT = "{n} anuncio(s)"
LISTINGS_CAPTION_BELOW = " — **↓ Oferta** = por debajo de la mediana del barrio (si la consulta lo calcula)."
LISTINGS_CAPTION_PRICE = ""
SEARCH_TABLE_EUR_M2 = "€/m²"
SEARCH_TABLE_ROOMS = "Hab."
SEARCH_TABLE_FLOOR = "Planta"
SEARCH_TABLE_P = "P"
SEARCH_TABLE_T = "T"
SEARCH_TABLE_R = "R"
SEARCH_TABLE_DEAL = "↓ Oferta"
SEARCH_TABLE_P_HELP = "Parking"
SEARCH_TABLE_T_HELP = "Terraza"
SEARCH_TABLE_R_HELP = "Reformado"
SEARCH_TABLE_DEAL_HELP = "Por debajo de la mediana del barrio"

# Mapas folium popup (chat)
POPUP_ROOMS = "hab."
POPUP_VIEW = "Ver →"
EUR_M2_CAPTION = "€/m²"
UNDERPRICED_BELOW = "↓ {pct:.0f}% bajo la mediana"

# Gráfico scatter / chart
LABEL_AREA = "Superficie (m²)"
LABEL_PRICE = "Precio (€)"
AMENITY_LABELS = {
    "Parking": "Parking",
    "Terrace": "Terraza",
    "Elevator": "Ascensor",
    "Exterior": "Exterior",
    "Renovated": "Reformado",
    "A/C": "Aire acond.",
}
