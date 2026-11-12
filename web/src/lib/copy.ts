// Spanish UI copy, ported from agent/ui_es.py (markdown emphasis removed) plus
// web-only strings. Templates use {name} placeholders.
export const COPY = {
  // Shell
  pageTitle: "Rooster · Valencia",
  wordmark: "Rooster",
  tagline: "Tu copiloto inmobiliario para Valencia",
  navChat: "Pregunta a Rooster",
  navIntel: "Inteligencia",
  close: "Cerrar",

  // Chat
  chatInputPlaceholder: "Pregunta por anuncios, rentabilidades, barrios…",
  statusThinking: "Pensando…",
  newConversation: "Nueva conversación",
  chatErrorSuggestion: "Vuelve a intentarlo en unos segundos; si persiste, recarga la página.",
  busy: "Rooster sigue respondiendo a tu mensaje anterior.",
  rateLimited: "Rooster está atendiendo muchas preguntas ahora mismo. Inténtalo de nuevo en un minuto.",
  chatRequestFailed: "No se pudo contactar con Rooster.",
  chatStreamInterrupted: "La respuesta se interrumpió antes de terminar.",
  chatSend: "Enviar",
  chatMessagesLabel: "Conversación",
  chatFollowUpsLabel: "Sugerencias:",
  chatExamples: [
    "¿Qué barrios de Valencia tienen la mejor rentabilidad bruta?",
    "Muéstrame en un mapa la puntuación de inversión por barrio",
    "¿Dónde está más barato el metro cuadrado en venta?",
    "¿Qué barrios tienen buen transporte y poca presión turística?",
  ],

  // Primitives
  emptyChart: "Sin datos para este gráfico.",
  noRowsTable: "No hay filas que mostrar.",
  mapLoadError: "No se pudo cargar el mapa.",
  popupIdealista: "Ver en Idealista →",
  codeMapTitle: "Mapa generado por Rooster",
  codeChartError: "No se pudo mostrar este gráfico.",
  popupVenta: "Venta",
  popupAlquiler: "Alquiler",
  tooltipNeighborhood: "Barrio",
  tooltipYield: "Rent. bruta %",

  // Intel page
  intelCaption:
    "De arriba abajo: contexto del mercado, mapa, ranking por barrio y el equilibrio " +
    "rentabilidad–liquidez. Objetivo: ¿dónde invertir en Valencia y por qué?",
  intelS1Title: "1 · El mercado en tres cifras",
  intelS2Title: "2 · Mapa de Valencia",
  intelS3Title: "3 · Ranking de oportunidades (decidir)",
  intelS4Title: "4 · Rentabilidad frente a liquidez (riesgo)",
  sectionMercado: "Mercado · Valencia",
  sectionMapa: "Mapa de Valencia",
  kpiLabelMedian: "Rentabilidad bruta mediana",
  kpiLabelSpread: "Dispersión de rentabilidad",
  kpiLabelStrong: "Barrios >6% rentabilidad",
  kpiSubYield: "ciudad completa",
  kpiSubDispersion: "entre barrios",
  kpiSubCount: "de {n} analizados",
  intelMapColorBy: "Colorear barrios por",
  toggleBarrios: "Barrios",
  toggleTransport: "Transporte",
  toggleVut: "VUT",
  toggleListings: "Anuncios",
  toggleParcels: "Parcelas",
  intelListingsOpLegend:
    "Anuncios: borde coral = venta · borde azul = alquiler. El relleno sigue el color por €/m².",
  mapLegendInvestment:
    "Más oscuro = mejor puntuación de inversión (rentabilidad + transporte + baja presión turística).",
  intelNoFraming: "No hay datos suficientes para enmarcar el mercado (analytics.neighborhood_metrics).",
  intelNoRanking: "Sin datos para el ranking (necesitas analytics.neighborhood_metrics o neighborhood_profile).",
  intelNoScatter: "Sin datos para el gráfico rentabilidad–liquidez.",
  intelLoading: "Cargando…",
  intelLoadError: "No se pudieron cargar estos datos.",
  intelScatterX: "Anuncios totales (liquidez →)",
  intelScatterY: "Rentabilidad bruta % (rentabilidad ↑)",
  intelRankCaption:
    "Ordenado por puntuación de inversión. Cada fila resume la rentabilidad, el turismo y la puntuación del barrio; " +
    "toca una cabecera para reordenar.",
  intelScatterCaption:
    "Tamaño del punto = mediana €/m² venta. Arriba-izquierda = alta rentabilidad, poca oferta. " +
    "Arriba-derecha = equilibrio raro: rentabilidad y liquidez.",
} as const;

/** Same order and values as UI.CHOROPLETH_METRICS; the first entry is the default. */
export const CHOROPLETH_METRICS: readonly { readonly label: string; readonly value: string }[] = [
  { label: "Puntuación inversión", value: "investment_score" },
  { label: "Mediana venta €/m²", value: "median_venta_eur_per_sqm" },
  { label: "Mediana alquiler €/m²", value: "median_alquiler_eur_per_sqm" },
  { label: "Rentabilidad bruta %", value: "gross_rental_yield_pct" },
  { label: "Anuncios totales", value: "total_count" },
  { label: "Paradas transporte (barrio)", value: "transit_stop_count" },
  { label: "Densidad viviendas turísticas %", value: "tourist_density_pct" },
];
