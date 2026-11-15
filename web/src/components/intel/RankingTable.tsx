// Parked for the Phase 2 Mapa page; not mounted anywhere.
"use client";

import DataTable from "@/components/primitives/DataTable";
import EmptyState from "@/components/primitives/EmptyState";
import { COPY } from "@/lib/copy";
import type { RankingRow, TableColumn } from "@/lib/types";
import styles from "./Intel.module.css";
import Section from "./Section";
import { useIntel } from "./useIntel";

function col(field: keyof RankingRow, header: string, formatter: TableColumn["formatter"]): TableColumn {
  const text = formatter === "text";
  return { field, header, formatter, alignment: text ? "left" : "right", width_hint: text ? "medium" : "narrow" };
}

// Headers as in the Streamlit ranking table.
const COLUMNS: TableColumn[] = [
  col("neighborhood_name", "Barrio", "text"),
  col("yield_pct", "Rent. %", "percentage"),
  col("eur_per_sqm", "€/m² (venta)", "currency"),
  col("transit_stop_count", "Paradas transporte", "integer"),
  col("transport_rating", "Transporte", "text"),
  col("tourist_pct", "Densidad turística %", "percentage"),
  col("tourism_pressure", "Presión turística", "text"),
  col("score", "Puntuación inversión", "number"),
  col("venta_count", "Anuncios venta", "integer"),
  col("alquiler_count", "Anuncios alquiler", "integer"),
];

/** Section 3: barrios in API order (by investment score), sortable by any column. */
export default function RankingTable() {
  const s = useIntel<RankingRow[]>("ranking");
  let body;
  if (s.status === "loading") body = <p className={styles.status}>{COPY.intelLoading}</p>;
  else if (s.status === "error") body = <EmptyState message={COPY.intelLoadError} />;
  else if (s.data.length === 0) body = <EmptyState message={COPY.intelNoRanking} />;
  else {
    const rows = s.data;
    const columns = COLUMNS.filter((c) => rows.some((r) => c.field in r));
    body = (
      <>
        <DataTable p={{ kind: "table", rows_json: JSON.stringify(rows), columns }} />
        <p className={styles.caption}>{COPY.intelRankCaption}</p>
      </>
    );
  }
  return <Section title={COPY.intelS3Title}>{body}</Section>;
}
