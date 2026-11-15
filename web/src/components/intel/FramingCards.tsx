// Parked for the Phase 2 Mapa page; not mounted anywhere.
"use client";

import EmptyState from "@/components/primitives/EmptyState";
import KpiCard from "@/components/primitives/KpiCard";
import { COPY } from "@/lib/copy";
import { formatCell, toNumber } from "@/lib/format";
import type { KpiPayload, MarketFraming } from "@/lib/types";
import styles from "./Intel.module.css";
import Section from "./Section";
import { useIntel } from "./useIntel";

const DASH = "—";
const ONE_DECIMAL = new Intl.NumberFormat("es-ES", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

function kpi(label: string, value: string): KpiPayload {
  return { kind: "kpi", label, value, unit: null, delta: null };
}

function Cards({ m }: { m: MarketFraming }) {
  const median = toNumber(m.city_median_yield);
  const spread = toNumber(m.spread_pp);
  const strong = toNumber(m.n_strong);
  const analysed = toNumber(m.n_with_yield);
  return (
    <div className={styles.kpis}>
      <KpiCard
        p={kpi(COPY.kpiLabelMedian, median === null ? DASH : formatCell(median, "percentage"))}
        sub={COPY.kpiSubYield}
      />
      <KpiCard
        p={kpi(COPY.kpiLabelSpread, spread === null ? DASH : `${ONE_DECIMAL.format(spread)} pp`)}
        sub={COPY.kpiSubDispersion}
      />
      <KpiCard
        p={kpi(COPY.kpiLabelStrong, strong === null ? DASH : formatCell(strong, "integer"))}
        sub={analysed === null ? DASH : COPY.kpiSubCount.replace("{n}", formatCell(analysed, "integer"))}
      />
    </div>
  );
}

/** Section 1: city median yield, yield spread and barrios above 6 %. */
export default function FramingCards() {
  const s = useIntel<MarketFraming | null>("framing");
  return (
    <Section title={COPY.sectionMercado}>
      {s.status === "loading" ? (
        <p className={styles.status}>{COPY.intelLoading}</p>
      ) : s.status === "error" ? (
        <EmptyState message={COPY.intelLoadError} />
      ) : s.data ? (
        <Cards m={s.data} />
      ) : (
        <EmptyState message={COPY.intelNoFraming} />
      )}
    </Section>
  );
}
