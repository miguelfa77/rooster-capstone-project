// Parked for the Phase 2 Mapa page; not mounted anywhere.
"use client";

import { COPY } from "@/lib/copy";
import FramingCards from "./FramingCards";
import IntelMap from "./IntelMap";
import styles from "./Intel.module.css";
import RankingTable from "./RankingTable";
import YieldLiquidity from "./YieldLiquidity";

/** The Inteligencia page: four independent sections, each with its own loading/empty/error state. */
export default function IntelView() {
  return (
    <div className={styles.view}>
      <header>
        <h1 className={styles.title}>{COPY.navIntel}</h1>
        <p className={styles.caption}>{COPY.intelCaption}</p>
      </header>
      <FramingCards />
      <IntelMap />
      <RankingTable />
      <YieldLiquidity />
    </div>
  );
}
