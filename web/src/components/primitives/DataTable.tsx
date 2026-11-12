"use client";

import { useMemo, useState } from "react";
import { COPY } from "@/lib/copy";
import { formatCell, toNumber } from "@/lib/format";
import type { TableColumn, TablePayload } from "@/lib/types";
import EmptyState from "./EmptyState";
import styles from "./DataTable.module.css";

type Row = Record<string, unknown>;
type Sort = { field: string; dir: "asc" | "desc" } | null;

function parseRows(json: string): Row[] {
  try {
    const parsed: unknown = JSON.parse(json);
    return Array.isArray(parsed) ? parsed.filter((r): r is Row => typeof r === "object" && r !== null) : [];
  } catch {
    return [];
  }
}

function compare(a: unknown, b: unknown): number {
  const na = toNumber(a);
  const nb = toNumber(b);
  if (na !== null && nb !== null) return na - nb;
  return String(a).localeCompare(String(b), "es");
}

function sortRows(rows: Row[], sort: Sort): Row[] {
  if (!sort) return rows;
  const sign = sort.dir === "asc" ? 1 : -1;
  return [...rows].sort((ra, rb) => {
    const a = ra[sort.field];
    const b = rb[sort.field];
    const aNull = a === null || a === undefined;
    const bNull = b === null || b === undefined;
    if (aNull || bNull) return Number(aNull) - Number(bNull); // nulls last in both directions
    return sign * compare(a, b);
  });
}

export default function DataTable({ p }: { p: TablePayload }) {
  const rows = useMemo(() => parseRows(p.rows_json), [p.rows_json]);
  const columns: TableColumn[] = useMemo(
    () =>
      p.columns && p.columns.length > 0
        ? p.columns
        : Object.keys(rows[0] ?? {}).map((field) => ({
            field,
            header: field,
            formatter: "text",
            alignment: "left",
            width_hint: "medium",
          })),
    [p.columns, rows],
  );
  const [sort, setSort] = useState<Sort>(null);
  const sorted = useMemo(() => sortRows(rows, sort), [rows, sort]);

  if (rows.length === 0) return <EmptyState message={COPY.noRowsTable} />;

  const toggle = (field: string) =>
    setSort((s) => (s?.field === field && s.dir === "asc" ? { field, dir: "desc" } : { field, dir: "asc" }));

  return (
    <div className={styles.scroll}>
      <table className={styles.table}>
        <thead>
          <tr>
            {columns.map((c) => {
              const dir = sort?.field === c.field ? sort.dir : null;
              return (
                <th
                  key={c.field}
                  className={`${styles[c.alignment]} ${styles[c.width_hint]}`}
                  aria-sort={dir === "asc" ? "ascending" : dir === "desc" ? "descending" : "none"}
                >
                  <button type="button" className={styles.sortButton} onClick={() => toggle(c.field)}>
                    {c.header}
                    <span className={styles.arrow} aria-hidden="true">
                      {dir === "asc" ? "▲" : dir === "desc" ? "▼" : ""}
                    </span>
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => {
                const value = row[c.field];
                const text = formatCell(value, c.formatter);
                const isLink = c.formatter === "link" && value !== null && value !== undefined;
                return (
                  <td key={c.field} className={styles[c.alignment]}>
                    {isLink ? (
                      <a href={text} target="_blank" rel="noopener noreferrer">
                        {COPY.popupIdealista}
                      </a>
                    ) : (
                      text
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
