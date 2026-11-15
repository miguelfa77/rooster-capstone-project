"use client";

import {
  ChartNoAxesColumnIcon,
  LayoutListIcon,
  MapIcon,
  Maximize2Icon,
  TableIcon,
  type LucideIcon,
} from "lucide-react";
import { forwardRef, type ComponentProps } from "react";
import { groupBy, seriesColor } from "@/components/primitives/chartTheme";
import { COPY } from "@/lib/copy";
import { toNumber } from "@/lib/format";
import type { PrimitivePayload } from "@/lib/types";
import { BlockDialog } from "./BlockDialog";
import { blockTitle, kindLabel, plotlySketch, type Sketch } from "./compact";

const W = 100;
const H = 40;

/** Maps values to 0..1; a flat series sits in the middle. */
function scale(values: number[]): (v: number) => number {
  const min = Math.min(...values);
  const max = Math.max(...values);
  return (v) => (max === min ? 0.5 : (v - min) / (max - min));
}

function BarSketch({ values: all }: { values: number[] }) {
  const values = [...all].sort((a, b) => b - a).slice(0, 5);
  const max = Math.max(...values.map(Math.abs)) || 1;
  const step = H / values.length;
  return (
    <>
      {values.map((v, i) => (
        <rect
          key={i}
          x={0}
          y={i * step + 1}
          width={(Math.abs(v) / max) * W}
          height={step - 2}
          rx={1}
          fill="var(--primary)"
        />
      ))}
    </>
  );
}

function LineSketch({ series }: { series: number[][] }) {
  const y = scale(series.flat());
  return (
    <>
      {series.map((vs, i) => (
        <polyline
          key={i}
          points={vs.map((v, j) => `${(j / Math.max(1, vs.length - 1)) * W},${H - 2 - y(v) * (H - 4)}`).join(" ")}
          fill="none"
          stroke={seriesColor(i)}
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
          strokeLinejoin="round"
        />
      ))}
    </>
  );
}

function ScatterSketch({ points }: { points: readonly (readonly [number, number])[] }) {
  const x = scale(points.map(([v]) => v));
  const y = scale(points.map(([, v]) => v));
  // Zero-length round-capped lines stay round under the stretched viewBox.
  return (
    <>
      {points.map(([a, b], i) => {
        const cx = 3 + x(a) * (W - 6);
        const cy = H - 3 - y(b) * (H - 6);
        return (
          <line
            key={i}
            x1={cx}
            y1={cy}
            x2={cx}
            y2={cy}
            stroke="var(--primary)"
            strokeOpacity={0.8}
            strokeWidth={7}
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        );
      })}
    </>
  );
}

const nums = (rows: Record<string, unknown>[], field: string) =>
  rows.map((r) => toNumber(r[field])).filter((v): v is number => v !== null);

/** The sketch of a block's data (a Plotly figure's first trace for `code`), or null for an icon. */
function sketchOf(p: PrimitivePayload): Sketch | null {
  switch (p.kind) {
    case "bar": {
      const values = nums(p.data, p.value_field);
      return values.length > 0 ? { kind: "bar", values } : null;
    }
    case "line": {
      if (nums(p.data, p.y_field).length < 2) return null;
      const groups = p.color_field ? groupBy(p.data, (r) => r[p.color_field!]) : [["", p.data] as const];
      return {
        kind: "line",
        series: groups.map(([, rows]) => nums(rows, p.y_field)),
      };
    }
    case "scatter": {
      const points = p.data
        .map((r) => [toNumber(r[p.x_field]), toNumber(r[p.y_field])] as const)
        .filter((xy): xy is readonly [number, number] => xy[0] !== null && xy[1] !== null)
        .map(([a, b]): [number, number] => [a, b]);
      return points.length > 0 ? { kind: "scatter", points } : null;
    }
    case "code":
      return p.output_type === "plotly_json" ? plotlySketch(p.output) : null;
    default:
      return null;
  }
}

const ICONS: Partial<Record<PrimitivePayload["kind"], LucideIcon>> = {
  choropleth: MapIcon,
  point_map: MapIcon,
  table: TableIcon,
  composite: LayoutListIcon,
};

/** A small, non-interactive picture of the block: a sketch for charts, an icon for the rest. */
function Preview({ p }: { p: PrimitivePayload }) {
  const sketch = sketchOf(p);
  if (sketch) {
    return (
      <svg aria-hidden viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-12 w-full overflow-visible">
        {sketch.kind === "bar" ? <BarSketch values={sketch.values} /> : null}
        {sketch.kind === "line" ? <LineSketch series={sketch.series} /> : null}
        {sketch.kind === "scatter" ? <ScatterSketch points={sketch.points} /> : null}
      </svg>
    );
  }
  const Icon =
    p.kind === "code" && p.output_type === "folium_html" ? MapIcon : (ICONS[p.kind] ?? ChartNoAxesColumnIcon);
  return (
    <div aria-hidden className="bg-muted text-muted-foreground grid h-12 place-items-center rounded-md">
      <Icon className="size-5" />
    </div>
  );
}

type CardProps = ComponentProps<"button"> & {
  p: PrimitivePayload;
  title: string;
  eyebrow: string | null;
};

/** The card is the dialog's trigger, so it must forward its ref and props. */
const Card = forwardRef<HTMLButtonElement, CardProps>(function Card({ p, title, eyebrow, className, ...props }, ref) {
  return (
    <button
      ref={ref}
      type="button"
      className={
        "bg-muted/60 hover:bg-accent focus-visible:ring-ring/50 group flex w-full flex-col gap-2 rounded-lg border p-2.5 text-left outline-none focus-visible:ring-[3px] dark:bg-card " +
        (className ?? "")
      }
      {...props}
    >
      <span className="flex w-full items-start gap-2">
        <span className="flex min-w-0 flex-1 flex-col">
          {eyebrow ? <span className="text-muted-foreground text-[11px] leading-tight">{eyebrow}</span> : null}
          <span className="text-[13px] leading-snug font-semibold [overflow-wrap:anywhere]">{title}</span>
        </span>
        <span className="text-muted-foreground group-hover:text-foreground flex shrink-0 items-center gap-1 text-[11px] font-medium">
          {COPY.chatBlockOpen}
          <Maximize2Icon aria-hidden className="size-3" />
        </span>
      </span>
      <Preview p={p} />
    </button>
  );
});

/** Compact density: a block too big for the column, as a card that opens the full block in a dialog. */
export function CompactBlock({ p, blockKey }: { p: PrimitivePayload; blockKey?: string }) {
  const own = blockTitle(p);
  const kind = kindLabel(p);
  const title = own ?? kind;
  return (
    <BlockDialog
      p={p}
      title={title}
      blockKey={blockKey}
      trigger={<Card p={p} title={title} eyebrow={own ? kind : null} />}
    />
  );
}
