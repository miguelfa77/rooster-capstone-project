"use client";

// Placeholder until Task 5 builds the real map; keeps Primitive's dynamic import resolvable.
import EmptyState from "@/components/primitives/EmptyState";
import type { PointMapPayload } from "@/lib/types";

export default function PointMap(props: { p: PointMapPayload }) {
  void props; // real map arrives in Task 5
  return <EmptyState />;
}
