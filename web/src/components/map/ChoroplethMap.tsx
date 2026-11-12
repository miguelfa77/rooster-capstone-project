"use client";

// Placeholder until Task 5 builds the real map; keeps Primitive's dynamic import resolvable.
import EmptyState from "@/components/primitives/EmptyState";
import type { ChoroplethPayload } from "@/lib/types";

export default function ChoroplethMap(props: { p: ChoroplethPayload }) {
  void props; // real map arrives in Task 5
  return <EmptyState />;
}
