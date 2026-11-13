"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";

// Kept here rather than in copy.ts for now: copy.ts was being edited concurrently by another task.
const TEXT = {
  title: "Algo ha fallado",
  body: "No pudimos cargar esta página. Inténtalo de nuevo en unos segundos.",
  retry: "Reintentar",
} as const;

/** Error boundary for the signed-in area (also re-exported as the root boundary for layout errors). */
export default function AppError({ error, retry }: { error: unknown; retry: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);
  return (
    <div role="alert" className="mx-auto flex min-h-[50svh] max-w-sm flex-col items-center justify-center gap-3 p-6 text-center">
      <h1 className="text-xl font-semibold">{TEXT.title}</h1>
      <p className="text-muted-foreground text-sm">{TEXT.body}</p>
      <Button onClick={() => retry()}>{TEXT.retry}</Button>
    </div>
  );
}
