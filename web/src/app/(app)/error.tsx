"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { COPY } from "@/lib/copy";

/** Error boundary for the signed-in area (also re-exported as the root boundary for layout errors). */
export default function AppError({ error, retry }: { error: unknown; retry: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);
  return (
    <div role="alert" className="mx-auto flex min-h-[50svh] max-w-sm flex-col items-center justify-center gap-3 p-6 text-center">
      <h1 className="text-xl font-semibold">{COPY.errorTitle}</h1>
      <p className="text-muted-foreground text-sm">{COPY.errorBody}</p>
      <Button onClick={() => retry()}>{COPY.errorRetry}</Button>
    </div>
  );
}
