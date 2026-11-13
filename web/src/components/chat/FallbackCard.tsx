/** The graceful-fallback card: why the turn could not answer and what to try. */
export default function FallbackCard({ reason, suggestion }: { reason: string; suggestion: string }) {
  return (
    <div role="alert" className="bg-card border-l-primary rounded-lg border border-l-[3px] px-3.5 py-3 [overflow-wrap:anywhere]">
      <p>{reason}</p>
      {suggestion ? <p className="text-muted-foreground mt-1.5 text-sm">{suggestion}</p> : null}
    </div>
  );
}
