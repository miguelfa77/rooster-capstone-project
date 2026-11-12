// Invite-only sign-in: ROOSTER_ALLOWED_EMAILS is a comma-separated list, compared
// case-insensitively after trimming. An empty or unset list lets nobody in.
export function isAllowed(email: string | null | undefined, raw = process.env.ROOSTER_ALLOWED_EMAILS): boolean {
  const wanted = email?.trim().toLowerCase();
  if (!wanted || !raw) return false;
  return raw.split(",").some((entry) => entry.trim().toLowerCase() === wanted);
}
