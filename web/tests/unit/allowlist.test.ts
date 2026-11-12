import { describe, expect, it } from "vitest";
import { isAllowed } from "@/lib/allowlist";

describe("isAllowed", () => {
  it("matches_case_insensitively_and_trims", () => {
    const raw = " Ana@Fund.com , b@x.es";
    expect(isAllowed("ana@fund.com", raw)).toBe(true);
    expect(isAllowed("  ANA@FUND.COM ", raw)).toBe(true);
    expect(isAllowed("b@x.es", raw)).toBe(true);
    expect(isAllowed("c@x.es", raw)).toBe(false);
  });

  it("empty_or_unset_rejects_everyone", () => {
    for (const raw of [undefined, "", "  ", " , ,"]) {
      expect(isAllowed("ana@fund.com", raw)).toBe(false);
      expect(isAllowed("", raw)).toBe(false);
    }
  });

  it("null_email_rejected", () => {
    expect(isAllowed(null, "ana@fund.com")).toBe(false);
    expect(isAllowed(undefined, "ana@fund.com")).toBe(false);
    expect(isAllowed("", "ana@fund.com,")).toBe(false);
  });

  it("reads_env_by_default", () => {
    const prev = process.env.ROOSTER_ALLOWED_EMAILS;
    process.env.ROOSTER_ALLOWED_EMAILS = "ana@fund.com";
    try {
      expect(isAllowed("Ana@fund.com")).toBe(true);
    } finally {
      if (prev === undefined) delete process.env.ROOSTER_ALLOWED_EMAILS;
      else process.env.ROOSTER_ALLOWED_EMAILS = prev;
    }
  });
});
