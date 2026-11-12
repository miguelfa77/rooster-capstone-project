import { describe, expect, it } from "vitest";
import { safeHttpUrl } from "@/lib/url";

describe("safeHttpUrl", () => {
  it("accepts_http_and_https", () => {
    expect(safeHttpUrl("https://www.idealista.com/inmueble/1/")).toBe("https://www.idealista.com/inmueble/1/");
    expect(safeHttpUrl("http://example.com")).toBe("http://example.com/");
  });

  it("rejects_other_values", () => {
    for (const v of ["javascript:alert(1)", "data:text/html,x", "/relative", "texto", "", null, undefined, 42]) {
      expect(safeHttpUrl(v)).toBeNull();
    }
  });
});
