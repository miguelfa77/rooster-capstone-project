// @vitest-environment node
// Contract check: the shared Python fixture must be assignable to the generated
// TypeScript types. A plain `import fixture from "....json"` widens every string
// literal (kind: "text" -> string), so instead we inline the JSON into an in-memory
// TS program as a contextually typed literal (`satisfies PrimitivePayload[]`) and
// assert the compiler reports no diagnostics.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import ts from "typescript";
import { describe, expect, it } from "vitest";

const webRoot = path.resolve(__dirname, "../..");
const fixturePath = path.resolve(webRoot, "../tests/fixtures/primitive_payloads.json");
const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as {
  payloads: { kind: string }[];
  folium_code: unknown;
};

function typeErrors(source: string): string[] {
  const file = path.join(webRoot, "tests/unit/__fixture_check__.ts");
  const options: ts.CompilerOptions = {
    strict: true,
    noEmit: true,
    skipLibCheck: true,
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
  };
  const host = ts.createCompilerHost(options);
  const readFile = host.readFile;
  host.readFile = (f) => (path.resolve(f) === file ? source : readFile(f));
  const fileExists = host.fileExists;
  host.fileExists = (f) => path.resolve(f) === file || fileExists(f);
  const program = ts.createProgram([file], options, host);
  return ts
    .getPreEmitDiagnostics(program)
    .map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n"));
}

describe("generated contract types", () => {
  it("fixture_has_all_ten_kinds", () => {
    expect(new Set(fixture.payloads.map((p) => p.kind))).toEqual(
      new Set(["text", "kpi", "table", "code", "choropleth", "bar", "scatter", "line", "point_map", "composite"]),
    );
  });

  it("fixture_is_assignable_to_PrimitivePayload", () => {
    const typesPath = path.join(webRoot, "src/lib/types").replaceAll("\\", "/");
    const source = [
      `import type { PrimitivePayload } from "${typesPath}";`,
      `export const payloads = ${JSON.stringify(fixture.payloads)} satisfies PrimitivePayload[];`,
      `export const folium = ${JSON.stringify(fixture.folium_code)} satisfies PrimitivePayload;`,
    ].join("\n");
    expect(typeErrors(source)).toEqual([]);
  });

  it("rejects_a_bad_payload", () => {
    const typesPath = path.join(webRoot, "src/lib/types").replaceAll("\\", "/");
    const source = [
      `import type { PrimitivePayload } from "${typesPath}";`,
      `export const bad = [{ kind: "kpi", label: 1 }] satisfies PrimitivePayload[];`,
    ].join("\n");
    expect(typeErrors(source).length).toBeGreaterThan(0);
  });

  it("check_types_exits_0", () => {
    expect(() =>
      execFileSync(process.execPath, [path.join(webRoot, "scripts/gen-types.mjs"), "--check"], { cwd: webRoot }),
    ).not.toThrow();
  });
});
