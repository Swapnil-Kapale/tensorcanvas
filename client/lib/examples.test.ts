import { describe, expect, it } from "vitest";

import { annExample, fileToGraph, graphToFile, NET_FORMAT, NET_VERSION } from "@/lib/examples";

describe(".tcnet.json round-trip", () => {
  it("round-trips the ANN example through JSON without losing nodes, edges, or params", () => {
    const graph = annExample();
    const file = graphToFile(graph.nodes, graph.edges, "mlp");
    expect(file.format).toBe(NET_FORMAT);
    expect(file.version).toBe(NET_VERSION);
    expect(file.name).toBe("mlp");

    // Simulate saving to disk and loading it back.
    const back = fileToGraph(JSON.parse(JSON.stringify(file)) as unknown);
    expect(back.nodes.map((n) => n.data.type)).toEqual(graph.nodes.map((n) => n.data.type));
    expect(back.nodes.map((n) => n.data.params)).toEqual(graph.nodes.map((n) => n.data.params));
    expect(back.nodes.map((n) => n.position)).toEqual(graph.nodes.map((n) => n.position));
    expect(back.edges.map((e) => [e.source, e.target])).toEqual(
      graph.edges.map((e) => [e.source, e.target]),
    );
  });

  it("rejects a file with the wrong format", () => {
    expect(() =>
      fileToGraph({ format: "not-tensorcanvas", version: 1, name: "x", nodes: [], edges: [] }),
    ).toThrow(/Not a TensorCanvas network file/);
  });

  it("writes node positions as whole pixels in the file", () => {
    const file = graphToFile(annExample().nodes, annExample().edges, "mlp");
    for (const n of file.nodes) {
      expect(Number.isInteger(n.position.x)).toBe(true);
      expect(Number.isInteger(n.position.y)).toBe(true);
    }
  });
});
