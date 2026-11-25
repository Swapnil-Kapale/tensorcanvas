"use client";

import { Suspense } from "react";
import { useParams, useSearchParams } from "next/navigation";
import { LayerEditor } from "@/components/build/edit/LayerEditor";

function EditLayer() {
  const params = useParams<{ nodeId: string }>();
  const search = useSearchParams();
  const mode = search.get("mode") === "live" ? "live" : "initial";
  return <LayerEditor nodeId={params.nodeId} mode={mode} />;
}

export default function EditLayerPage() {
  return (
    <Suspense fallback={<div className="p-8 text-neutral-400">Loading…</div>}>
      <EditLayer />
    </Suspense>
  );
}
