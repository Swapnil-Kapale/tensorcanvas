"use client";

import { Suspense } from "react";
import { useParams } from "next/navigation";
import { LayerEditor } from "@/components/build/edit/LayerEditor";

function EditLayer() {
  const params = useParams<{ nodeId: string }>();
  return <LayerEditor nodeId={params.nodeId} />;
}

export default function EditLayerPage() {
  return (
    <Suspense fallback={<div className="p-8 text-neutral-400">Loading…</div>}>
      <EditLayer />
    </Suspense>
  );
}
