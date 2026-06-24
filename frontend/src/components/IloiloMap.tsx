"use client";

import { useEffect, useRef } from "react";

/* ── Types ─────────────────────────────────────────── */
export interface ProjectPin {
  id: string;
  name: string;
  municipality: string;
  status: string;
  risk: string;
  lat: number;
  lng: number;
}

interface IloiloMapProps {
  projects: ProjectPin[];
  activeId: string | null;
  onPinClick: (id: string) => void;
}

/* ── Status → pin color ────────────────────────────── */
const STATUS_COLOR: Record<string, string> = {
  "Delayed":     "#e74c3c",
  "On Schedule": "#f39c12",
  "Completed":   "#27ae60",
  "In Progress": "#f59e0b",
};

/* ── SVG pin factory ───────────────────────────────── */
function makePinSvg(color: string, active: boolean): string {
  const scale = active ? 1.4 : 1;
  const W     = Math.round(36 * scale);
  const H     = Math.round(48 * scale);
  const glow  = active
    ? `drop-shadow(0 0 8px ${color}88) drop-shadow(0 4px 12px rgba(0,0,0,0.5))`
    : "drop-shadow(0 3px 6px rgba(0,0,0,0.35))";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 36 48" style="filter:${glow}">
    <path d="M18 1C10.268 1 4 7.268 4 15c0 11 14 32 14 32S32 26 32 15C32 7.268 25.732 1 18 1z"
          fill="${color}" stroke="white" stroke-width="2.5"/>
    <circle cx="18" cy="15" r="7" fill="white" fill-opacity="0.95"/>
    <circle cx="18" cy="15" r="3.5" fill="${color}"/>
  </svg>`;
}

/* ── Map ID (stable, module-level) ─────────────────── */
const MAP_ID = "maagap-iloilo-map";

/* ─────────────────────────────────────────────────────
   CSS injected once into <head>.
   Targets ONLY the tile pane inside our specific map div
   → markers (.leaflet-marker-pane) are NOT affected.

   Why sepia first?
   sepia(1) converts near-white (#F5F5F5 land) to warm
   yellow (#FFD580-ish) which actually has a hue to rotate.
   hue-rotate(185°) then shifts yellow(60°) → blue(245°).
   ───────────────────────────────────────────────────── */
const BLUE_TILE_CSS = `
  #${MAP_ID} .leaflet-tile-pane {
    filter: invert(1) hue-rotate(200deg) saturate(0.6) brightness(1.1);
  }
`;

function injectStyle() {
  if (typeof document === "undefined") return;
  if (document.getElementById("maagap-map-style")) return;
  const el = document.createElement("style");
  el.id = "maagap-map-style";
  el.textContent = BLUE_TILE_CSS;
  document.head.appendChild(el);
}

export default function IloiloMap({ projects, activeId, onPinClick }: IloiloMapProps) {
  const mapRef       = useRef<HTMLDivElement>(null);
  const leafletMap   = useRef<import("leaflet").Map | null>(null);
  const markersRef   = useRef<Map<string, import("leaflet").Marker>>(new Map());
  const initingRef   = useRef(false);   // prevents async race in Strict Mode

  /* ── Inject blue-tile CSS once ──────────────────── */
  useEffect(() => { injectStyle(); }, []);

  /* ── Init map once ──────────────────────────────── */
  useEffect(() => {
    if (!mapRef.current || leafletMap.current || initingRef.current) return;
    initingRef.current = true;

    (async () => {
      try {
        const L = (await import("leaflet")).default;
        await import("leaflet/dist/leaflet.css");

        /* Safety: cleanup may have run while we awaited */
        if (!mapRef.current || leafletMap.current) return;

        /* Clear any stale _leaflet_id (React Strict Mode / HMR) */
        const el = mapRef.current as HTMLElement & { _leaflet_id?: number };
        if (el._leaflet_id) delete el._leaflet_id;

        const map = L.map(el, {
          center:             [10.75, 122.56],
          zoom:               10,
          zoomControl:        false,
          attributionControl: false,
        });

        leafletMap.current = map;

        /* CartoDB Dark Matter + invert(1) = proven blue map trick:
           Dark Matter is black/dark-navy tiles.
           invert(1) flips black→white, white→black, giving
           a light-gray map with inverted colors.
           hue-rotate(200°) then tints everything to blue.
           Result: light-blue land, slightly-darker-blue water,
           white roads — matching the reference design exactly. */
        L.tileLayer(
          "https://{s}.basemaps.cartocdn.com/dark_matter_nolabels/{z}/{x}/{y}{r}.png",
          { maxZoom: 19, subdomains: "abcd" }
        ).addTo(map);

        /* Labels layer on top — not filtered, stays readable */
        L.tileLayer(
          "https://{s}.basemaps.cartocdn.com/dark_matter_only_labels/{z}/{x}/{y}{r}.png",
          { maxZoom: 19, subdomains: "abcd", opacity: 0.6 }
        ).addTo(map);

        /* Zoom controls */
        L.control.zoom({ position: "bottomright" }).addTo(map);

        /* Project markers */
        projects.forEach(p => addMarker(L, map, p, false));
      } finally {
        initingRef.current = false;
      }
    })();

    return () => {
      initingRef.current = false;
      if (leafletMap.current) {
        leafletMap.current.remove();
        leafletMap.current = null;
        markersRef.current.clear();
      }
      if (mapRef.current) {
        const el = mapRef.current as HTMLElement & { _leaflet_id?: number };
        delete el._leaflet_id;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── Add a single marker ────────────────────────── */
  function addMarker(
    L:      typeof import("leaflet").default,
    map:    import("leaflet").Map,
    p:      ProjectPin,
    active: boolean
  ) {
    const color  = STATUS_COLOR[p.status] ?? "#1264ae";
    const size   = active ? 50 : 36;
    const icon   = L.divIcon({
      html:        makePinSvg(color, active),
      className:   "",
      iconSize:    [size, Math.round(size * 1.33)],
      iconAnchor:  [size / 2, Math.round(size * 1.33)],
      popupAnchor: [0, -Math.round(size * 1.4)],
    });

    const marker = L.marker([p.lat, p.lng], { icon })
      .addTo(map)
      .bindPopup(`
        <div style="font-family:Inter,system-ui,sans-serif;min-width:200px;padding:2px 0">
          <div style="font-weight:700;font-size:0.84rem;color:#1b3a5e;margin-bottom:5px;line-height:1.3">${p.name}</div>
          <div style="display:flex;align-items:center;gap:5px;font-size:0.72rem;color:#6b7a8d;margin-bottom:8px">
            <span>📍</span><span>${p.municipality}</span>
          </div>
          <span style="display:inline-block;padding:3px 12px;border-radius:50px;background:${color};color:white;font-size:0.68rem;font-weight:700;letter-spacing:.03em">${p.status}</span>
        </div>
      `, { offset: [0, -6] })
      .on("click", () => onPinClick(p.id));

    markersRef.current.set(p.id, marker);
  }

  /* ── Update pins on activeId change ────────────── */
  useEffect(() => {
    if (!leafletMap.current) return;
    (async () => {
      const L   = (await import("leaflet")).default;
      const map = leafletMap.current;
      if (!map) return;

      markersRef.current.forEach((marker, id) => {
        const proj   = projects.find(p => p.id === id);
        if (!proj) return;
        const color  = STATUS_COLOR[proj.status] ?? "#1264ae";
        const active = id === activeId;
        const size   = active ? 50 : 36;
        marker.setIcon(L.divIcon({
          html:        makePinSvg(color, active),
          className:   "",
          iconSize:    [size, Math.round(size * 1.33)],
          iconAnchor:  [size / 2, Math.round(size * 1.33)],
          popupAnchor: [0, -Math.round(size * 1.4)],
        }));
        if (active) {
          map.panTo(marker.getLatLng(), { animate: true, duration: 0.5 });
          marker.openPopup();
        }
      });
    })();
  }, [activeId, projects]);

  /* ── Render ─────────────────────────────────────── */
  return (
    <div style={{ position: "relative", width: "100%", height: "100%", minHeight: 0 }}>
      {/* The actual Leaflet map — ID must match MAP_ID constant above */}
      <div id={MAP_ID} ref={mapRef} style={{ width: "100%", height: "100%" }} />

      {/* Legend — bottom left, outside the filtered tile pane */}
      <div style={{
        position: "absolute", bottom: "1rem", left: "1rem", zIndex: 1000,
        background: "rgba(255,255,255,0.93)", backdropFilter: "blur(8px)",
        borderRadius: "10px", padding: "0.6rem 0.9rem",
        boxShadow: "0 2px 12px rgba(0,0,0,0.14)", border: "1px solid #dce8f5",
        display: "flex", flexDirection: "column", gap: "0.4rem",
        pointerEvents: "none",
      }}>
        {([
          ["Delayed",     "#e74c3c"],
          ["On Schedule", "#f39c12"],
          ["Completed",   "#27ae60"],
          ["In Progress", "#f59e0b"],
        ] as [string, string][]).map(([label, color]) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <div style={{
              width: 10, height: 10, borderRadius: "50%",
              background: color, flexShrink: 0,
              boxShadow: `0 0 4px ${color}88`,
            }} />
            <span style={{
              fontSize: "0.68rem", fontWeight: 600,
              color: "#4a5a6a", fontFamily: "Inter,system-ui,sans-serif",
            }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
