"use client";

import { useEffect, useRef, useState } from "react";

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
  "On Schedule": "#2756c5",
  "Completed":   "#27ae60",
  "In Progress": "#f59e0b",
};

/* ── SVG pin factory ───────────────────────────────── */
function makePinSvg(color: string, active: boolean): string {
  const scale = active ? 1.3 : 1;
  const W     = Math.round(32 * scale);
  const H     = Math.round(44 * scale);
  const shadow = active 
    ? `drop-shadow(0 6px 12px ${color}88)`
    : `drop-shadow(0 3px 6px rgba(0,0,0,0.3))`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 32 44" style="filter:${shadow}">
    <path d="M16 1C7.716 1 1 7.716 1 16c0 10.5 15 27 15 27s15-16.5 15-27c0-8.284-6.716-15-15-15z" fill="${color}" stroke="white" stroke-width="2"/>
    <circle cx="16" cy="16" r="6" fill="white"/>
  </svg>`;
}

export default function IloiloMap({ projects, activeId, onPinClick }: IloiloMapProps) {
  const [mapReady, setMapReady] = useState(false);
  const mapRef       = useRef<HTMLDivElement>(null);
  const leafletMap   = useRef<import("leaflet").Map | null>(null);
  const markersRef   = useRef<Map<string, import("leaflet").Marker>>(new Map());

  /* ── Add a single marker ────────────────────────── */
  function addMarker(
    L:      typeof import("leaflet"),
    map:    import("leaflet").Map,
    p:      ProjectPin,
    active: boolean
  ) {
    const color  = STATUS_COLOR[p.status] ?? "#1264ae";
    const size   = active ? 50 : 36;
    const icon   = L.divIcon({
      html:        makePinSvg(color, active),
      className:   "",
      iconSize:    [size, Math.round(size * 1.375)],
      iconAnchor:  [size / 2, Math.round(size * 1.375)],
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

  /* ── Init map once ──────────────────────────────── */
  useEffect(() => {
    if (!mapRef.current) return;
    
    let isMounted = true;

    (async () => {
      try {
        const L = (await import("leaflet")).default;
        await import("leaflet/dist/leaflet.css");
        if (!isMounted) return;

        const el = mapRef.current as HTMLElement & { _leaflet_id?: null | string | number };
        if (el._leaflet_id) {
          el._leaflet_id = null;
          el.innerHTML = "";
        }

        const map = L.map(el, {
          center:             [10.95, 122.65],
          zoom:               10,
          minZoom:            8,
          maxBounds:          [[9.8, 121.0], [12.2, 124.0]],
          maxBoundsViscosity: 1.0,
          zoomControl:        false,
          attributionControl: true,
        });

        leafletMap.current = map;
        setMapReady(true);

        /* Standard bright map (CartoDB Voyager) */
        L.tileLayer(
          "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
          { maxZoom: 19, subdomains: "abcd", attribution: '&copy; OpenStreetMap &copy; CARTO' }
        ).addTo(map);

        /* Zoom controls */
        L.control.zoom({ position: "bottomright" }).addTo(map);

      } catch (err) {
        console.error("Leaflet init error:", err);
      }
    })();

    const activeMarkers = markersRef.current;
    return () => {
      isMounted = false;
      if (leafletMap.current) {
        leafletMap.current.remove();
        leafletMap.current = null;
        activeMarkers.clear();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);



  /* ── Sync markers when filtered projects change ── */
  useEffect(() => {
    if (!mapReady || !leafletMap.current) return;
    (async () => {
      const L = (await import("leaflet")).default;
      const map = leafletMap.current;
      if (!map) return;

      const projectIds = new Set(projects.map(p => p.id));
      for (const [id, marker] of markersRef.current.entries()) {
        if (!projectIds.has(id)) {
          marker.remove();
          markersRef.current.delete(id);
        }
      }

      projects.forEach(p => {
        if (!markersRef.current.has(p.id)) {
          addMarker(L, map, p, activeId === p.id);
        }
      });
    })();
  }, [projects, activeId, mapReady]);

  /* ── Update pins on activeId change ────────────── */
  useEffect(() => {
    if (!mapReady || !leafletMap.current) return;
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
          iconSize:    [size, Math.round(size * 1.375)],
          iconAnchor:  [size / 2, Math.round(size * 1.375)],
          popupAnchor: [0, -Math.round(size * 1.4)],
        }));
        if (active) {
          map.panTo(marker.getLatLng(), { animate: true, duration: 0.5 });
          marker.openPopup();
        }
      });
    })();
  }, [activeId, projects, mapReady]);

  /* ── Render ─────────────────────────────────────── */
  return (
    <div style={{ position: "relative", width: "100%", height: "100%", minHeight: 0 }}>
      {/* The actual Leaflet map */}
      <div ref={mapRef} style={{ width: "100%", height: "100%" }} />

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
          ["On Schedule", "#2756c5"],
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
