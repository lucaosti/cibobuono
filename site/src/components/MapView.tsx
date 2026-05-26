import { useEffect, useMemo } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import type { LocaleWithVisits } from "../types";
import { useT } from "../i18n/useLanguage";

import "leaflet/dist/leaflet.css";

/* Fix default marker icon path broken by bundlers */
import iconUrl from "leaflet/dist/images/marker-icon.png";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";
L.Icon.Default.mergeOptions({ iconUrl, iconRetinaUrl, shadowUrl });

const SENTIMENT_COLORS: Record<string, string> = {
  positive: "#4caf50",
  neutral: "#ff9800",
  negative: "#f44336",
};

function sentimentIcon(sentiment: string) {
  const color = SENTIMENT_COLORS[sentiment] ?? "#666";
  return L.divIcon({
    className: "custom-marker",
    html: `<div style="
      width:28px;height:28px;border-radius:50%;
      background:${color};border:3px solid #fff;
      box-shadow:0 2px 6px rgba(0,0,0,.35);
    "></div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    popupAnchor: [0, -16],
  });
}

/** Auto-fit bounds when locales change */
function FitBounds({ locales }: { locales: LocaleWithVisits[] }) {
  const map = useMap();
  useEffect(() => {
    if (locales.length === 0) return;
    const bounds = L.latLngBounds(locales.map((l) => [l.lat, l.lon]));
    map.fitBounds(bounds, { padding: [40, 40], maxZoom: 14 });
  }, [locales, map]);
  return null;
}

interface Props {
  locales: LocaleWithVisits[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export default function MapView({ locales, selected, onSelect }: Props) {
  const t = useT();
  const center = useMemo<[number, number]>(() => {
    if (locales.length === 0) return [41.9, 12.5]; // Roma default
    const lat = locales.reduce((s, l) => s + l.lat, 0) / locales.length;
    const lon = locales.reduce((s, l) => s + l.lon, 0) / locales.length;
    return [lat, lon];
  }, [locales]);

  return (
    <MapContainer
      center={center}
      zoom={12}
      style={{ height: "100%", width: "100%" }}
      scrollWheelZoom
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FitBounds locales={locales} />

      {locales.map((loc) => (
        <Marker
          key={loc.locale_id}
          position={[loc.lat, loc.lon]}
          icon={sentimentIcon(loc.dominantSentiment)}
          eventHandlers={{
            click: () => onSelect(loc.locale_id),
          }}
          opacity={selected && selected !== loc.locale_id ? 0.4 : 1}
        >
          <Popup>
            <strong>{loc.name}</strong>
            <br />
            {loc.city && <span>{loc.city}</span>}
            {loc.avgRating != null && (
              <>
                <br />
                {t.rating}: {t.ratingOutOf10(loc.avgRating)}
              </>
            )}
            <br />
            {t.visitsCount(loc.visits.length)}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
