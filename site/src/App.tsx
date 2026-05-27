import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Locale, Visit, LocaleWithVisits, Correction } from "./types";
import { fetchLocales, fetchVisits, fetchCorrections } from "./api";
import Header from "./components/Header";
import MapView from "./components/MapView";
import LocaleList from "./components/LocaleList";
import StatusBar from "./components/StatusBar";
import { useT } from "./i18n/useLanguage";
import "./App.css";

function haversineKm(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
): number {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function applyCorrections(
  locales: Locale[],
  corrections: Correction[],
): Locale[] {
  const corrMap = new Map(corrections.map((c) => [c.locale_id, c]));
  return locales
    .filter((l) => {
      const c = corrMap.get(l.locale_id);
      return !c || c.type !== "hide";
    })
    .map((l) => {
      const c = corrMap.get(l.locale_id);
      if (!c || c.type !== "edit" || !c.overrides) return l;
      return {
        ...l,
        name: c.overrides.name ?? l.name,
        city: c.overrides.city ?? l.city,
      };
    });
}

function enrichLocales(
  locales: Locale[],
  visits: Visit[],
  corrections: Correction[],
): LocaleWithVisits[] {
  const visitsByLocale = new Map<string, Visit[]>();
  for (const v of visits) {
    const arr = visitsByLocale.get(v.locale_id) ?? [];
    arr.push(v);
    visitsByLocale.set(v.locale_id, arr);
  }

  const corrMap = new Map(corrections.map((c) => [c.locale_id, c]));

  return locales.map((loc) => {
    const locVisits = visitsByLocale.get(loc.locale_id) ?? [];
    const corr = corrMap.get(loc.locale_id);

    const ratings = locVisits
      .map((v) => v.rating)
      .filter((r): r is string => r != null && r !== "");
    let avgRating: string | null = ratings.length > 0 ? ratings[0] : null;

    // Override rating from correction
    if (corr?.type === "edit" && corr.overrides?.rating !== undefined) {
      avgRating = corr.overrides.rating != null ? String(corr.overrides.rating) : null;
    }

    const sentCount = { positive: 0, neutral: 0, negative: 0 };
    for (const v of locVisits) sentCount[v.sentiment]++;
    let dominant = (
      Object.entries(sentCount) as [keyof typeof sentCount, number][]
    ).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "neutral";

    // Override sentiment from correction
    if (corr?.type === "edit" && corr.overrides?.sentiment) {
      dominant = corr.overrides.sentiment;
    }

    return { ...loc, visits: locVisits, avgRating, dominantSentiment: dominant };
  });
}

export default function App() {
  const t = useT();
  const tRef = useRef(t);
  useEffect(() => { tRef.current = t; });

  const [locales, setLocales] = useState<Locale[]>([]);
  const [visits, setVisits] = useState<Visit[]>([]);
  const [corrections, setCorrections] = useState<Correction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [sentimentFilter, setSentimentFilter] = useState("");
  const [cityFilter, setCityFilter] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [nearMeKm, setNearMeKm] = useState<number | null>(null);
  const [userPos, setUserPos] = useState<{ lat: number; lon: number } | null>(
    null,
  );
  const [geoError, setGeoError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [l, v, c] = await Promise.all([
          fetchLocales(),
          fetchVisits(),
          fetchCorrections().catch(() => [] as Correction[]),
        ]);
        setLocales(l);
        setVisits(v);
        setCorrections(c);
      } catch (e) {
        setError(e instanceof Error ? e.message : tRef.current.errorLoadFailed);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []); // run once on mount — language changes must not re-fetch

  useEffect(() => {
    if (nearMeKm == null || userPos != null) return;
    if (!navigator.geolocation) {
      setGeoError(tRef.current.errorGeolocationUnsupported);
      setNearMeKm(null);
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setUserPos({ lat: pos.coords.latitude, lon: pos.coords.longitude });
        setGeoError(null);
      },
      (err) => {
        setGeoError(err.message);
        setNearMeKm(null);
      },
    );
  }, [nearMeKm, userPos]);

  const correctedLocales = useMemo(
    () => applyCorrections(locales, corrections),
    [locales, corrections],
  );

  const enriched = useMemo(
    () => enrichLocales(correctedLocales, visits, corrections),
    [correctedLocales, visits, corrections],
  );

  const cities = useMemo(() => {
    const set = new Set(enriched.map((l) => l.city).filter(Boolean));
    return [...set].sort();
  }, [enriched]);

  const filtered = useMemo(() => {
    let result = enriched;
    if (sentimentFilter) {
      result = result.filter((l) => l.dominantSentiment === sentimentFilter);
    }
    if (cityFilter) {
      result = result.filter(
        (l) => l.city.toLowerCase() === cityFilter.toLowerCase(),
      );
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (l) =>
          l.name.toLowerCase().includes(q) ||
          l.city.toLowerCase().includes(q) ||
          l.category.some((c) => c.toLowerCase().includes(q)),
      );
    }
    if (nearMeKm != null && userPos) {
      result = result.filter(
        (l) => haversineKm(userPos.lat, userPos.lon, l.lat, l.lon) <= nearMeKm,
      );
    }
    return result;
  }, [enriched, sentimentFilter, cityFilter, searchQuery, nearMeKm, userPos]);

  const handleSelect = useCallback(
    (id: string) => setSelected((prev) => (prev === id ? null : id)),
    [],
  );

  return (
    <div className="app">
      <Header
        total={filtered.length}
        sentimentFilter={sentimentFilter}
        onSentimentChange={setSentimentFilter}
        cityFilter={cityFilter}
        onCityChange={setCityFilter}
        cities={cities}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        nearMeKm={nearMeKm}
        onNearMeChange={setNearMeKm}
        geoError={geoError}
      />

      <StatusBar loading={loading} error={error} />

      <main className="content">
        <aside className="sidebar">
          <LocaleList
            locales={filtered}
            selected={selected}
            onSelect={handleSelect}
          />
        </aside>
        <section className="map-area">
          <MapView
            locales={filtered}
            selected={selected}
            onSelect={handleSelect}
          />
        </section>
      </main>
    </div>
  );
}
