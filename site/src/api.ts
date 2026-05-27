import type { Channel, Locale, Visit, Video, FlaggedSegment, Correction } from "./types";

const BASE = import.meta.env.BASE_URL; // "/cibobuono/" in prod

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}data/${path}`);
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return res.json();
}

/**
 * Fetch a list that may have been split into paged files by the pipeline.
 * If the primary file contains {"_pages": N}, fetches stem_0.json … stem_{N-1}.json
 * and concatenates them. Falls back to the primary file as a plain list otherwise.
 */
async function fetchPaged<T>(filename: string): Promise<T[]> {
  const data = await fetchJson<T[] | { _pages: number }>(filename);
  if (!Array.isArray(data) && typeof data === "object" && "_pages" in data) {
    const n = (data as { _pages: number })._pages;
    const stem = filename.replace(/\.json$/, "");
    const pages = await Promise.all(
      Array.from({ length: n }, (_, i) => fetchJson<T[]>(`${stem}_${i}.json`))
    );
    return pages.flat();
  }
  return data as T[];
}

export const fetchChannels = () => fetchJson<Channel[]>("channels.json");
export const fetchLocales = () => fetchPaged<Locale>("locales.json");
export const fetchVisits = () => fetchPaged<Visit>("visits.json");
export const fetchVideos = () => fetchJson<Video[]>("videos.json");
export const fetchFlagged = () => fetchJson<FlaggedSegment[]>("flagged_segments.json");
export const fetchCorrections = () => fetchJson<Correction[]>("corrections.json");
