import type { Channel, Locale, Visit, Video, FlaggedSegment, Correction } from "./types";

const BASE = import.meta.env.BASE_URL; // "/cibobuono/" in prod

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}data/${path}`);
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return res.json();
}

export const fetchChannels = () => fetchJson<Channel[]>("channels.json");
export const fetchLocales = () => fetchJson<Locale[]>("locales.json");
export const fetchVisits = () => fetchJson<Visit[]>("visits.json");
export const fetchVideos = () => fetchJson<Video[]>("videos.json");
export const fetchFlagged = () => fetchJson<FlaggedSegment[]>("flagged_segments.json");
export const fetchCorrections = () => fetchJson<Correction[]>("corrections.json");
