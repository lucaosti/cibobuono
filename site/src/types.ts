/** Data types matching the Python pipeline JSON schemas */

export interface Channel {
  channel_id: string;
  name: string;
  url: string;
  description: string;
  rubriche: string[];
}

export interface Locale {
  locale_id: string;
  name: string;
  aliases: string[];
  address: string;
  city: string;
  lat: number;
  lon: number;
  category: string[];
  google_maps_url: string;
}

export interface Visit {
  visit_id: string;
  locale_id: string;
  video_id: string;
  channel_id: string;
  timestamp_start: string;
  timestamp_end: string;
  youtube_url: string;
  rating: string | null;
  sentiment: "positive" | "neutral" | "negative";
  notes: string;
  rubrica: string;
  llm_confidence: number;
  extraction_date: string;
  date: string;
}

export interface Video {
  video_id: string;
  channel_id: string;
  title: string;
  url: string;
  publish_date: string;
  processed_date: string;
  status: "pending" | "processed" | "errored";
}

export interface FlaggedSegment {
  video_id: string;
  channel_id: string;
  timestamp_start: string;
  timestamp_end: string;
  youtube_url: string;
  reason: string;
  extracted_text: string;
  llm_confidence: number;
  reviewed_by_human: boolean;
  reviewed_date: string | null;
  locale_name: string | null;
  rating: string | null;
  city: string | null;
}

/** Manual correction entry — stored in corrections.json */
export interface Correction {
  locale_id: string;
  type: "hide" | "edit";
  reason?: string;
  overrides?: {
    name?: string;
    city?: string;
    rating?: number | null;
    sentiment?: "positive" | "neutral" | "negative";
  };
}

/** Enriched locale with its visits for display */
export interface LocaleWithVisits extends Locale {
  visits: Visit[];
  avgRating: string | null;
  dominantSentiment: "positive" | "neutral" | "negative";
  hidden?: boolean;
}

/** GitHub repo config for issue submission */
export const GITHUB_REPO = "lucaosti/cibobuono";
