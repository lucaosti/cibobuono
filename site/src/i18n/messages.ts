/**
 * Typed message dictionaries for the CiboBuono web UI.
 *
 * Add new keys to BOTH `en` and `it`. The English dictionary is the source of
 * truth — every key in `en` MUST have a counterpart in `it` (enforced at
 * compile time via the Messages type).
 */

export type Language = "en" | "it";

/**
 * Message dictionary contract. Every locale must implement every field; the
 * compiler will reject any locale missing a key.
 */
export interface Messages {
  // App-wide
  appTitle: string;
  appTagline: string;
  documentTitle: string;

  // Header
  searchPlaceholder: string;
  filterByCityTitle: string;
  allCities: string;
  filterBySentimentTitle: string;
  allSentiments: string;
  sentimentPositive: string;
  sentimentNeutral: string;
  sentimentNegative: string;
  nearMe: string;
  nearMeTitle: string;
  km: (n: number) => string;
  venuesCount: (n: number) => string;
  geoErrorMarker: string;

  // Status bar
  loadingData: string;

  // Locale list
  noVenuesFound: string;
  noVenuesHint: string;
  videosCount: (n: number) => string;
  visitsCount: (n: number) => string;
  rating: string;
  ratingOutOf10: (rating: string) => string;

  // Actions
  googleMaps: string;
  edit: string;
  cancel: string;
  suggestCorrection: string;
  submitEdit: string;
  reportFalsePositive: string;

  // Edit form labels
  fieldName: string;
  fieldCity: string;
  fieldNotes: string;
  notesPlaceholder: string;

  // Issue-template strings
  issueRemoveTitle: (name: string) => string;
  issueEditTitle: (name: string) => string;
  issueRemoveHeader: string;
  issueEditHeader: string;
  issueFieldLocaleId: string;
  issueFieldCurrentName: string;
  issueFieldCity: string;
  issueFieldNotes: string;
  issueCorrectionEntryHeader: string;
  issueReasonFalsePositive: string;

  // Category labels (backend produces Italian raw strings)
  categoryLabel: (cat: string) => string;

  // Errors
  errorLoadFailed: string;
  errorGeolocationUnsupported: string;

  // Language toggle
  languageToggleLabel: string;
  languageEnglish: string;
  languageItalian: string;
}

export const en: Messages = {
  appTitle: "CiboBuono",
  appTagline: "Italian food venues, automatically extracted from YouTube",
  documentTitle: "CiboBuono — Italian food venues from YouTube",

  searchPlaceholder: "Search...",
  filterByCityTitle: "Filter by city",
  allCities: "All cities",
  filterBySentimentTitle: "Filter by sentiment",
  allSentiments: "All",
  sentimentPositive: "Positive",
  sentimentNeutral: "Neutral",
  sentimentNegative: "Negative",
  nearMe: "Near me",
  nearMeTitle: "Near me",
  km: (n: number) => `${n} km`,
  venuesCount: (n: number) => `${n} ${n === 1 ? "venue" : "venues"}`,
  geoErrorMarker: "!",

  loadingData: "Loading data...",

  noVenuesFound: "No venues found.",
  noVenuesHint: "The pipeline hasn't extracted any locales yet.",
  videosCount: (n: number) => `${n} ${n === 1 ? "video" : "videos"}`,
  visitsCount: (n: number) => `${n} ${n === 1 ? "visit" : "visits"}`,
  rating: "Rating",
  ratingOutOf10: (rating: string) => `${rating}/10`,

  googleMaps: "Google Maps",
  edit: "Edit",
  cancel: "Cancel",
  suggestCorrection: "Suggest correction",
  submitEdit: "Submit edit",
  reportFalsePositive: "Report false positive",

  fieldName: "Name",
  fieldCity: "City",
  fieldNotes: "Notes",
  notesPlaceholder: "What's wrong?",

  issueRemoveTitle: (name: string) => `[Correction] Remove: ${name}`,
  issueEditTitle: (name: string) => `[Correction] Edit: ${name}`,
  issueRemoveHeader: "## Correction: Remove false positive",
  issueEditHeader: "## Correction: Edit locale",
  issueFieldLocaleId: "**Locale ID:**",
  issueFieldCurrentName: "**Current name:**",
  issueFieldCity: "**City:**",
  issueFieldNotes: "**Notes:**",
  issueCorrectionEntryHeader: "### corrections.json entry",
  issueReasonFalsePositive: "False positive — not actually visited",

  categoryLabel: (cat: string) => {
    const map: Record<string, string> = {
      forno: "bakery",
      panificio: "bakery",
      ristorante: "restaurant",
      trattoria: "trattoria",
      osteria: "osteria",
      pizzeria: "pizzeria",
      bar: "bar",
      caffe: "café",
      caffè: "café",
      pasticceria: "pastry shop",
      gelateria: "ice cream shop",
      street_food: "street food",
      mercato: "market",
      enoteca: "wine bar",
      rosticceria: "rotisserie",
      braceria: "grill",
      pescheria: "fish restaurant",
    };
    return map[cat.toLowerCase()] ?? cat;
  },

  errorLoadFailed: "Failed to load data",
  errorGeolocationUnsupported: "Geolocation not supported",

  languageToggleLabel: "Language",
  languageEnglish: "EN",
  languageItalian: "IT",
};

export const it: Messages = {
  appTitle: "CiboBuono",
  appTagline: "Locali italiani estratti automaticamente da YouTube",
  documentTitle: "CiboBuono — Locali italiani da YouTube",

  searchPlaceholder: "Cerca...",
  filterByCityTitle: "Filtra per città",
  allCities: "Tutte le città",
  filterBySentimentTitle: "Filtra per sentiment",
  allSentiments: "Tutti",
  sentimentPositive: "Positivo",
  sentimentNeutral: "Neutro",
  sentimentNegative: "Negativo",
  nearMe: "Vicino a me",
  nearMeTitle: "Vicino a me",
  km: (n: number) => `${n} km`,
  venuesCount: (n: number) => `${n} ${n === 1 ? "locale" : "locali"}`,
  geoErrorMarker: "!",

  loadingData: "Caricamento dati...",

  noVenuesFound: "Nessun locale trovato.",
  noVenuesHint: "La pipeline non ha ancora estratto nessun locale.",
  videosCount: (n: number) => `${n} ${n === 1 ? "video" : "video"}`,
  visitsCount: (n: number) => `${n} ${n === 1 ? "visita" : "visite"}`,
  rating: "Voto",
  ratingOutOf10: (rating: string) => `${rating}/10`,

  googleMaps: "Google Maps",
  edit: "Modifica",
  cancel: "Annulla",
  suggestCorrection: "Suggerisci correzione",
  submitEdit: "Invia modifica",
  reportFalsePositive: "Segnala falso positivo",

  fieldName: "Nome",
  fieldCity: "Città",
  fieldNotes: "Note",
  notesPlaceholder: "Che cosa non va?",

  issueRemoveTitle: (name: string) => `[Correzione] Rimuovi: ${name}`,
  issueEditTitle: (name: string) => `[Correzione] Modifica: ${name}`,
  issueRemoveHeader: "## Correzione: Rimuovi falso positivo",
  issueEditHeader: "## Correzione: Modifica locale",
  issueFieldLocaleId: "**ID locale:**",
  issueFieldCurrentName: "**Nome attuale:**",
  issueFieldCity: "**Città:**",
  issueFieldNotes: "**Note:**",
  issueCorrectionEntryHeader: "### Voce per corrections.json",
  issueReasonFalsePositive: "Falso positivo — non effettivamente visitato",

  categoryLabel: (cat: string) => {
    const map: Record<string, string> = {
      forno: "forno",
      panificio: "panificio",
      ristorante: "ristorante",
      trattoria: "trattoria",
      osteria: "osteria",
      pizzeria: "pizzeria",
      bar: "bar",
      caffe: "caffè",
      caffè: "caffè",
      pasticceria: "pasticceria",
      gelateria: "gelateria",
      street_food: "street food",
      mercato: "mercato",
      enoteca: "enoteca",
      rosticceria: "rosticceria",
      braceria: "braceria",
      pescheria: "pescheria",
    };
    return map[cat.toLowerCase()] ?? cat;
  },

  errorLoadFailed: "Impossibile caricare i dati",
  errorGeolocationUnsupported: "Geolocalizzazione non supportata",

  languageToggleLabel: "Lingua",
  languageEnglish: "EN",
  languageItalian: "IT",
};

export const messages: Record<Language, Messages> = { en, it };

export type MessageKey = keyof Messages;

/** Available languages for the language switcher. */
export const LANGUAGES: { code: Language; label: string }[] = [
  { code: "en", label: "EN" },
  { code: "it", label: "IT" },
];
