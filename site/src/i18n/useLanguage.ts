/**
 * Hooks for reading the active language and the translated messages
 * dictionary. Kept in a separate file so LanguageContext.tsx exports only the
 * provider component (Fast Refresh requirement).
 */
import { createContext, useContext } from "react";
import { en, messages, type Language, type Messages } from "./messages";

export interface LanguageContextValue {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: Messages;
}

export const LanguageContext = createContext<LanguageContextValue | null>(null);

/** Read the active language + setter. Throws if used outside the provider. */
export function useLanguage(): LanguageContextValue {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useLanguage must be used inside <LanguageProvider>");
  }
  return ctx;
}

/** Shortcut hook returning just the translated messages dictionary.
 *
 * During SSR / before the provider is mounted we degrade to the English
 * dictionary instead of throwing — useful for static prerendering tools. */
export function useT(): Messages {
  const ctx = useContext(LanguageContext);
  return ctx?.t ?? en;
}

/** Lookup the full message dictionary for an arbitrary language code.
 * Falls back to English when the code is unknown. */
export function getMessagesFor(lang: Language | string): Messages {
  return (messages as Record<string, Messages>)[lang] ?? en;
}
