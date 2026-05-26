/**
 * Bilingual support for the CiboBuono web UI.
 *
 * Zero-dependency: just a React context. Default language is detected from
 * navigator.language; user choice persists in localStorage and overrides the
 * auto-detected value on subsequent loads.
 *
 * Hooks (useLanguage / useT) live in ./useLanguage.ts so this file exports
 * only the provider component, satisfying React Fast Refresh's
 * one-component-per-file requirement.
 */
import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import { messages, type Language } from "./messages";
import { LanguageContext, type LanguageContextValue } from "./useLanguage";

const STORAGE_KEY = "cibobuono.lang";

function detectInitialLanguage(): Language {
  if (typeof window === "undefined") return "en";
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "en" || stored === "it") return stored;
  } catch {
    /* localStorage may be unavailable in private mode — fall through */
  }
  const navLang = window.navigator.language?.toLowerCase() ?? "";
  if (navLang.startsWith("it")) return "it";
  return "en";
}

interface LanguageProviderProps {
  children: ReactNode;
}

export function LanguageProvider({ children }: LanguageProviderProps) {
  const [language, setLanguageState] = useState<Language>(detectInitialLanguage);

  const setLanguage = useCallback((lang: Language) => {
    setLanguageState(lang);
    try {
      window.localStorage.setItem(STORAGE_KEY, lang);
    } catch {
      /* ignore — non-fatal */
    }
  }, []);

  // Sync <html lang> and document.title so screen readers, search engines,
  // and browser UI all reflect the active language.
  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.lang = language;
    document.title = messages[language].documentTitle;
  }, [language]);

  const value = useMemo<LanguageContextValue>(
    () => ({
      language,
      setLanguage,
      t: messages[language],
    }),
    [language, setLanguage],
  );

  return (
    <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
  );
}
