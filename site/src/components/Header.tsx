import { useLanguage } from "../i18n/useLanguage";
import { LANGUAGES, type Language } from "../i18n/messages";

interface Props {
  total: number;
  sentimentFilter: string;
  onSentimentChange: (v: string) => void;
  cityFilter: string;
  onCityChange: (v: string) => void;
  cities: string[];
  searchQuery: string;
  onSearchChange: (v: string) => void;
  nearMeKm: number | null;
  onNearMeChange: (km: number | null) => void;
  geoError: string | null;
}

export default function Header({
  total,
  sentimentFilter,
  onSentimentChange,
  cityFilter,
  onCityChange,
  cities,
  searchQuery,
  onSearchChange,
  nearMeKm,
  onNearMeChange,
  geoError,
}: Props) {
  const { language, setLanguage, t } = useLanguage();

  return (
    <header>
      <div className="header-left">
        <h1>{t.appTitle}</h1>
        <span className="header-count">{t.venuesCount(total)}</span>
      </div>

      <div className="header-controls">
        <input
          type="search"
          placeholder={t.searchPlaceholder}
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="search-input"
          aria-label={t.searchPlaceholder}
        />

        <select
          value={cityFilter}
          onChange={(e) => onCityChange(e.target.value)}
          className="filter-select"
          title={t.filterByCityTitle}
          aria-label={t.filterByCityTitle}
        >
          <option value="">{t.allCities}</option>
          {cities.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>

        <select
          value={sentimentFilter}
          onChange={(e) => onSentimentChange(e.target.value)}
          className="filter-select"
          title={t.filterBySentimentTitle}
          aria-label={t.filterBySentimentTitle}
        >
          <option value="">{t.allSentiments}</option>
          <option value="positive">{t.sentimentPositive}</option>
          <option value="neutral">{t.sentimentNeutral}</option>
          <option value="negative">{t.sentimentNegative}</option>
        </select>

        <div className="near-me-control">
          <select
            value={nearMeKm ?? ""}
            onChange={(e) => {
              const v = e.target.value;
              onNearMeChange(v ? Number(v) : null);
            }}
            className="filter-select"
            title={t.nearMeTitle}
            aria-label={t.nearMeTitle}
          >
            <option value="">{t.nearMe}</option>
            <option value="1">{t.km(1)}</option>
            <option value="2">{t.km(2)}</option>
            <option value="5">{t.km(5)}</option>
            <option value="10">{t.km(10)}</option>
            <option value="25">{t.km(25)}</option>
            <option value="50">{t.km(50)}</option>
          </select>
          {geoError && (
            <span
              className="geo-error"
              title={geoError}
              role="alert"
              aria-label={geoError}
            >
              {t.geoErrorMarker}
            </span>
          )}
        </div>

        <div
          className="language-toggle"
          role="group"
          aria-label={t.languageToggleLabel}
        >
          {LANGUAGES.map((opt) => (
            <button
              key={opt.code}
              type="button"
              className={`language-button ${
                language === opt.code ? "active" : ""
              }`}
              aria-pressed={language === opt.code}
              onClick={() => setLanguage(opt.code as Language)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
