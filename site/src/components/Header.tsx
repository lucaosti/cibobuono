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
  return (
    <header>
      <div className="header-left">
        <h1>CiboBuono</h1>
        <span className="header-count">{total} venues</span>
      </div>

      <div className="header-controls">
        <input
          type="search"
          placeholder="Search..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="search-input"
        />

        <select
          value={cityFilter}
          onChange={(e) => onCityChange(e.target.value)}
          className="filter-select"
          title="Filter by city"
        >
          <option value="">All cities</option>
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
          title="Filter by sentiment"
        >
          <option value="">All</option>
          <option value="positive">Positive</option>
          <option value="neutral">Neutral</option>
          <option value="negative">Negative</option>
        </select>

        <div className="near-me-control">
          <select
            value={nearMeKm ?? ""}
            onChange={(e) => {
              const v = e.target.value;
              onNearMeChange(v ? Number(v) : null);
            }}
            className="filter-select"
            title="Near me"
          >
            <option value="">Near me</option>
            <option value="1">1 km</option>
            <option value="2">2 km</option>
            <option value="5">5 km</option>
            <option value="10">10 km</option>
            <option value="25">25 km</option>
            <option value="50">50 km</option>
          </select>
          {geoError && <span className="geo-error" title={geoError}>!</span>}
        </div>
      </div>
    </header>
  );
}
