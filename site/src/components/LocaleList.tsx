import { useState } from "react";
import type { LocaleWithVisits } from "../types";
import { GITHUB_REPO } from "../types";

const SENTIMENT_LABEL: Record<string, string> = {
  positive: "Positive",
  neutral: "Neutral",
  negative: "Negative",
};

function buildIssueUrl(
  locale: LocaleWithVisits,
  action: "hide" | "edit",
  editData?: { name?: string; city?: string; notes?: string },
): string {
  const title =
    action === "hide"
      ? `[Correction] Remove: ${locale.name}`
      : `[Correction] Edit: ${locale.name}`;

  let body: string;
  if (action === "hide") {
    body = [
      "## Correction: Remove false positive",
      "",
      `**Locale ID:** \`${locale.locale_id}\``,
      `**Current name:** ${locale.name}`,
      `**City:** ${locale.city}`,
      "",
      "### corrections.json entry",
      "```json",
      JSON.stringify(
        { locale_id: locale.locale_id, type: "hide", reason: "False positive — not actually visited" },
        null,
        2,
      ),
      "```",
    ].join("\n");
  } else {
    const overrides: Record<string, string> = {};
    if (editData?.name) overrides.name = editData.name;
    if (editData?.city) overrides.city = editData.city;
    body = [
      "## Correction: Edit locale",
      "",
      `**Locale ID:** \`${locale.locale_id}\``,
      `**Current name:** ${locale.name}`,
      `**City:** ${locale.city}`,
      editData?.notes ? `**Notes:** ${editData.notes}` : "",
      "",
      "### corrections.json entry",
      "```json",
      JSON.stringify(
        { locale_id: locale.locale_id, type: "edit", overrides },
        null,
        2,
      ),
      "```",
    ].join("\n");
  }

  const params = new URLSearchParams({
    title,
    body,
    labels: "correction",
  });
  return `https://github.com/${GITHUB_REPO}/issues/new?${params}`;
}

interface EditFormProps {
  locale: LocaleWithVisits;
  onClose: () => void;
}

function EditForm({ locale, onClose }: EditFormProps) {
  const [name, setName] = useState(locale.name);
  const [city, setCity] = useState(locale.city);
  const [notes, setNotes] = useState("");

  return (
    <div className="edit-form" onClick={(e) => e.stopPropagation()}>
      <div className="edit-form-header">
        <strong>Suggest correction</strong>
        <button className="edit-close" onClick={onClose}>&times;</button>
      </div>

      <label>
        Name
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <label>
        City
        <input value={city} onChange={(e) => setCity(e.target.value)} />
      </label>
      <label>
        Notes
        <input
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="What's wrong?"
        />
      </label>

      <div className="edit-actions">
        <a
          href={buildIssueUrl(locale, "edit", { name, city, notes })}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-primary"
        >
          Submit edit
        </a>
        <a
          href={buildIssueUrl(locale, "hide")}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-danger"
        >
          Report false positive
        </a>
      </div>
    </div>
  );
}

interface Props {
  locales: LocaleWithVisits[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export default function LocaleList({ locales, selected, onSelect }: Props) {
  const [editing, setEditing] = useState<string | null>(null);

  if (locales.length === 0) {
    return (
      <div className="locale-list-empty">
        <p>No venues found.</p>
        <p className="hint">The pipeline hasn't extracted any locales yet.</p>
      </div>
    );
  }

  return (
    <ul className="locale-list">
      {locales.map((loc) => {
        const isSelected = selected === loc.locale_id;
        return (
          <li
            key={loc.locale_id}
            className={`locale-card ${isSelected ? "active" : ""}`}
            onClick={() => onSelect(loc.locale_id)}
          >
            <div className="locale-header">
              <h3>{loc.name}</h3>
              <div className="locale-badges">
                {loc.avgRating != null && (
                  <span className="rating-badge">{loc.avgRating}/10</span>
                )}
                <span className={`sentiment-badge ${loc.dominantSentiment}`}>
                  {SENTIMENT_LABEL[loc.dominantSentiment]}
                </span>
              </div>
            </div>

            {loc.city && <p className="locale-city">{loc.city}</p>}

            <div className="locale-meta">
              <span className="visits-count">
                {loc.visits.length} video{loc.visits.length !== 1 ? "s" : ""}
              </span>
              {loc.category.length > 0 && (
                <span className="categories">
                  {loc.category.join(", ")}
                </span>
              )}
            </div>

            {isSelected && (
              <div className="locale-details">
                <div className="locale-actions-row">
                  {loc.google_maps_url && (
                    <a
                      href={loc.google_maps_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn btn-maps"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Google Maps
                    </a>
                  )}
                  <button
                    className="btn btn-edit"
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditing(editing === loc.locale_id ? null : loc.locale_id);
                    }}
                  >
                    {editing === loc.locale_id ? "Cancel" : "Edit"}
                  </button>
                </div>

                {editing === loc.locale_id && (
                  <EditForm locale={loc} onClose={() => setEditing(null)} />
                )}

                <div className="locale-visits">
                  {loc.visits.map((v) => (
                    <div key={v.visit_id} className="visit-entry">
                      <a
                        href={v.youtube_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="visit-link"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <span className={`visit-sentiment ${v.sentiment}`}>
                          {SENTIMENT_LABEL[v.sentiment]?.[0]}
                        </span>
                        <span className="visit-info">
                          {v.timestamp_start} &mdash; {v.date}
                          {v.rating && (
                            <span className="visit-rating"> {v.rating}/10</span>
                          )}
                        </span>
                        <span className="visit-arrow">&#9654;</span>
                      </a>
                      {v.notes && (
                        <p className="visit-notes">{v.notes}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}
