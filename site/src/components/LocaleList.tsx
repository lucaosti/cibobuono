import { useState } from "react";
import type { LocaleWithVisits } from "../types";
import { GITHUB_REPO } from "../types";
import { useT } from "../i18n/useLanguage";
import type { Messages } from "../i18n/messages";

function sentimentLabel(t: Messages, sentiment: string): string {
  switch (sentiment) {
    case "positive":
      return t.sentimentPositive;
    case "negative":
      return t.sentimentNegative;
    default:
      return t.sentimentNeutral;
  }
}

function buildIssueUrl(
  t: Messages,
  locale: LocaleWithVisits,
  action: "hide" | "edit",
  editData?: { name?: string; city?: string; notes?: string },
): string {
  const title =
    action === "hide"
      ? t.issueRemoveTitle(locale.name)
      : t.issueEditTitle(locale.name);

  let body: string;
  if (action === "hide") {
    body = [
      t.issueRemoveHeader,
      "",
      `${t.issueFieldLocaleId} \`${locale.locale_id}\``,
      `${t.issueFieldCurrentName} ${locale.name}`,
      `${t.issueFieldCity} ${locale.city}`,
      "",
      t.issueCorrectionEntryHeader,
      "```json",
      JSON.stringify(
        {
          locale_id: locale.locale_id,
          type: "hide",
          reason: t.issueReasonFalsePositive,
        },
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
      t.issueEditHeader,
      "",
      `${t.issueFieldLocaleId} \`${locale.locale_id}\``,
      `${t.issueFieldCurrentName} ${locale.name}`,
      `${t.issueFieldCity} ${locale.city}`,
      editData?.notes ? `${t.issueFieldNotes} ${editData.notes}` : "",
      "",
      t.issueCorrectionEntryHeader,
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
  const t = useT();
  const [name, setName] = useState(locale.name);
  const [city, setCity] = useState(locale.city);
  const [notes, setNotes] = useState("");

  return (
    <div className="edit-form" onClick={(e) => e.stopPropagation()}>
      <div className="edit-form-header">
        <strong>{t.suggestCorrection}</strong>
        <button className="edit-close" onClick={onClose}>
          &times;
        </button>
      </div>

      <label>
        {t.fieldName}
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <label>
        {t.fieldCity}
        <input value={city} onChange={(e) => setCity(e.target.value)} />
      </label>
      <label>
        {t.fieldNotes}
        <input
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder={t.notesPlaceholder}
        />
      </label>

      <div className="edit-actions">
        <a
          href={buildIssueUrl(t, locale, "edit", { name, city, notes })}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-primary"
        >
          {t.submitEdit}
        </a>
        <a
          href={buildIssueUrl(t, locale, "hide")}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-danger"
        >
          {t.reportFalsePositive}
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
  const t = useT();
  const [editing, setEditing] = useState<string | null>(null);

  if (locales.length === 0) {
    return (
      <div className="locale-list-empty">
        <p>{t.noVenuesFound}</p>
        <p className="hint">{t.noVenuesHint}</p>
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
                  <span className="rating-badge">
                    {t.ratingOutOf10(loc.avgRating)}
                  </span>
                )}
                <span className={`sentiment-badge ${loc.dominantSentiment}`}>
                  {sentimentLabel(t, loc.dominantSentiment)}
                </span>
              </div>
            </div>

            {loc.city && <p className="locale-city">{loc.city}</p>}

            <div className="locale-meta">
              <span className="visits-count">
                {t.videosCount(loc.visits.length)}
              </span>
              {loc.category.length > 0 && (
                <span className="categories">{loc.category.join(", ")}</span>
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
                      {t.googleMaps}
                    </a>
                  )}
                  <button
                    className="btn btn-edit"
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditing(
                        editing === loc.locale_id ? null : loc.locale_id,
                      );
                    }}
                  >
                    {editing === loc.locale_id ? t.cancel : t.edit}
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
                          {sentimentLabel(t, v.sentiment).charAt(0)}
                        </span>
                        <span className="visit-info">
                          {v.timestamp_start} &mdash; {v.date}
                          {v.rating && (
                            <span className="visit-rating">
                              {" "}
                              {t.ratingOutOf10(v.rating)}
                            </span>
                          )}
                        </span>
                        <span className="visit-arrow">&#9654;</span>
                      </a>
                      {v.notes && <p className="visit-notes">{v.notes}</p>}
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
