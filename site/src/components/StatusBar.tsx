import { useT } from "../i18n/useLanguage";

interface Props {
  loading: boolean;
  error: string | null;
}

export default function StatusBar({ loading, error }: Props) {
  const t = useT();

  if (loading) {
    return (
      <div className="status-bar loading">
        <div className="spinner" />
        {t.loadingData}
      </div>
    );
  }
  if (error) {
    return (
      <div className="status-bar error">
        {error}
      </div>
    );
  }
  return null;
}
