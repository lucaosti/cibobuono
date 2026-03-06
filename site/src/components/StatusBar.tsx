interface Props {
  loading: boolean;
  error: string | null;
}

export default function StatusBar({ loading, error }: Props) {
  if (loading) {
    return (
      <div className="status-bar loading">
        <div className="spinner" />
        Loading data...
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
