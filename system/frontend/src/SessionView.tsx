/**
 * SessionView.tsx - one session, in full.
 *
 * What is shown is what the API sends, and the API sends no teacher id, no teacher code and
 * no media filename (R1, D76). A blank subject or grade level is drawn as "not recorded"
 * rather than left empty: the gap is information. Phase 6's attribution cannot use a
 * covariate nobody wrote down, and a blank cell in a table looks like a rendering fault
 * instead of a missing field in the field log.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { getSessionDetail } from './dashboardApi';
import type { SessionDetail } from './types';
import './styles.css';

function orMissing(value: string | null): JSX.Element {
  if (value) return <>{value}</>;
  return <span className="not-recorded">not recorded</span>;
}

function seconds(value: number): string {
  const whole = Math.round(value);
  const minutes = Math.floor(whole / 60);
  return `${minutes}m ${String(whole % 60).padStart(2, '0')}s (${value.toFixed(2)}s)`;
}

export function SessionView(props: {
  sessionId: string;
  onBack: () => void;
  onReviewTeacher: () => void;
}): JSX.Element {
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setError('');
    setDetail(null);
    try {
      setDetail(await getSessionDetail(props.sessionId));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  }, [props.sessionId]);

  useEffect(() => {
    void load();
  }, [load]);

  if (error) {
    return (
      <div className="error-container">
        <p>{error}</p>
        <button type="button" onClick={props.onBack}>
          Back to dashboard
        </button>
      </div>
    );
  }

  if (!detail) {
    return <div className="loading-container">Loading…</div>;
  }

  const rows: Array<[string, JSX.Element]> = [
    ['Session', <span className="mono">{detail.session_id}</span>],
    ['Domain', <>{detail.domain}</>],
    ['Recorded on', orMissing(detail.recorded_on)],
    ['Length', <>{seconds(detail.duration_s)}</>],
    [
      'Quality verdict',
      <span className={`verdict verdict-${detail.quality_verdict}`}>
        {detail.quality_verdict}
      </span>,
    ],
    ['Subject', orMissing(detail.subject)],
    ['Grade level', orMissing(detail.grade_level)],
    [
      'Preprocessed',
      <>
        {detail.preprocessed ? 'yes' : 'no'}
        {detail.preprocessed ? (
          <span className="field-note">
            {' '}
            faces blurred and the original deleted (D18)
          </span>
        ) : null}
      </>,
    ],
    ['Annotations', <>{detail.annotation_count}</>],
    ['Ingested at', orMissing(detail.created_at)],
  ];

  return (
    <div className="session-view">
      <button type="button" className="link-button" onClick={props.onBack}>
        ← Dashboard
      </button>
      <h2>Session detail</h2>
      <table className="detail-table">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <th>{label}</th>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Offered for a preprocessed session only, because there are no tracks to choose
          between before phase 3 has run. Whether the teacher is already confirmed is not read
          from this payload: the dashboard detail does not carry it, and inferring confirmation
          from `preprocessed` would offer the step as done when nobody had done it. */}
      {detail.preprocessed ? (
        <div className="review-actions">
          <button type="button" onClick={props.onReviewTeacher}>
            Confirm the teacher track
          </button>
          <span className="field-note">
            No clip can be cut for annotation until somebody identifies the teacher.
          </span>
        </div>
      ) : null}
    </div>
  );
}
