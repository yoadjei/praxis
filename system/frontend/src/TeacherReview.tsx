/**
 * TeacherReview.tsx - which of these people is the teacher.
 *
 * The one screen the corpus cannot get past preprocessing without. `plan_annotation` refuses to
 * cut clips for a session whose teacher track nobody confirmed, because a clip cut on the
 * heuristic's guess could show somebody who is not the teacher and the label would be attached
 * to them (R1).
 *
 * **The heuristic's pick is shown as a suggestion and never as a default.** No track is
 * selected when the screen opens, including the proposed one. A pre-selected answer that a
 * reviewer confirms with one click is how a guess becomes a human judgement without anybody
 * having judged anything, and the whole point of this step is that a person looked.
 *
 * The score and its three signals are shown beside each candidate rather than only the ranking,
 * so a reviewer who disagrees can see what the heuristic was reading. They may pick any
 * candidate, and `praxis/preprocess/store.py` records a disagreement as such - which is what
 * makes the heuristic's field accuracy measurable rather than assumed.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getCandidates, confirmTrack, fetchThumbnail } from './tracksApi';
import type { Reviewer } from './tracksApi';
import type { TrackCandidate, TrackCandidates, ThumbnailRef } from './types';
import './styles.css';

const STRIP_LENGTH = 5;

function percent(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(0)}%`;
}

function score(value: number | null): string {
  return value === null ? 'no score' : value.toFixed(3);
}

/**
 * One still. Fetched through the API with the actor header, which an `img src` cannot carry, so
 * the blob URL is created here and revoked on unmount - one per picture, released when the
 * picture goes away rather than when the tab closes.
 */
function Still(props: { thumbnail: ThumbnailRef; reviewer: Reviewer }): JSX.Element {
  const [src, setSrc] = useState('');
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let revoked = false;
    let created = '';

    void fetchThumbnail(props.thumbnail.url, props.reviewer)
      .then((url) => {
        if (revoked) {
          URL.revokeObjectURL(url);
          return;
        }
        created = url;
        setSrc(url);
      })
      .catch(() => setFailed(true));

    return () => {
      revoked = true;
      if (created) URL.revokeObjectURL(created);
    };
  }, [props.thumbnail.url, props.reviewer]);

  if (failed) {
    return (
      <div className="still still-failed" title={props.thumbnail.url}>
        frame unavailable
      </div>
    );
  }

  return (
    <figure className="still">
      {src ? (
        <img src={src} alt={`track at ${props.thumbnail.at_seconds.toFixed(1)}s`} />
      ) : (
        <div className="still-loading" />
      )}
      <figcaption>{props.thumbnail.at_seconds.toFixed(1)}s</figcaption>
    </figure>
  );
}

function CandidateCard(props: {
  candidate: TrackCandidate;
  reviewer: Reviewer;
  selected: boolean;
  onSelect: () => void;
}): JSX.Element {
  const { candidate } = props;
  const classes = ['candidate-card'];
  if (props.selected) classes.push('candidate-selected');
  if (candidate.is_confirmed) classes.push('candidate-confirmed');

  return (
    <div className={classes.join(' ')}>
      <div className="candidate-header">
        <label>
          <input
            type="radio"
            name="teacher-track"
            checked={props.selected}
            onChange={props.onSelect}
          />
          <strong>Track {candidate.track_id}</strong>
        </label>
        <span className="candidate-score">{score(candidate.score)}</span>
      </div>

      {candidate.is_proposed ? (
        <p className="candidate-badge">suggested by the heuristic</p>
      ) : null}
      {candidate.is_confirmed ? (
        <p className="candidate-badge candidate-badge-confirmed">
          currently confirmed
        </p>
      ) : null}

      <dl className="candidate-signals">
        <div>
          <dt>present</dt>
          <dd>{percent(candidate.presence_fraction)}</dd>
        </div>
        <div>
          <dt>size</dt>
          <dd>{percent(candidate.median_area_fraction)} of frame</dd>
        </div>
        <div>
          <dt>front zone</dt>
          <dd>{percent(candidate.front_zone_fraction)}</dd>
        </div>
      </dl>

      {candidate.thumbnails.length ? (
        <div className="candidate-strip">
          {candidate.thumbnails.map((thumbnail) => (
            <Still
              key={thumbnail.index}
              thumbnail={thumbnail}
              reviewer={props.reviewer}
            />
          ))}
        </div>
      ) : (
        <p className="not-recorded">
          no still could be located for this track, so there is nothing to look at
        </p>
      )}
    </div>
  );
}

export function TeacherReview(props: {
  sessionId: string;
  reviewer: Reviewer;
  onBack: () => void;
}): JSX.Element {
  const [data, setData] = useState<TrackCandidates | null>(null);
  const [error, setError] = useState('');
  const [chosen, setChosen] = useState<number | null>(null);
  const [custom, setCustom] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState('');
  const reviewer = useRef(props.reviewer);
  reviewer.current = props.reviewer;

  const load = useCallback(async () => {
    setError('');
    setSaved('');
    setData(null);
    // Deliberately not pre-selected from the response. See the file header: a pre-filled
    // answer turns a confirmation into a click.
    setChosen(null);
    setCustom('');
    try {
      setData(await getCandidates(props.sessionId, reviewer.current, STRIP_LENGTH));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  }, [props.sessionId]);

  useEffect(() => {
    void load();
  }, [load]);

  const submit = async () => {
    const track = chosen !== null ? chosen : Number.parseInt(custom, 10);
    if (!Number.isInteger(track) || track < 0) return;

    setSaving(true);
    setError('');
    try {
      const result = await confirmTrack(props.sessionId, track, reviewer.current);
      setSaved(`track ${result.track_id} confirmed by ${result.confirmed_by}`);
      await load();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setSaving(false);
    }
  };

  if (error && !data) {
    return (
      <div className="error-container">
        <p>{error}</p>
        <button type="button" onClick={props.onBack}>
          Back to dashboard
        </button>
      </div>
    );
  }

  if (!data) {
    return <div className="loading-container">Loading candidates…</div>;
  }

  const pending = chosen !== null || /^\d+$/.test(custom.trim());

  return (
    <div className="teacher-review">
      <button type="button" className="link-button" onClick={props.onBack}>
        ← Dashboard
      </button>
      <h2>Who is the teacher?</h2>
      <p className="session-id mono">{data.session_id}</p>

      {/* The heuristic's reasoning, verbatim. It carries the scaled-floor caveat when the
          camera setup has no marked zones, and a reviewer judging a score needs to know the
          score was measured against two signals instead of three. */}
      <section className="proposal-panel">
        <h3>
          {data.proposal.track_id === null
            ? 'The heuristic proposed nothing'
            : `The heuristic suggests track ${data.proposal.track_id}`}
        </h3>
        <p>{data.proposal.reason}</p>
        {data.confirmation.confirmed_by ? (
          <p className="confirmed-note">
            Confirmed by {data.confirmation.confirmed_by} at{' '}
            {data.confirmation.confirmed_at}. Confirming again replaces it; the audit trail
            keeps both.
          </p>
        ) : (
          <p className="unconfirmed-note">
            Not yet confirmed. No clip can be cut for annotation until somebody identifies the
            teacher.
          </p>
        )}
      </section>

      {!data.ranking.available ? (
        <section className="abstention-panel">
          <h3>The ranking was never recorded</h3>
          <p>{data.ranking.detail}</p>
        </section>
      ) : null}

      {data.thumbnails_unavailable ? (
        <p className="abstention-panel">
          The stills could not be located: {data.thumbnails_unavailable}. The scores below are
          still the stored ones; there is nothing to look at beside them.
        </p>
      ) : null}

      {data.ranking.zones_available === false ? (
        <p className="field-note">
          This session has no marked seating zones, so the front-zone signal below is unmeasured
          rather than zero and the scores are not comparable with a session that has them.
        </p>
      ) : null}

      <div className="candidate-grid">
        {data.candidates.map((candidate) => (
          <CandidateCard
            key={candidate.track_id}
            candidate={candidate}
            reviewer={props.reviewer}
            selected={chosen === candidate.track_id}
            onSelect={() => {
              setChosen(candidate.track_id);
              setCustom('');
            }}
          />
        ))}
      </div>

      {data.ranking.truncated > 0 ? (
        <p className="field-note">
          {data.ranking.truncated} further candidate
          {data.ranking.truncated === 1 ? '' : 's'} scored below these and were not stored.
        </p>
      ) : null}

      {/* A track the ranking does not contain. Reachable on purpose: the heuristic may have
          ranked nobody, or the person it ranked may not be the teacher, and a screen that only
          offered its own candidates would make the reviewer's judgement a subset of the
          heuristic's. */}
      <section className="other-track">
        <label>
          None of these — the teacher is track
          <input
            type="number"
            min={0}
            value={custom}
            onChange={(event) => {
              setCustom(event.target.value);
              setChosen(null);
            }}
          />
        </label>
      </section>

      <div className="review-actions">
        <button type="button" onClick={() => void submit()} disabled={!pending || saving}>
          {saving ? 'Confirming…' : 'Confirm this track'}
        </button>
        {saved ? <span className="saved-note">{saved}</span> : null}
        {error ? <span className="error-note">{error}</span> : null}
      </div>
    </div>
  );
}
