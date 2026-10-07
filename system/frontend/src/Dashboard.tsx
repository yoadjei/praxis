/**
 * Dashboard.tsx - what the system currently knows and what it has refused to say.
 *
 * Three panels: the corpus (how much footage, how far through preprocessing and
 * annotation), the sessions (one row each, with the quality verdict the ingest probe
 * recorded), and the phase runs.
 *
 * The runs panel gives an abstention the same weight as a success, because that is what it
 * is: a phase that cannot run says so and names what is missing (D48). A dashboard that
 * drew abstentions as failures, or hid them, would make the build look broken on a corpus
 * that is simply not labelled yet.
 *
 * Model accuracy is deliberately not shown on its own anywhere. R4 requires calibration
 * wherever accuracy appears, so until the API sends both this panel reports the model as
 * absent rather than filling the space with a number.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  getSummary,
  getSessions,
  getRuns,
} from './dashboardApi';
import type {
  DashboardSummary,
  SessionPage,
  SessionRow,
  RunPage,
} from './types';
import './styles.css';

const PAGE_SIZE = 25;

/** Phase titles, so a run reads as work rather than as a number. Keyed as the API sends. */
const PHASE_TITLES: Record<string, string> = {
  '0': 'Foundations',
  '1': 'Ingest',
  '2': 'Quality',
  '3': 'Preprocess',
  '4': 'Features',
  '5': 'Model',
  '6': 'Evaluation',
  '7': 'Explanation',
  '8': 'Reliance',
  '9': 'Monitoring',
  '10': 'Instrumentation',
  '11': 'Export',
  noop: 'No-op',
};

function phaseLabel(phase: string): string {
  const title = PHASE_TITLES[phase];
  return title ? `${phase} · ${title}` : phase;
}

/** Seconds as hours and minutes, since the corpus is measured in hours of footage. */
function duration(totalSeconds: number): string {
  const seconds = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, '0')}m`;
  }
  return `${minutes}m ${String(seconds % 60).padStart(2, '0')}s`;
}

/**
 * The quality vocabulary, in the order a reader expects to see it. It is pass, warn and fail -
 * QualityVerdict in praxis/contracts/session.py, with a CHECK constraint on the column.
 */
const VERDICTS = ['pass', 'warn', 'fail'] as const;

function verdictClass(verdict: string): string {
  return VERDICTS.includes(verdict as (typeof VERDICTS)[number])
    ? `verdict verdict-${verdict}`
    : 'verdict verdict-unknown';
}

/** A labelled number with an optional second line of context. */
function Stat(props: {
  label: string;
  value: string;
  detail?: string;
}): JSX.Element {
  return (
    <div className="stat">
      <div className="stat-label">{props.label}</div>
      <div className="stat-value">{props.value}</div>
      {props.detail ? <div className="stat-detail">{props.detail}</div> : null}
    </div>
  );
}

function CorpusPanel(props: { summary: DashboardSummary }): JSX.Element {
  const { summary } = props;
  const domains = Object.entries(summary.corpus_by_domain);
  const total = domains.reduce(
    (sum, [, verdicts]) =>
      sum + Object.values(verdicts).reduce((a, b) => a + b, 0),
    0
  );

  // The known verdicts first, in their canonical order, then anything the server sent that this
  // build does not know about. Columns are derived rather than written out so that a verdict
  // added to the contract after this bundle was built still appears, with its real count,
  // instead of being quietly left out of a table that looks complete.
  const present = new Set(domains.flatMap(([, verdicts]) => Object.keys(verdicts)));
  const columns = [
    ...VERDICTS.filter((verdict) => present.has(verdict)),
    ...[...present].filter((verdict) => !VERDICTS.includes(verdict as never)).sort(),
  ];

  return (
    <section className="panel">
      <h3>Corpus</h3>
      {total === 0 ? (
        <p className="panel-empty">
          No sessions ingested yet. Run phase 1 to load a manifest.
        </p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Domain</th>
              {columns.map((verdict) => (
                <th key={verdict}>{verdict}</th>
              ))}
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {domains.map(([domain, verdicts]) => (
              <tr key={domain}>
                <td>{domain}</td>
                {columns.map((verdict) => (
                  <td key={verdict}>{verdicts[verdict] ?? 0}</td>
                ))}
                {/* Summed over every verdict in the row, not over the columns. The two are
                    the same once `columns` includes everything present, and summing the
                    data is what guarantees it: a row total that added up named columns
                    would drop a verdict this build had not heard of, and report a corpus
                    smaller than it is. */}
                <td>{Object.values(verdicts).reduce((a, b) => a + b, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function SessionsPanel(props: {
  page: SessionPage;
  onPage: (skip: number) => void;
  onSelect: (sessionId: string) => void;
}): JSX.Element {
  const { page } = props;
  const from = page.sessions.length === 0 ? 0 : page.skip + 1;
  const to = page.skip + page.sessions.length;
  const totalSeconds = page.sessions.reduce((sum, s) => sum + s.duration_s, 0);

  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Sessions</h3>
        <span className="panel-note">
          {page.total_count === 0
            ? 'nothing ingested'
            : `${from}–${to} of ${page.total_count} · ${duration(totalSeconds)} on this page`}
        </span>
      </div>

      {page.sessions.length === 0 ? (
        <p className="panel-empty">No sessions to show.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Session</th>
              <th>Domain</th>
              <th>Recorded</th>
              <th>Length</th>
              <th>Quality</th>
              <th>Preprocessed</th>
              <th>Annotated</th>
            </tr>
          </thead>
          <tbody>
            {page.sessions.map((session: SessionRow) => (
              <tr
                key={session.session_id}
                className="row-clickable"
                onClick={() => props.onSelect(session.session_id)}
              >
                <td className="mono">{session.session_id}</td>
                <td>{session.domain}</td>
                <td>{session.recorded_on ?? '—'}</td>
                <td>{duration(session.duration_s)}</td>
                <td>
                  <span className={verdictClass(session.quality_verdict)}>
                    {session.quality_verdict}
                  </span>
                </td>
                <td>{session.preprocessed ? 'yes' : 'no'}</td>
                <td>
                  {session.excluded_at === null ? (
                    session.annotated ? 'yes' : 'no'
                  ) : (
                    // Not 'no'. An excluded session is a decision, and reporting it as
                    // un-annotated puts it back in the queue the reviewer works through. The
                    // ground is on the title so it can be read without leaving the list. D96.
                    <span className="excluded" title={session.exclusion_reason ?? undefined}>
                      excluded
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div className="pager">
        <button
          type="button"
          disabled={page.skip === 0}
          onClick={() => props.onPage(Math.max(0, page.skip - page.limit))}
        >
          Previous
        </button>
        <button
          type="button"
          disabled={to >= page.total_count}
          onClick={() => props.onPage(page.skip + page.limit)}
        >
          Next
        </button>
      </div>
    </section>
  );
}

function RunsPanel(props: { page: RunPage }): JSX.Element {
  const runs = props.page.runs;
  const ran = runs.filter((r) => r.verdict === 'ran').length;
  // The directory count is shown whenever it exceeds what came back, so a reader is never
  // left thinking the newest fifty are all there have ever been.
  const truncated = props.page.run_directories > runs.length;

  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Phase runs</h3>
        <span className="panel-note">
          {runs.length === 0
            ? 'no runs recorded'
            : `${ran} ran, ${runs.length - ran} abstained` +
              (truncated
                ? ` · newest ${runs.length} of ${props.page.run_directories} directories`
                : '')}
        </span>
      </div>

      {runs.length === 0 ? (
        <p className="panel-empty">
          No runs yet. Run one with scripts/run_build.py.
        </p>
      ) : (
        <ul className="run-list">
          {runs.map((run) => (
            <li key={run.run_id} className={`run run-${run.verdict}`}>
              <div className="run-head">
                <span className="run-phase">{phaseLabel(run.phase)}</span>
                <span className={`verdict verdict-${run.verdict}`}>
                  {run.verdict}
                </span>
                <span className="run-when">{run.finished_at ?? ''}</span>
              </div>
              {run.abstention_reason ? (
                <div className="run-reason">{run.abstention_reason}</div>
              ) : null}
              {run.abstention_missing && run.abstention_missing.length > 0 ? (
                <div className="run-missing">
                  missing: {run.abstention_missing.join(', ')}
                </div>
              ) : null}
              <div className="run-id mono">{run.run_id}</div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function Dashboard(props: {
  onSelectSession: (sessionId: string) => void;
}): JSX.Element {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [page, setPage] = useState<SessionPage | null>(null);
  const [runs, setRuns] = useState<RunPage | null>(null);
  const [skip, setSkip] = useState(0);
  const [error, setError] = useState('');

  const load = useCallback(async (offset: number) => {
    setError('');
    try {
      const [nextSummary, nextPage, nextRuns] = await Promise.all([
        getSummary(),
        getSessions({ skip: offset, limit: PAGE_SIZE }),
        getRuns(),
      ]);
      setSummary(nextSummary);
      setPage(nextPage);
      setRuns(nextRuns);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  }, []);

  useEffect(() => {
    void load(skip);
  }, [load, skip]);

  if (error) {
    return (
      <div className="error-container">
        <p>{error}</p>
        <button type="button" onClick={() => void load(skip)}>
          Retry
        </button>
      </div>
    );
  }

  if (!summary || !page || !runs) {
    return <div className="loading-container">Loading…</div>;
  }

  const { preprocessed_count, total_sessions } = summary.preprocessing;

  return (
    <div className="dashboard">
      <div className="stat-row">
        <Stat label="Sessions" value={String(total_sessions)} />
        <Stat
          label="Preprocessed"
          value={`${preprocessed_count} / ${total_sessions}`}
          detail="pose extracted, faces blurred, original deleted"
        />
        <Stat
          label="Annotations"
          value={String(summary.annotation.annotations)}
          detail={`${summary.annotation.assignments} assignments`}
        />
        <Stat
          label="Model"
          value={summary.model.status}
          detail="accuracy appears only alongside calibration (R4)"
        />
        <Stat
          label="Last audit entry"
          value={summary.last_audit_at ? 'recorded' : 'none'}
          detail={summary.last_audit_at ?? 'the chain is empty'}
        />
      </div>

      <CorpusPanel summary={summary} />
      <SessionsPanel
        page={page}
        onPage={setSkip}
        onSelect={props.onSelectSession}
      />
      <RunsPanel page={runs} />
    </div>
  );
}
