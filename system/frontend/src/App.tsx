/**
 * App.tsx - the shell: which view is showing, and who is looking at it.
 *
 * Routing is the URL fragment and a switch, not a router library. Three views and no nested
 * routes do not justify a dependency, and keeping react and react-dom as the only two means
 * the Docker build has nothing to resolve beyond them.
 *
 * The rater id is asked for in the page rather than through window.prompt. The previous entry
 * point called prompt() at module scope, so nothing rendered until it was answered, a blocked
 * dialog looked identical to a broken build, and "cancel" annotated the corpus as
 * 'unknown-rater'. An annotation is attributed evidence; it cannot be filed under a rater who
 * declined to say who they were.
 */

import React, { useState, useEffect } from 'react';
import { AnnotationTool } from './AnnotationTool';
import { CalibrationReview } from './CalibrationReview';
import { Dashboard } from './Dashboard';
import { SessionView } from './SessionView';
import { TeacherReview } from './TeacherReview';
import { getHealth } from './dashboardApi';
import type { Reviewer } from './tracksApi';
import './styles.css';

const RATER_KEY = 'praxis.raterId';
const REVIEWER_KEY = 'praxis.reviewer';
const DEFAULT_ROUND = 'calibration-1';
const REVIEWER_ROLES = ['supervisor', 'researcher', 'admin'];

type View =
  | { name: 'dashboard' }
  | { name: 'session'; sessionId: string }
  | { name: 'teacher'; sessionId: string }
  | { name: 'annotate' }
  | { name: 'calibration'; round: string };

/** Parse the fragment into a view. An unrecognised fragment falls back to the dashboard. */
function viewFromHash(hash: string): View {
  const path = hash.replace(/^#\/?/, '');
  const [head, tail] = [path.split('/')[0], path.split('/').slice(1).join('/')];
  if (head === 'annotate') return { name: 'annotate' };
  if (head === 'calibration') {
    return { name: 'calibration', round: tail || DEFAULT_ROUND };
  }
  if (head === 'session' && tail) return { name: 'session', sessionId: tail };
  if (head === 'teacher' && tail) return { name: 'teacher', sessionId: tail };
  return { name: 'dashboard' };
}

/** A ULID, which is what the audit trail can store an actor as. 26 characters of Crockford
 *  base32, which excludes I, L, O and U so they cannot be confused with 1 and 0. The server
 *  refuses anything else with a 401 rather than writing a row that would break the chain, and
 *  saying so here costs the reviewer one round trip less. */
function isUlid(value: string): boolean {
  return /^[0-9A-HJKMNP-TV-Z]{26}$/.test(value.toUpperCase());
}

function storedReviewer(): Reviewer | null {
  try {
    const raw = window.localStorage.getItem(REVIEWER_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (
      parsed &&
      typeof parsed === 'object' &&
      typeof (parsed as Reviewer).role === 'string' &&
      typeof (parsed as Reviewer).userId === 'string' &&
      REVIEWER_ROLES.includes((parsed as Reviewer).role) &&
      isUlid((parsed as Reviewer).userId)
    ) {
      return parsed as Reviewer;
    }
  } catch {
    // A corrupted or hand-edited value is treated as absent. Asking again costs a moment;
    // sending a malformed actor would put a row in the trail attributed to nobody.
  }
  return null;
}

function navigate(fragment: string): void {
  window.location.hash = fragment;
}

/**
 * Ask for the rater id. Shown instead of the annotation tool, never over it, so there is no
 * state to lose if it is dismissed - it cannot be dismissed.
 */
function RaterGate(props: { onSet: (raterId: string) => void }): JSX.Element {
  const [draft, setDraft] = useState('');
  const trimmed = draft.trim();

  return (
    <form
      className="rater-gate"
      onSubmit={(event) => {
        event.preventDefault();
        if (trimmed) props.onSet(trimmed);
      }}
    >
      <h2>Who is annotating?</h2>
      <p>
        Every annotation is filed against a rater id, and inter-rater agreement is computed
        across those ids. An annotation with no rater attached cannot be used.
      </p>
      <input
        type="text"
        value={draft}
        placeholder="rater id"
        autoFocus
        onChange={(event) => setDraft(event.target.value)}
      />
      <button type="submit" disabled={!trimmed}>
        Start
      </button>
    </form>
  );
}

/**
 * Ask who is reviewing. A role and a ULID, because a confirmation is attributed evidence:
 * `teacher_tracks.confirmed_by` is the column R1 turns on, and the audit trail stores the actor
 * in a fixed-width ULID column. Shown instead of the review screen, never over it.
 */
function ReviewerGate(props: { onSet: (reviewer: Reviewer) => void }): JSX.Element {
  const [role, setRole] = useState(REVIEWER_ROLES[0]);
  const [userId, setUserId] = useState('');
  const trimmed = userId.trim().toUpperCase();
  const valid = isUlid(trimmed);

  return (
    <form
      className="rater-gate reviewer-gate"
      onSubmit={(event) => {
        event.preventDefault();
        if (valid) props.onSet({ role, userId: trimmed });
      }}
    >
      <h2>Who is reviewing?</h2>
      <p>
        Confirming a teacher track is a judgement filed against a person, and every downstream
        phase reads who made it. The user id is a ULID: the audit trail stores it in a
        fixed-width column, so a shorter value would be padded and the chain would no longer
        verify.
      </p>
      <select value={role} onChange={(event) => setRole(event.target.value)}>
        {REVIEWER_ROLES.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
      <input
        type="text"
        value={userId}
        placeholder="user id (26-character ULID)"
        autoFocus
        onChange={(event) => setUserId(event.target.value)}
      />
      {trimmed && !valid ? (
        <p className="error-note">
          {trimmed.length} characters; a ULID is 26 and uses no I, L, O or U.
        </p>
      ) : null}
      <button type="submit" disabled={!valid}>
        Start reviewing
      </button>
    </form>
  );
}

export function App(): JSX.Element {
  const [view, setView] = useState<View>(() =>
    viewFromHash(window.location.hash)
  );
  const [raterId, setRaterId] = useState<string>(
    () => window.localStorage.getItem(RATER_KEY) ?? ''
  );
  const [reviewer, setReviewer] = useState<Reviewer | null>(storedReviewer);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  useEffect(() => {
    const onHashChange = () => setView(viewFromHash(window.location.hash));
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    void getHealth().then(setApiUp);
  }, []);

  const rememberRater = (next: string) => {
    window.localStorage.setItem(RATER_KEY, next);
    setRaterId(next);
  };

  const rememberReviewer = (next: Reviewer) => {
    window.localStorage.setItem(REVIEWER_KEY, JSON.stringify(next));
    setReviewer(next);
  };

  const tabs: Array<{ label: string; fragment: string; active: boolean }> = [
    {
      label: 'Dashboard',
      fragment: '#/',
      active:
        view.name === 'dashboard' ||
        view.name === 'session' ||
        view.name === 'teacher',
    },
    { label: 'Annotate', fragment: '#/annotate', active: view.name === 'annotate' },
    {
      label: 'Calibration',
      fragment: '#/calibration',
      active: view.name === 'calibration',
    },
  ];

  let body: JSX.Element;
  if (view.name === 'dashboard') {
    body = <Dashboard onSelectSession={(id: string) => navigate(`#/session/${id}`)} />;
  } else if (view.name === 'session') {
    body = (
      <SessionView
        sessionId={view.sessionId}
        onBack={() => navigate('#/')}
        onReviewTeacher={() => navigate(`#/teacher/${view.sessionId}`)}
      />
    );
  } else if (view.name === 'teacher') {
    body = reviewer ? (
      <TeacherReview
        sessionId={view.sessionId}
        reviewer={reviewer}
        onBack={() => navigate(`#/session/${view.sessionId}`)}
      />
    ) : (
      <ReviewerGate onSet={rememberReviewer} />
    );
  } else if (view.name === 'annotate') {
    body = raterId ? (
      <AnnotationTool raterId={raterId} />
    ) : (
      <RaterGate onSet={rememberRater} />
    );
  } else {
    body = <CalibrationReview roundName={view.round} />;
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>PRAXIS</h1>
        <nav className="app-nav">
          {tabs.map((tab) => (
            <a
              key={tab.fragment}
              href={tab.fragment}
              className={tab.active ? 'nav-tab nav-tab-active' : 'nav-tab'}
            >
              {tab.label}
            </a>
          ))}
        </nav>
        <div className="header-info">
          {/* An unreachable API and an empty corpus render almost identically, and the
              remedies have nothing in common, so the state is named here rather than left
              for the reader to infer from empty tables. */}
          <span className={apiUp === false ? 'api-down' : 'api-up'}>
            {apiUp === null ? 'checking API…' : apiUp ? 'API up' : 'API unreachable'}
          </span>
          {view.name === 'teacher' && reviewer ? (
            <span className="rater-id">
              {reviewer.role} {reviewer.userId}{' '}
              <button
                type="button"
                className="link-button"
                onClick={() => {
                  window.localStorage.removeItem(REVIEWER_KEY);
                  setReviewer(null);
                }}
              >
                change
              </button>
            </span>
          ) : raterId ? (
            <span className="rater-id">
              {raterId}{' '}
              <button
                type="button"
                className="link-button"
                onClick={() => {
                  window.localStorage.removeItem(RATER_KEY);
                  setRaterId('');
                }}
              >
                change
              </button>
            </span>
          ) : null}
        </div>
      </header>
      <main className="app-main">{body}</main>
    </div>
  );
}
