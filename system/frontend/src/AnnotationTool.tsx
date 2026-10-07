/**
 * AnnotationTool.tsx - The rater's screen for labelling teaching behaviours.
 *
 * Renders a video player, 8-second clip stepper, one panel per behaviour
 * from the codebook (never hardcoded), keyboard shortcuts, non-scorable escape,
 * and free-text note field. Codebook version is visible at all times.
 *
 * The tool fetches the codebook at startup and refuses to emit any label
 * not present in it. Every annotation records the codebook version it was
 * made under, enforced at the API level.
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  getCodebook,
  getQueue,
  submitAnnotation,
} from './api';
import type {
  Codebook,
  Assignment,
  BehaviourSpec,
  FieldSpec,
} from './types';
import './styles.css';

interface AnnotationState {
  [behaviourId: string]: {
    [fieldName: string]: unknown;
  };
}

interface AnnotationToolProps {
  raterId: string;
}

/**
 * Render a control for a single field based on its scale type.
 * The codebook declares scale_type and levels; the UI renders accordingly.
 */
function FieldControl(props: {
  field: FieldSpec;
  behaviour: BehaviourSpec;
  value: unknown;
  onChange: (value: unknown) => void;
  disabled?: boolean;
}): JSX.Element {
  const { field, value, onChange, disabled } = props;

  switch (field.scale_type) {
    case 'boolean':
      return (
        <div className="field-control">
          <label>{field.field_name}</label>
          <div className="boolean-buttons">
            <button
              onClick={() => onChange(true)}
              className={value === true ? 'active' : ''}
              disabled={disabled}
            >
              Yes
            </button>
            <button
              onClick={() => onChange(false)}
              className={value === false ? 'active' : ''}
              disabled={disabled}
            >
              No
            </button>
          </div>
        </div>
      );

    case 'count':
      return (
        <div className="field-control">
          <label>{field.field_name}</label>
          <input
            type="number"
            min="0"
            value={typeof value === 'number' ? value : ''}
            onChange={(e) => onChange(parseInt(e.target.value, 10) || 0)}
            disabled={disabled}
          />
        </div>
      );

    case 'proportion':
      return (
        <div className="field-control">
          <label>{field.field_name}</label>
          <div className="proportion-buttons">
            {field.levels.map((level) => (
              <button
                key={String(level)}
                onClick={() => onChange(level)}
                className={value === level ? 'active' : ''}
                disabled={disabled}
              >
                {String(level)}
              </button>
            ))}
          </div>
        </div>
      );

    case 'ordinal':
    case 'category':
      return (
        <div className="field-control">
          <label>{field.field_name}</label>
          <select
            value={typeof value === 'string' || typeof value === 'number' ? String(value) : ''}
            onChange={(e) => onChange(e.target.value || field.levels[0])}
            disabled={disabled}
          >
            <option value="">-- Select {field.field_name} --</option>
            {field.levels.map((level) => (
              <option key={String(level)} value={String(level)}>
                {String(level)}
              </option>
            ))}
          </select>
        </div>
      );

    default:
      return <div className="field-control">Unknown scale type: {field.scale_type}</div>;
  }
}

/**
 * Panel for a single behaviour. Contains controls for all fields defined
 * in the codebook for this behaviour.
 */
function BehaviourPanel(props: {
  behaviour: BehaviourSpec;
  value: Record<string, unknown> | undefined;
  onChange: (value: Record<string, unknown>) => void;
  disabled?: boolean;
  keyHint: string;
}): JSX.Element {
  const { behaviour, value = {}, onChange, disabled, keyHint } = props;

  return (
    <div className="behaviour-panel">
      <div className="behaviour-header">
        <h3>
          {behaviour.behaviour_id}: {behaviour.name}
        </h3>
        <div className="key-hint">{keyHint}</div>
      </div>
      <p className="behaviour-definition">{behaviour.definition}</p>
      <div className="behaviour-fields">
        {behaviour.fields.map((field) => (
          <FieldControl
            key={field.field_name}
            field={field}
            behaviour={behaviour}
            value={value[field.field_name]}
            onChange={(fieldValue) => {
              onChange({
                ...value,
                [field.field_name]: fieldValue,
              });
            }}
            disabled={disabled}
          />
        ))}
      </div>
    </div>
  );
}

export function AnnotationTool(props: AnnotationToolProps): JSX.Element {
  const { raterId } = props;

  const [codebook, setCodebook] = useState<Codebook | null>(null);
  const [codebookError, setCodebookError] = useState<string>('');

  const [assignments, setAssignments] = useState<Assignment[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [queueError, setQueueError] = useState<string>('');

  const [videoRef, setVideoRef] = useState<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const [annotations, setAnnotations] = useState<AnnotationState>({});
  const [isNonscorable, setIsNonscorable] = useState(false);
  const [note, setNote] = useState('');
  const [confidence, setConfidence] = useState<'certain' | 'uncertain'>('certain');

  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string>('');

  const [showKeyboardHints, setShowKeyboardHints] = useState(false);

  // Initialize: load codebook and queue
  useEffect(() => {
    async function init() {
      try {
        const cb = await getCodebook();
        setCodebook(cb);

        const q = await getQueue({
          rater_id: raterId,
          limit: 25,
        });
        setAssignments(q.assignments);
      } catch (error) {
        setCodebookError(`Failed to load codebook: ${String(error)}`);
      }
    }

    init();
  }, [raterId]);

  // Reset form when clip changes
  useEffect(() => {
    setAnnotations({});
    setIsNonscorable(false);
    setNote('');
    setConfidence('certain');
    setSubmitError('');
    setIsPlaying(false);
  }, [currentIndex]);

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyPress(event: KeyboardEvent) {
      switch (event.code) {
        case 'Space':
          event.preventDefault();
          if (videoRef) {
            if (videoRef.paused) {
              videoRef.play().catch(() => {});
            } else {
              videoRef.pause();
            }
            setIsPlaying(!videoRef.paused);
          }
          break;

        case 'KeyJ':
        case 'ArrowLeft':
          event.preventDefault();
          setCurrentIndex((i) => Math.max(0, i - 1));
          break;

        case 'KeyK':
        case 'ArrowRight':
          event.preventDefault();
          setCurrentIndex((i) => Math.min((assignments.length || 1) - 1, i + 1));
          break;

        case 'Digit1':
        case 'Digit2':
        case 'Digit3':
        case 'Digit4':
        case 'Digit5': {
          // Focus behaviour panel (1-5 for B1-B5)
          const idx = parseInt(event.code.charAt(5), 10) - 1;
          const panelId = `behaviour-${idx}`;
          document.getElementById(panelId)?.scrollIntoView({ behavior: 'smooth' });
          break;
        }

        case 'KeyN':
          event.preventDefault();
          setIsNonscorable((x) => !x);
          break;

        case 'Enter':
          event.preventDefault();
          handleSubmit();
          break;

        case 'Slash':
          if (event.shiftKey) {
            event.preventDefault();
            setShowKeyboardHints((x) => !x);
          }
          break;

        default:
          break;
      }
    }

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [videoRef, currentIndex, assignments, annotations, isNonscorable]);

  async function handleSubmit() {
    if (!codebook || assignments.length === 0) {
      setSubmitError('No codebook or assignments loaded');
      return;
    }

    const currentAssignment = assignments[currentIndex];
    if (!currentAssignment) {
      setSubmitError('No current assignment');
      return;
    }

    // Build labels object from annotations, or emit empty if non-scorable
    const labels: Record<string, unknown> = isNonscorable
      ? {}
      : annotations[currentAssignment.behaviour] || {};

    // Validate that all required fields are present (unless non-scorable)
    if (!isNonscorable) {
      const behaviourSpec = codebook.behaviours.find(
        (b) => b.behaviour_id === currentAssignment.behaviour
      );
      if (!behaviourSpec) {
        setSubmitError(`Behaviour ${currentAssignment.behaviour} not in codebook`);
        return;
      }

      const missingFields = behaviourSpec.fields.filter(
        (f) => !(f.field_name in labels)
      );
      if (missingFields.length > 0) {
        setSubmitError(
          `Missing required fields: ${missingFields.map((f) => f.field_name).join(', ')}`
        );
        return;
      }
    }

    setSubmitting(true);
    setSubmitError('');

    try {
      await submitAnnotation({
        clip_id: currentAssignment.clip.clip_id,
        rater_id: raterId,
        behaviour: currentAssignment.behaviour,
        labels,
        is_nonscorable: isNonscorable,
        note: note || undefined,
        rater_confidence: confidence,
        session_college_id: undefined,
      });

      // Move to next assignment
      if (currentIndex < assignments.length - 1) {
        setCurrentIndex(currentIndex + 1);
      } else {
        setSubmitError('All assignments completed! Reload to get more.');
      }
    } catch (error) {
      const err = error as Error & { status?: number };
      if (err.status === 409) {
        setSubmitError('This clip+behaviour has already been labelled by you.');
      } else if (err.status === 422) {
        setSubmitError(`Invalid label or behaviour: ${err.message}`);
      } else {
        setSubmitError(`Failed to submit: ${err.message}`);
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (codebookError) {
    return <div className="error-container">Error loading codebook: {codebookError}</div>;
  }

  if (!codebook) {
    return <div className="loading-container">Loading codebook...</div>;
  }

  if (queueError || assignments.length === 0) {
    return (
      <div className="error-container">
        No assignments in queue. {queueError && `Error: ${queueError}`}
      </div>
    );
  }

  const currentAssignment = assignments[currentIndex];
  const currentBehaviour = codebook.behaviours.find(
    (b) => b.behaviour_id === currentAssignment.behaviour
  );
  const clipStart = currentAssignment.clip.start_seconds;
  const clipEnd = currentAssignment.clip.end_seconds;

  return (
    <div className="annotation-tool">
      <header className="app-header">
        <h1>PRAXIS Annotation Tool</h1>
        <div className="header-info">
          <span className="codebook-version">Codebook {codebook.version}</span>
          <span className="rater-id">Rater: {raterId}</span>
        </div>
      </header>

      {showKeyboardHints && (
        <div className="keyboard-hints">
          <h4>Keyboard Shortcuts</h4>
          <ul>
            <li><kbd>Space</kbd> Play / Pause</li>
            <li><kbd>J</kbd> / <kbd>←</kbd> Previous clip</li>
            <li><kbd>K</kbd> / <kbd>→</kbd> Next clip</li>
            <li><kbd>1</kbd>-<kbd>5</kbd> Jump to behaviour panel</li>
            <li><kbd>N</kbd> Toggle non-scorable</li>
            <li><kbd>Enter</kbd> Submit annotation</li>
            <li><kbd>?</kbd> Toggle this help</li>
          </ul>
          <button onClick={() => setShowKeyboardHints(false)}>Close</button>
        </div>
      )}

      <div className="main-content">
        <div className="video-section">
          <div className="video-player-container">
            <video
              ref={setVideoRef}
              src={`/api/v1/media/${currentAssignment.clip.session_id}/video`}
              controls
              className="video-player"
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
            />
          </div>

          <div className="clip-info">
            <p>
              Clip {currentIndex + 1} of {assignments.length} |{' '}
              {currentAssignment.clip.start_seconds.toFixed(1)}s -{' '}
              {currentAssignment.clip.end_seconds.toFixed(1)}s (8 seconds)
            </p>
            <div className="clip-navigation">
              <button
                onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}
                disabled={currentIndex === 0}
              >
                ← Previous
              </button>
              <span className="progress-indicator">
                {currentIndex + 1} / {assignments.length}
              </span>
              <button
                onClick={() =>
                  setCurrentIndex(Math.min(assignments.length - 1, currentIndex + 1))
                }
                disabled={currentIndex === assignments.length - 1}
              >
                Next →
              </button>
            </div>
          </div>

          <div className="note-section">
            <label htmlFor="note-input">Free-text note:</label>
            <textarea
              id="note-input"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Optional note about this clip (e.g., recording quality, anomalies)"
              rows={3}
            />
          </div>
        </div>

        <div className="annotation-panels">
          <div className="behaviour-list">
            {codebook.behaviours.map((behaviour, idx) => (
              <div id={`behaviour-${idx}`} key={behaviour.behaviour_id}>
                <BehaviourPanel
                  behaviour={behaviour}
                  value={annotations[behaviour.behaviour_id]}
                  onChange={(value) => {
                    setAnnotations({
                      ...annotations,
                      [behaviour.behaviour_id]: value,
                    });
                  }}
                  disabled={isNonscorable}
                  keyHint={`Press ${idx + 1} to jump here`}
                />
              </div>
            ))}
          </div>

          <div className="submission-section">
            <div className="non-scorable-control">
              <label>
                <input
                  type="checkbox"
                  checked={isNonscorable}
                  onChange={(e) => setIsNonscorable(e.target.checked)}
                />
                Non-scorable (escape hatch)
              </label>
              <p className="help-text">Check if the clip cannot be labelled (e.g., no teacher visible).</p>
            </div>

            <div className="confidence-control">
              <label>Rater confidence:</label>
              <div className="confidence-buttons">
                <button
                  onClick={() => setConfidence('certain')}
                  className={confidence === 'certain' ? 'active' : ''}
                >
                  Certain
                </button>
                <button
                  onClick={() => setConfidence('uncertain')}
                  className={confidence === 'uncertain' ? 'active' : ''}
                >
                  Uncertain
                </button>
              </div>
            </div>

            {submitError && (
              <div className="error-message">
                {submitError}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={submitting}
              className="submit-button"
            >
              {submitting ? 'Submitting...' : 'Submit Annotation (Enter)'}
            </button>

            <button
              onClick={() => setShowKeyboardHints((x) => !x)}
              className="help-button"
            >
              ? Keyboard Shortcuts
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
