/**
 * HTTP API layer for the annotation tool.
 * Fetches from /api/v1/annotation endpoints, with the dev server proxying
 * to http://localhost:8000.
 */

import type {
  Codebook,
  Assignment,
  Annotation,
  Disagreement,
  CalibrationOutcome,
} from './types';

const API_BASE = '/api/v1/annotation';

/**
 * Fetch the active codebook for the UI to render.
 * Raises on network error; caller is responsible for displaying feedback.
 */
export async function getCodebook(): Promise<Codebook> {
  const response = await fetch(`${API_BASE}/codebook`);
  if (!response.ok) {
    throw new Error(`Failed to fetch codebook: ${response.status}`);
  }
  return response.json();
}

/**
 * Fetch the queue of assignments for a rater.
 * limit: maximum assignments to return, optional
 * round_name: filter by round, optional
 */
export async function getQueue(params: {
  rater_id: string;
  limit?: number;
  round_name?: string;
}): Promise<{ assignments: Assignment[] }> {
  const searchParams = new URLSearchParams({
    rater_id: params.rater_id,
  });
  if (params.limit !== undefined) {
    searchParams.append('limit', params.limit.toString());
  }
  if (params.round_name !== undefined) {
    searchParams.append('round_name', params.round_name);
  }

  const response = await fetch(`${API_BASE}/queue?${searchParams}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch queue: ${response.status}`);
  }
  return response.json();
}

/**
 * Submit an annotation for a clip behaviour.
 * Raises with status 422 if label is unknown or behaviour is unknown.
 * Raises with status 409 if the rater already labelled this clip+behaviour.
 */
export async function submitAnnotation(params: {
  clip_id: string;
  rater_id: string;
  behaviour: string; // BehaviourId
  labels: Record<string, unknown>; // Shape must match codebook
  is_nonscorable?: boolean;
  note?: string;
  rater_confidence?: 'certain' | 'uncertain';
  session_college_id?: string;
}): Promise<{ annotation_id: string; codebook_version: string }> {
  const payload = {
    clip_id: params.clip_id,
    rater_id: params.rater_id,
    behaviour: params.behaviour,
    labels: params.labels,
    is_nonscorable: params.is_nonscorable || false,
    note: params.note || null,
    rater_confidence: params.rater_confidence || 'certain',
    session_college_id: params.session_college_id || null,
  };

  const response = await fetch(`${API_BASE}/labels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error: unknown = await response.json().catch(() => ({}));
    const message =
      error instanceof Object && 'reason' in error
        ? (error as { reason?: string }).reason
        : `Failed to submit annotation: ${response.status}`;
    const err = new Error(message) as Error & { status: number };
    err.status = response.status;
    throw err;
  }

  return response.json();
}

/**
 * Fetch disagreements for a calibration round.
 * Used by CalibrationReview to display side-by-side comparison.
 */
export async function getDisagreements(params: {
  round_name: string;
}): Promise<{ disagreements: Disagreement[] }> {
  const searchParams = new URLSearchParams({
    round_name: params.round_name,
  });

  const response = await fetch(
    `${API_BASE}/calibration/${params.round_name}/disagreements?${searchParams}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch disagreements: ${response.status}`);
  }
  return response.json();
}

/**
 * Fetch the agreement report for a calibration round.
 * Returns per-behaviour agreement metrics.
 */
export async function getAgreement(params: {
  round_name?: string;
}): Promise<CalibrationOutcome> {
  const searchParams = new URLSearchParams();
  if (params.round_name !== undefined) {
    searchParams.append('round_name', params.round_name);
  }

  const response = await fetch(`${API_BASE}/agreement?${searchParams}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch agreement: ${response.status}`);
  }
  return response.json();
}
