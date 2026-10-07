/**
 * HTTP layer for teacher confirmation.
 *
 * Every call here carries the actor header, because the endpoints refuse a request without one:
 * a confirmation is attributed evidence and `confirmed_by` is the column R1 turns on. The token
 * is `role:user_id` and is recorded, not verified - Phase 9 adds verification, and until then
 * the server trusts what it is told and says so.
 *
 * The user id has to be a ULID. The audit trail stores an actor in a fixed-width column, so a
 * shorter value would be padded and every later verification of the chain would report a break;
 * the server refuses one with a 401 rather than writing it, and this client passes the reason
 * through rather than reducing it to "unauthorised".
 */

import type { ConfirmResult, TrackCandidates } from './types';

const API_BASE = '/api/v1/sessions';

export interface Reviewer {
  role: string;
  userId: string;
}

export function authHeader(reviewer: Reviewer): Record<string, string> {
  return { Authorization: `Bearer ${reviewer.role}:${reviewer.userId}` };
}

/**
 * Read a JSON response, or throw with the server's own reason. The API answers a refusal with
 * an RFC 7807 style body whose `detail` is written for a person to act on, so discarding it in
 * favour of a status code would throw away the only part that says what to do.
 */
async function request<T>(
  path: string,
  reviewer: Reviewer,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...authHeader(reviewer),
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? (body as { detail?: unknown }).detail
        : null;
    const reason =
      typeof detail === 'string'
        ? detail
        : detail && typeof detail === 'object' && 'detail' in detail
          ? String((detail as { detail?: unknown }).detail)
          : `${path} failed: ${response.status}`;
    const error = new Error(reason) as Error & { status: number; slug?: string };
    error.status = response.status;
    if (body && typeof body === 'object' && 'type' in body) {
      error.slug = String((body as { type?: unknown }).type);
    }
    throw error;
  }
  return response.json() as Promise<T>;
}

export function getCandidates(
  sessionId: string,
  reviewer: Reviewer,
  thumbnails = 5
): Promise<TrackCandidates> {
  const query = new URLSearchParams({ thumbnails: String(thumbnails) });
  return request<TrackCandidates>(
    `/${encodeURIComponent(sessionId)}/tracks/candidates?${query}`,
    reviewer
  );
}

export function confirmTrack(
  sessionId: string,
  trackId: number,
  reviewer: Reviewer
): Promise<ConfirmResult> {
  return request<ConfirmResult>(
    `/${encodeURIComponent(sessionId)}/tracks/confirm`,
    reviewer,
    { method: 'POST', body: JSON.stringify({ track_id: trackId }) }
  );
}

/**
 * A thumbnail is fetched rather than put straight in an `img src`, because the endpoint needs
 * the actor header and an `img` tag cannot carry one. The blob URL must be revoked by the
 * caller; a screen that creates one per candidate per render and never revokes them leaks the
 * decoded picture for as long as the tab is open.
 */
export async function fetchThumbnail(
  url: string,
  reviewer: Reviewer
): Promise<string> {
  const response = await fetch(url, { headers: authHeader(reviewer) });
  if (!response.ok) {
    throw new Error(`thumbnail ${url} failed: ${response.status}`);
  }
  return URL.createObjectURL(await response.blob());
}
