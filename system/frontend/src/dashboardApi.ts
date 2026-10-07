/**
 * HTTP layer for /api/v1/dashboard/*.
 *
 * Kept apart from api.ts because the two serve different readers: api.ts is the rater's
 * annotation queue, this is the researcher's view of the corpus and the phase runs. They
 * share a base path and nothing else.
 */

import type {
  DashboardSummary,
  SessionPage,
  SessionDetail,
  RunPage,
} from './types';

const API_BASE = '/api/v1/dashboard';

/**
 * Read a JSON response or throw with the status and, when the server sent one, its reason.
 * The API answers a refusal with an RFC 7807 style body, so the message a user sees is the
 * one the server chose rather than a bare status code.
 */
async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? (body as { detail?: unknown }).detail
        : null;
    const reason =
      detail && typeof detail === 'object' && 'detail' in detail
        ? String((detail as { detail?: unknown }).detail)
        : typeof detail === 'string'
          ? detail
          : `${path} failed: ${response.status}`;
    const error = new Error(reason) as Error & { status: number };
    error.status = response.status;
    throw error;
  }
  return response.json() as Promise<T>;
}

export function getSummary(): Promise<DashboardSummary> {
  return getJson<DashboardSummary>('/summary');
}

export function getSessions(params: {
  skip?: number;
  limit?: number;
} = {}): Promise<SessionPage> {
  const search = new URLSearchParams();
  if (params.skip !== undefined) search.append('skip', String(params.skip));
  if (params.limit !== undefined) search.append('limit', String(params.limit));
  const query = search.toString();
  return getJson<SessionPage>(`/sessions${query ? `?${query}` : ''}`);
}

export function getSessionDetail(sessionId: string): Promise<SessionDetail> {
  return getJson<SessionDetail>(`/sessions/${encodeURIComponent(sessionId)}`);
}

export function getRuns(limit?: number): Promise<RunPage> {
  return getJson<RunPage>(`/runs${limit === undefined ? '' : `?limit=${limit}`}`);
}

/**
 * Whether the API is reachable at all. Used by the shell to tell "the server is down" apart
 * from "the corpus is empty", which look identical in an empty dashboard and have entirely
 * different remedies.
 */
export async function getHealth(): Promise<boolean> {
  try {
    const response = await fetch('/api/v1/health');
    return response.ok;
  } catch {
    return false;
  }
}
