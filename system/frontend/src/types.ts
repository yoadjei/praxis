/**
 * API contract types for the annotation frontend.
 * Mirror the shapes from praxis/annotation/server.py and praxis/contracts/
 */

export interface ClipRef {
  session_id: string;
  clip_index: number; // 0-based
  start_seconds: number;
  end_seconds: number;
  clip_id: string; // Computed as `${session_id}:${clip_index:05d}`
}

export interface Assignment {
  assignment_id: string;
  rater_id: string;
  clip: ClipRef;
  behaviour: string; // BehaviourId: "B1" | "B2" | "B3" | "B4" | "B5"
  round_name: string; // "calibration-1" | "calibration-2" | "production"
}

export interface FieldSpec {
  field_name: string;
  scale_type: "boolean" | "count" | "proportion" | "ordinal" | "category";
  levels: (string | number | boolean)[];
  description: string;
}

export interface BehaviourSpec {
  behaviour_id: string; // "B1" | "B2" | ... | "B5"
  name: string;
  definition: string;
  fields: FieldSpec[];
}

export interface Codebook {
  version: string; // e.g., "1.0.0"
  created_at: string;
  behaviours: BehaviourSpec[];
}

export interface Annotation {
  annotation_id: string;
  clip_id: string;
  rater_id: string;
  behaviour: string; // BehaviourId
  labels: Record<string, unknown>; // Shape matches codebook field definitions
  is_nonscorable: boolean;
  note: string | null;
  rater_confidence: "certain" | "uncertain";
  codebook_version: string;
  created_at: string;
}

export interface Disagreement {
  clip_id: string;
  behaviour: string;
  field: string;
  by_rater: Record<string, unknown>;
  distinct_values: number;
}

export interface AgreementReport {
  agreement: number;
  method: string;
  confidence_interval: [number, number];
}

export interface CalibrationOutcome {
  round_name: string;
  n_clips: number;
  n_raters: number;
  agreement: Record<string, AgreementReport>; // Per behaviour
  disagreements: Disagreement[];
  gate_value: number | null;
  gate_passed: boolean;
}

export interface AnnotationRefusedError {
  reason: string;
  http_status: number;
}

/**
 * Dashboard shapes, mirroring praxis/api/routes/dashboard.py.
 *
 * Nothing here carries a teacher id, a teacher code or a media filename. That is not an
 * oversight in the types: R1 keeps identity out of the system and D76 keeps the
 * re-identifying key out of the database, so the server never sends those fields and the
 * dashboard has no way to render them.
 */

export interface DashboardSummary {
  corpus_by_domain: Record<string, Record<string, number>>;
  preprocessing: { preprocessed_count: number; total_sessions: number };
  annotation: { assignments: number; annotations: number };
  last_audit_at: string | null;
  model: { status: string } & Record<string, unknown>;
  confidence_distribution: { status?: string } & Record<string, unknown>;
}

export interface SessionRow {
  session_id: string;
  domain: string;
  recorded_on: string | null;
  quality_verdict: string;
  duration_s: number;
  preprocessed: boolean;
  annotated: boolean;
  /**
   * When a researcher decided this session will not be annotated, and why, in their own words.
   * Null for every session nobody has excluded.
   *
   * Shown rather than filtered. An excluded session hidden from the list is indistinguishable
   * from one nobody has reached - which is the state D96 exists to end - so the row stays and
   * carries its ground. Distinct from quality_verdict: that is what the gate measured about the
   * file, this is a decision about research use.
   */
  excluded_at: string | null;
  exclusion_reason: string | null;
}

export interface SessionPage {
  sessions: SessionRow[];
  total_count: number;
  skip: number;
  limit: number;
}

export interface SessionDetail {
  session_id: string;
  domain: string;
  recorded_on: string | null;
  quality_verdict: string;
  subject: string | null;
  grade_level: string | null;
  duration_s: number;
  preprocessed: boolean;
  annotation_count: number;
  created_at: string | null;
}

/**
 * A phase run. `verdict` is "ran" or "abstained"; an abstention is its own verdict and
 * carries the reason it could not run, which is the whole point of recording it (D48).
 */
export interface RunRow {
  run_id: string;
  phase: string;
  verdict: 'ran' | 'abstained';
  finished_at: string | null;
  abstention_reason?: string;
  abstention_missing?: string[];
}

/**
 * `run_directories` counts what is on disk, which is not the same as the number of runs: a
 * half-written or corrupted directory is skipped and still counted here. The panel says
 * "directories" for that reason rather than quietly calling them runs.
 */
export interface RunPage {
  runs: RunRow[];
  run_directories: number;
  limit: number;
}

/**
 * Teacher confirmation, mirroring praxis/api/routes/tracks.py.
 *
 * `proposal` and `confirmation` are separate on purpose and are not merged anywhere in this
 * client either. The heuristic's pick is a guess and the reviewer's is a judgement, and the
 * whole of `praxis/preprocess/teacher.py` exists so that no object in the system can carry an
 * unconfirmed identification as though somebody had agreed to it.
 */

export interface ThumbnailRef {
  index: number;
  at_seconds: number;
  frame: number;
  box: { x: number; y: number; width: number; height: number };
  url: string;
}

export interface TrackCandidate {
  track_id: number;
  score: number | null;
  presence_fraction: number | null;
  median_area_fraction: number | null;
  front_zone_fraction: number | null;
  is_proposed: boolean;
  is_confirmed: boolean;
  thumbnails: ThumbnailRef[];
}

/**
 * `available` false means the ranking was never recorded, which is not the same as a session
 * in which the heuristic found nobody. `detail` carries the remedy; an empty candidate list
 * with no explanation would read as the second while being the first (D48).
 */
export interface Ranking {
  source: string;
  zones_available: boolean | null;
  truncated: number;
  available: boolean;
  detail: string | null;
}

export interface TrackCandidates {
  session_id: string;
  proposal: {
    track_id: number | null;
    score: number | null;
    proposed_by: string;
    reason: string;
  };
  confirmation: { confirmed_by: string | null; confirmed_at: string | null };
  ranking: Ranking;
  thumbnails_unavailable: string | null;
  candidates: TrackCandidate[];
}

export interface ConfirmResult {
  session_id: string;
  track_id: number;
  confirmed_by: string;
}
