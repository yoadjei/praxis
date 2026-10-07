/**
 * CalibrationReview.tsx - Side-by-side disagreement review screen.
 *
 * Shows disagreements between raters on a calibration round, allowing
 * researchers to identify and address sources of systematic disagreement
 * in the codebook definitions.
 */

import React, { useState, useEffect } from 'react';
import { getDisagreements, getAgreement } from './api';
import type { Disagreement, CalibrationOutcome } from './types';
import './styles.css';

interface CalibrationReviewProps {
  roundName: string; // e.g., "calibration-1", "calibration-2"
}

/**
 * Display a single disagreement with values from each rater side-by-side.
 */
function DisagreementCard(props: {
  disagreement: Disagreement;
  roundName: string;
}): JSX.Element {
  const { disagreement } = props;

  return (
    <div className="disagreement-card">
      <div className="disagreement-header">
        <h4>
          {disagreement.behaviour} - {disagreement.field}
        </h4>
        <span className="distinct-count">{disagreement.distinct_values} distinct values</span>
      </div>

      <div className="disagreement-values">
        {Object.entries(disagreement.by_rater).map(([raterId, value]) => (
          <div key={raterId} className="rater-column">
            <div className="rater-id">{raterId}</div>
            <div className="rater-value">
              <code>{JSON.stringify(value, null, 2)}</code>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function CalibrationReview(props: CalibrationReviewProps): JSX.Element {
  const { roundName } = props;

  const [disagreements, setDisagreements] = useState<Disagreement[]>([]);
  const [agreement, setAgreement] = useState<CalibrationOutcome | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');

  const [filteredBehaviour, setFilteredBehaviour] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [disagreementData, agreementData] = await Promise.all([
          getDisagreements({ round_name: roundName }),
          getAgreement({ round_name: roundName }),
        ]);

        setDisagreements(disagreementData.disagreements);
        setAgreement(agreementData);
      } catch (err) {
        setError(`Failed to load calibration data: ${String(err)}`);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [roundName]);

  if (loading) {
    return <div className="loading-container">Loading calibration data...</div>;
  }

  if (error) {
    return <div className="error-container">Error: {error}</div>;
  }

  if (!agreement) {
    return <div className="error-container">No calibration data available</div>;
  }

  // Filter disagreements by behaviour if selected
  const filtered =
    filteredBehaviour === null
      ? disagreements
      : disagreements.filter((d) => d.behaviour === filteredBehaviour);

  // Get unique behaviours for filter
  const uniqueBehaviours = Array.from(
    new Set(disagreements.map((d) => d.behaviour))
  ).sort();

  return (
    <div className="calibration-review">
      <header className="app-header">
        <h1>Calibration Review: {roundName}</h1>
        <div className="header-info">
          <span>
            {agreement.n_clips} clips × {agreement.n_raters} raters
          </span>
        </div>
      </header>

      <section className="agreement-summary">
        <h2>Inter-Rater Agreement</h2>
        <div className="agreement-table">
          <table>
            <thead>
              <tr>
                <th>Behaviour</th>
                <th>Agreement</th>
                <th>Method</th>
                <th>95% CI</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(agreement.agreement).map(([behaviour, report]) => (
                <tr key={behaviour}>
                  <td className="behaviour-id">{behaviour}</td>
                  <td className="agreement-value">
                    {(report.agreement * 100).toFixed(1)}%
                  </td>
                  <td className="method">{report.method}</td>
                  <td className="confidence-interval">
                    [{report.confidence_interval[0].toFixed(3)}, {report.confidence_interval[1].toFixed(3)}]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {agreement.gate_passed ? (
          <p className="gate-passed">
            ✓ Calibration gate passed (threshold: {agreement.gate_value})
          </p>
        ) : (
          <p className="gate-failed">
            ✗ Calibration gate not passed (threshold: {agreement.gate_value})
          </p>
        )}
      </section>

      <section className="disagreements-section">
        <div className="disagreements-header">
          <h2>Disagreements</h2>
          <div className="filter-controls">
            <label>Filter by behaviour:</label>
            <select
              value={filteredBehaviour || 'all'}
              onChange={(e) => setFilteredBehaviour(e.target.value === 'all' ? null : e.target.value)}
            >
              <option value="all">All behaviours</option>
              {uniqueBehaviours.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </div>
        </div>

        {filtered.length === 0 ? (
          <p className="no-disagreements">No disagreements found.</p>
        ) : (
          <div className="disagreement-list">
            {filtered.map((disagreement, idx) => (
              <DisagreementCard
                key={idx}
                disagreement={disagreement}
                roundName={roundName}
              />
            ))}
          </div>
        )}
      </section>

      <section className="calibration-actions">
        <p>
          Review these disagreements and update the codebook definitions as needed.
          Re-run the calibration round after making changes.
        </p>
      </section>
    </div>
  );
}
