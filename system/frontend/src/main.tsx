/**
 * main.tsx - application entry point.
 *
 * Mounts the shell, which decides which view to show. Nothing blocking happens here: the
 * entry point used to call window.prompt() at module scope for a rater id, so the page
 * rendered nothing at all until the dialog was answered and cancelling it filed annotations
 * under 'unknown-rater'. App asks in the page, and only on the view that needs an answer.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import './styles.css';

const root = document.getElementById('root');
if (!root) {
  throw new Error('no #root element: index.html and main.tsx disagree');
}

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
