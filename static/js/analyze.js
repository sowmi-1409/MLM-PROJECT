/**
 * NewsGuard News Analysis & Real-Time Forensics Engine
 */

let currentSamples = {};

document.addEventListener('DOMContentLoaded', () => {
  initCounters();
  initFormActions();
  initUrlFetcher();
  loadSampleRegistry();
});

// Character and Word Counters
function initCounters() {
  const headlineInput = document.getElementById('headline');
  const bodyInput = document.getElementById('body-text');
  const headlineCount = document.getElementById('headline-count');
  const bodyCount = document.getElementById('body-count');

  if (headlineInput && headlineCount) {
    headlineInput.addEventListener('input', () => {
      const len = headlineInput.value.length;
      headlineCount.textContent = `${len} / 500 chars`;
    });
  }

  if (bodyInput && bodyCount) {
    bodyInput.addEventListener('input', () => {
      const words = bodyInput.value.trim().split(/\s+/).filter(w => w.length > 0);
      bodyCount.textContent = `${words.length} words`;
    });
  }
}

// Sample Articles Registry
async function loadSampleRegistry() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    if (data.success && data.samples) {
      data.samples.forEach(s => {
        currentSamples[s.id] = s;
      });
    }
  } catch (err) {
    console.error('Could not load sample registry:', err);
  }
}

function loadSample(sampleId) {
  const sample = currentSamples[sampleId];
  if (!sample) return;

  const headlineInput = document.getElementById('headline');
  const bodyInput = document.getElementById('body-text');
  const urlInput = document.getElementById('article-url');

  if (headlineInput) {
    headlineInput.value = sample.headline;
    headlineInput.dispatchEvent(new Event('input'));
  }
  if (bodyInput) {
    bodyInput.value = sample.body;
    bodyInput.dispatchEvent(new Event('input'));
  }
  if (urlInput) {
    urlInput.value = '';
  }

  hideValidationError();
  // Smooth scroll to form
  document.getElementById('analysis-form').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// URL Article Fetcher
function initUrlFetcher() {
  const fetchBtn = document.getElementById('btn-fetch-url');
  const urlInput = document.getElementById('article-url');
  const spinner = document.getElementById('fetch-spinner');

  if (!fetchBtn || !urlInput) return;

  fetchBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url) {
      showValidationError('Please enter a valid website URL to fetch.');
      return;
    }

    hideValidationError();
    fetchBtn.disabled = true;
    if (spinner) spinner.classList.remove('d-none');

    try {
      const res = await fetch('/api/fetch-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      const data = await res.json();

      if (data.success) {
        if (data.headline) {
          const h = document.getElementById('headline');
          h.value = data.headline;
          h.dispatchEvent(new Event('input'));
        }
        if (data.body) {
          const b = document.getElementById('body-text');
          b.value = data.body;
          b.dispatchEvent(new Event('input'));
        }
      } else {
        showValidationError(data.error || 'Failed to extract content from URL.');
      }
    } catch (err) {
      showValidationError('Network error while connecting to server.');
    } finally {
      fetchBtn.disabled = false;
      if (spinner) spinner.classList.add('d-none');
    }
  });
}

// Form Submission & Analysis
function initFormActions() {
  const form = document.getElementById('analysis-form');
  const clearBtn = document.getElementById('btn-clear');

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      document.getElementById('headline').value = '';
      document.getElementById('body-text').value = '';
      document.getElementById('article-url').value = '';
      document.getElementById('headline').dispatchEvent(new Event('input'));
      document.getElementById('body-text').dispatchEvent(new Event('input'));
      hideValidationError();
      document.getElementById('results-section').classList.add('d-none');
      document.getElementById('loading-state').classList.add('d-none');
    });
  }

  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const headline = document.getElementById('headline').value.trim();
      const body = document.getElementById('body-text').value.trim();
      const url = document.getElementById('article-url').value.trim();

      // Client validation
      if (!headline) {
        showValidationError('News headline is required.');
        return;
      }
      if (headline.split(/\s+/).length < 3) {
        showValidationError('Headline is too brief. Please enter at least 3 words.');
        return;
      }

      hideValidationError();
      setLoadingState(true);

      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ headline, body, url })
        });
        const data = await response.json();

        if (data.success) {
          renderResults(data.results);
          updateSidebarCount(data.session_stats.total_analyzed);
        } else {
          showValidationError(data.error || 'Analysis failed. Please check inputs.');
        }
      } catch (err) {
        showValidationError('Network error while analyzing article. Please check backend server.');
      } finally {
        setLoadingState(false);
      }
    });
  }
}

function setLoadingState(isLoading) {
  const analyzeBtn = document.getElementById('btn-analyze');
  const btnText = analyzeBtn.querySelector('.btn-text');
  const btnSpinner = document.getElementById('analyze-spinner');
  const loadingState = document.getElementById('loading-state');
  const resultsSection = document.getElementById('results-section');

  if (isLoading) {
    analyzeBtn.disabled = true;
    btnText.classList.add('d-none');
    btnSpinner.classList.remove('d-none');
    loadingState.classList.remove('d-none');
    resultsSection.classList.add('d-none');
  } else {
    analyzeBtn.disabled = false;
    btnText.classList.remove('d-none');
    btnSpinner.classList.add('d-none');
    loadingState.classList.add('d-none');
  }
}

function showValidationError(msg) {
  const errBox = document.getElementById('validation-error');
  const errText = document.getElementById('validation-error-text');
  if (errBox && errText) {
    errText.textContent = msg;
    errBox.classList.remove('d-none');
  }
}

function hideValidationError() {
  const errBox = document.getElementById('validation-error');
  if (errBox) errBox.classList.add('d-none');
}

function updateSidebarCount(count) {
  const badge = document.getElementById('session-count-badge');
  if (badge) badge.textContent = `Analyzed: ${count}`;
}

// Render Results into UI Cards
function renderResults(res) {
  const resultsSection = document.getElementById('results-section');
  resultsSection.classList.remove('d-none');

  const { fake_news, clickbait, mismatch, external_verification, synthesis } = res;

  // 1. Overall Synthesis Banner
  const synthBadge = document.getElementById('synthesis-badge');
  synthBadge.className = `synthesis-badge badge-${synthesis.overall_badge}`;
  synthBadge.textContent = synthesis.overall_badge.toUpperCase() + ' PROFILE';

  document.getElementById('synthesis-title').textContent = synthesis.overall_summary;
  document.getElementById('ethical-ai-notice').textContent = synthesis.core_disclaimer;

  // Risk Flags List
  const riskList = document.getElementById('risk-flags-list');
  if (synthesis.risk_flags && synthesis.risk_flags.length > 0) {
    riskList.innerHTML = synthesis.risk_flags.map(f => `
      <div class="signal-item border-danger">
        <strong class="text-danger">${f.title}</strong>
        <p class="text-muted text-xs mb-0">${f.description}</p>
      </div>
    `).join('');
  } else {
    riskList.innerHTML = '<span class="text-muted text-sm">No significant forensic risk flags detected.</span>';
  }

  // Positive Signals List
  const posList = document.getElementById('positive-signals-list');
  if (synthesis.positive_signals && synthesis.positive_signals.length > 0) {
    posList.innerHTML = synthesis.positive_signals.map(s => `
      <div class="signal-item border-success">
        <strong class="text-success">${s.title}</strong>
        <p class="text-muted text-xs mb-0">${s.description}</p>
      </div>
    `).join('');
  } else {
    posList.innerHTML = '<span class="text-muted text-sm">No positive verification signals identified.</span>';
  }

  // 2. Fake News Card
  const fnBadge = document.getElementById('fn-badge');
  fnBadge.className = `badge ${fake_news.prediction_code === 1 ? 'badge-danger' : 'badge-success'}`;
  fnBadge.textContent = fake_news.label;

  document.getElementById('fn-label').textContent = fake_news.label;
  document.getElementById('fn-confidence').textContent = `${fake_news.confidence_percent}%`;
  document.getElementById('fn-explanation').textContent = fake_news.explanation;
  document.getElementById('fn-limitations').textContent = fake_news.limitations;

  const probReal = Math.round(fake_news.probabilities.real * 100);
  const probFake = Math.round(fake_news.probabilities.fake * 100);
  document.getElementById('fn-prob-real').textContent = `${probReal}%`;
  document.getElementById('fn-prob-fake').textContent = `${probFake}%`;
  document.getElementById('fn-bar-real').style.width = `${probReal}%`;
  document.getElementById('fn-bar-fake').style.width = `${probFake}%`;

  const fnTokensWrap = document.getElementById('fn-influential-tokens');
  if (fake_news.influential_features && fake_news.influential_features.length > 0) {
    fnTokensWrap.innerHTML = fake_news.influential_features.map(t => `
      <span class="token-tag ${t.lean === 'Fake' ? 'tag-fake' : 'tag-real'}" title="Weight impact: ${t.weight}">
        ${t.word} (${t.weight > 0 ? '+' : ''}${t.weight})
      </span>
    `).join('');
  } else {
    fnTokensWrap.innerHTML = '<span class="text-muted text-xs">No prominent lexical markers identified.</span>';
  }

  // 3. Clickbait Card
  const cbBadge = document.getElementById('cb-badge');
  cbBadge.className = `badge ${clickbait.prediction_code === 1 ? 'badge-warning' : 'badge-neutral'}`;
  cbBadge.textContent = clickbait.label;

  document.getElementById('cb-label').textContent = clickbait.label;
  document.getElementById('cb-confidence').textContent = `${clickbait.confidence_percent}%`;
  document.getElementById('cb-explanation').textContent = clickbait.explanation;
  document.getElementById('cb-limitations').textContent = clickbait.limitations;

  const cbCuesWrap = document.getElementById('cb-cues');
  if (clickbait.stylistic_cues && clickbait.stylistic_cues.length > 0) {
    cbCuesWrap.innerHTML = clickbait.stylistic_cues.map(c => `
      <span class="badge badge-warning">
        <i class="fa-solid fa-bolt"></i> ${c.pattern}: "${c.matched_text}"
      </span>
    `).join('');
  } else {
    cbCuesWrap.innerHTML = '<span class="badge badge-neutral"><i class="fa-solid fa-check"></i> Standard journalistic phrasing</span>';
  }

  const cbTokensWrap = document.getElementById('cb-influential-tokens');
  if (clickbait.influential_words && clickbait.influential_words.length > 0) {
    cbTokensWrap.innerHTML = clickbait.influential_words.map(w => `
      <span class="token-tag ${w.lean === 'Clickbait' ? 'tag-clickbait' : 'tag-informative'}">
        ${w.word} (${w.weight > 0 ? '+' : ''}${w.weight})
      </span>
    `).join('');
  } else {
    cbTokensWrap.innerHTML = '<span class="text-muted text-xs">No distinctive clickbait terms found.</span>';
  }

  // 4. Headline-Body Consistency Card
  const mmBadge = document.getElementById('mm-badge');
  mmBadge.className = `badge badge-${mismatch.badge || 'info'}`;
  mmBadge.textContent = mismatch.label;

  document.getElementById('mm-label').textContent = mismatch.label;
  document.getElementById('mm-score').textContent = mismatch.similarity_score.toFixed(4);
  document.getElementById('mm-score-text').textContent = mismatch.similarity_score.toFixed(4);
  document.getElementById('mm-threshold-val').textContent = mismatch.calibrated_threshold || '0.100';
  document.getElementById('mm-explanation').textContent = mismatch.explanation;
  document.getElementById('mm-limitations').textContent = mismatch.limitations;

  const pct = Math.min(Math.max(mismatch.similarity_percentage || 0, 5), 100);
  document.getElementById('mm-progress-bar').style.width = `${pct}%`;

  const sharedTerms = document.getElementById('mm-shared-terms');
  if (mismatch.shared_keywords && mismatch.shared_keywords.length > 0) {
    sharedTerms.innerHTML = mismatch.shared_keywords.map(k => `
      <span class="token-tag tag-real">${k}</span>
    `).join('');
  } else {
    sharedTerms.innerHTML = '<span class="text-muted text-xs">No significant shared entities found.</span>';
  }

  const missingTerms = document.getElementById('mm-missing-terms');
  if (mismatch.unmatched_headline_terms && mismatch.unmatched_headline_terms.length > 0) {
    missingTerms.innerHTML = mismatch.unmatched_headline_terms.map(k => `
      <span class="token-tag tag-fake">${k}</span>
    `).join('');
  } else {
    missingTerms.innerHTML = '<span class="text-muted text-xs">All headline entities appear in body.</span>';
  }

  // 5. External Verification Card
  const evBadge = document.getElementById('ev-badge');
  evBadge.className = `badge badge-${external_verification.badge || 'neutral'}`;
  evBadge.textContent = external_verification.verdict;

  document.getElementById('ev-verdict').textContent = external_verification.verdict;
  document.getElementById('ev-query').textContent = external_verification.query_used || 'None';
  document.getElementById('ev-explanation').textContent = external_verification.explanation;
  document.getElementById('ev-limitations').textContent = external_verification.limitations;

  const sourcesList = document.getElementById('ev-sources-list');
  if (external_verification.sources && external_verification.sources.length > 0) {
    sourcesList.innerHTML = external_verification.sources.map(s => `
      <div class="source-card p-2 mb-2 bg-secondary border rounded">
        <a href="${s.url}" target="_blank" rel="noopener noreferrer" class="source-link fw-bold d-block text-primary">
          <i class="fa-solid fa-arrow-up-right-from-square me-1"></i> ${s.title}
        </a>
        <p class="text-muted text-xs mb-0">${s.snippet || 'Public encyclopedia entry matching query.'}</p>
      </div>
    `).join('');
  } else {
    sourcesList.innerHTML = '<span class="text-muted text-xs">No direct corroborating records retrieved. Manual checking suggested.</span>';
  }

  // Smooth scroll to results
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
