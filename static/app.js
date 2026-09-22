/* No build step, external scripts, telemetry, or third-party image requests. */
'use strict';
const $ = id => document.getElementById(id);
const $$ = selector => [...document.querySelectorAll(selector)];
const state = {data: null, sample: 'shapes', file: null, tab: 'activations', layer: 'features.0.0', page: 0, filterPage: 0, generation: 0, viewGeneration: 0, classes: [], channelImages: {}, filterImages: {}};
const esc = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const compact = value => value >= 1e6 ? `${(value / 1e6).toFixed(2)}M` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}K` : String(value);
const number = value => Number(value).toPrecision(4);
function busy(on, message = '') {
  $('loading-line').hidden = !on;
  $('workspace').setAttribute('aria-busy', String(on));
  if (message) $('workspace-status').textContent = message;
}
function error(message) { $('error-box').hidden = false; $('error-message').textContent = message; }
function clearError() { $('error-box').hidden = true; }
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try { const body = await response.json(); message = typeof body.detail === 'string' ? body.detail : 'Check the selected input and display options.'; } catch (_) { /* keep HTTP status */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}
function endpoint(action, params = {}) {
  return `/api/analysis/${encodeURIComponent(state.data.id)}/${action}?${new URLSearchParams(params)}`;
}
function enableAnalysisControls(enabled) {
  ['layer-select','export-button','cam-button','clear-session'].forEach(id => $(id).disabled = !enabled);
  if (!enabled) ['prev-layer','next-layer','prev-page','next-page','prev-filter-page','next-filter-page'].forEach(id => $(id).disabled = true);
}
async function analyze() {
  const generation = ++state.generation;
  ++state.viewGeneration;
  const oldId = state.data?.id;
  state.data = null;
  resetVisuals();
  state.page = state.filterPage = 0;
  enableAnalysisControls(false);
  clearError();
  $('demo-warning').hidden = true;
  $('cam-results').hidden = true;
  $('cam-download').hidden = true;
  $('cam-note').textContent = '';
  $('prediction-status').textContent = 'Analyzing…';
  busy(true, 'Loading model and tracing the image. A first trained run may download weights…');
  try {
    const params = {weights: $('weights').value};
    if (!state.file) params.sample = state.sample;
    const data = await api(`/api/analyze?${new URLSearchParams(params)}`, {
      method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: state.file || undefined
    });
    if (generation !== state.generation) { api(`/api/analysis/${data.id}`, {method:'DELETE'}).catch(() => {}); return; }
    state.data = data;
    if (oldId) api(`/api/analysis/${oldId}`, {method:'DELETE'}).catch(() => {});
    if (!data.model.layers.some(layer => layer.name === state.layer)) state.layer = data.model.layers[0].name;
    enableAnalysisControls(true);
    renderSummary();
    renderArchitecture();
    renderLayerControls();
    $('class-target').value = classLabel(data.predictions[0].index);
    closeMenu();
    await renderView();
  } catch (exception) {
    if (generation !== state.generation) return;
    error(exception.message);
    $('prediction-status').textContent = 'Load failed';
    $('workspace-status').textContent = 'No active analysis. Previous visuals, if present, are inactive. Retry or choose offline demo.';
  } finally {
    if (generation === state.generation) busy(false);
  }
}
function renderSummary() {
  const {model, predictions, crop, elapsed_ms, warning, source, original_size, entropy_bits} = state.data;
  $('metric-params').textContent = compact(model.parameters);
  $('metric-macs').textContent = compact(model.macs);
  $('metric-layers').textContent = model.layers.length;
  $('metric-time').textContent = `${elapsed_ms.toFixed(0)} ms`;
  $('metric-device').textContent = `${model.device.toUpperCase()} · one instrumented run`;
  $('runtime-badge').innerHTML = `<i></i> ${esc(model.device.toUpperCase())} · ${model.trained ? 'Trained model' : 'Untrained demo'}`;
  $('demo-warning').hidden = !warning;
  $('demo-warning').textContent = warning || '';
  $('input-preview').hidden = false;
  $('input-image').src = crop;
  $('input-caption').textContent = `${source} · ${original_size.join(' × ')} → resize ${model.preprocessing.resize[0]} → center crop 224.`;
  $('prediction-status').textContent = `Entropy ${entropy_bits.toFixed(2)} bits`;
  $('prediction-title').textContent = model.trained ? 'The model’s first impression' : 'Random-weight output · demonstration only';
  $('prediction-note').textContent = model.trained ? 'Top-five softmax scores. Not calibrated confidence.' : 'These scores are not meaningful predictions. Select trained weights for classification.';
  $('predictions').innerHTML = predictions.map(item => `<div class="prediction-item"><div class="label-row"><strong title="${esc(item.label)}">${esc(item.label)}</strong><span class="score">${(item.probability * 100).toFixed(2)}%</span></div><div class="bar" role="img" aria-label="${esc(item.label)}: ${(item.probability * 100).toFixed(2)} percent"><span style="width:${item.probability * 100}%"></span></div></div>`).join('');
}
function renderArchitecture() {
  const model = state.data.model;
  $('architecture-body').innerHTML = model.layers.map(layer => `<tr><td><button data-layer="${esc(layer.name)}">${esc(layer.name)} ↗</button></td><td><span class="tag">${esc(layer.kind)}</span></td><td>${layer.kernel.join('×')} / ${layer.stride.join('×')}</td><td>${layer.output.join(' × ')}</td><td>${compact(layer.parameters)}</td><td>${compact(layer.macs)}</td></tr>`).join('');
  $('stage-track').innerHTML = model.stages.map(stage => `<button class="stage ${stage.residual ? 'residual' : ''}" data-stage="${stage.index}" title="Stage ${stage.index}: ${stage.output.join(' × ')}; ${stage.residual ? 'residual connection' : 'no residual'}" aria-label="Explore stage ${stage.index}"><span class="block" style="height:${25 + Math.log2(stage.output[0]) * 2}px"></span><span class="stage-index">${String(stage.index).padStart(2,'0')}</span></button>`).join('');
  $('layer-select').replaceChildren(...model.layers.map((layer, index) => {
    const option = document.createElement('option'); option.value = layer.name;
    option.textContent = `${String(index + 1).padStart(2,'0')} · ${layer.name}`;
    return option;
  }));
}
function renderLayerControls() {
  if (!state.data) return;
  const layers = state.data.model.layers;
  const index = layers.findIndex(layer => layer.name === state.layer);
  const layer = layers[index];
  $('layer-select').value = state.layer;
  $('prev-layer').disabled = index <= 0; $('next-layer').disabled = index >= layers.length - 1;
  $('layer-tags').innerHTML = `<span class="tag accent">${esc(layer.kind)}</span><span class="tag mono">${layer.kernel.join('×')} / s${layer.stride[0]}</span><span class="tag mono">${layer.channels_in} → ${layer.channels_out}</span>`;
  $$('.stage').forEach(button => {
    const active = Number(button.dataset.stage) === layer.stage;
    button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  });
  $('filter-input').replaceChildren(...Array.from({length: layer.channels_in / layer.groups}, (_, i) => {
    const option = document.createElement('option'); option.value = String(i); option.textContent = `Channel ${i}`; return option;
  }));
  $('filter-input').disabled = layer.channels_in / layer.groups === 1;
}
function selectLayer(name) { if (!state.data) return; state.layer = name; state.page = state.filterPage = 0; renderLayerControls(); renderView(); }
function selectTab(tab) {
  state.tab = tab;
  $$('[role=tab]').forEach(button => {
    const active = button.dataset.tab === tab;
    button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1;
  });
  $$('[role=tabpanel]').forEach(panel => panel.hidden = panel.id !== `panel-${tab}`);
  $('layer-controls').hidden = tab === 'explain' || tab === 'architecture';
  $('stage-track').hidden = tab === 'explain';
  renderView();
}
function renderChannels(data, target, kind) {
  const images = {};
  $(target).innerHTML = data.channels.map(channel => {
    images[channel.index] = channel;
    return `<button class="channel-card" data-channel="${channel.index}" data-kind="${kind}" aria-label="Inspect ${kind} channel ${channel.index}"><img src="${channel.image}" alt="${kind} channel ${channel.index}" loading="lazy"><span class="channel-meta"><strong>C${String(channel.index).padStart(3,'0')}</strong><span>${channel.energy.toPrecision(2)}</span></span></button>`;
  }).join('');
  if (kind === 'activation') state.channelImages = images; else state.filterImages = images;
}
async function renderView() {
  const generation = ++state.viewGeneration;
  if (!state.data) return;
  const currentId = state.data.id;
  clearError();
  if (state.tab === 'architecture' || state.tab === 'explain') {
    busy(false, state.tab === 'architecture' ? `${state.data.model.layers.length} convolutions · ${state.data.model.stages.length} feature stages · live model topology` : 'Choose any class to generate a class-specific explanation.');
    return;
  }
  busy(true, `Inspecting ${state.layer}…`);
  try {
    if (state.tab === 'activations') {
      const data = await api(endpoint('activations', {layer:state.layer, page:state.page, sort:$('sort').value, normalization:$('normalization').value, palette:$('palette').value}));
      if (generation !== state.viewGeneration || currentId !== state.data?.id) return;
      renderChannels(data, 'activation-grid', 'activation');
      $('activation-shape').textContent = data.shape.join(' × ');
      $('channel-count').textContent = `${data.total} channels · raw range ${number(data.range[0])} to ${number(data.range[1])}`;
      $('page-label').textContent = `${data.page + 1} / ${data.pages}`;
      $('prev-page').disabled = data.page === 0; $('next-page').disabled = data.page + 1 === data.pages;
      busy(false, `${state.layer} · ${data.shape.join(' × ')} · showing ${data.channels.length} of ${data.total} channels`);
    } else {
      const data = await api(endpoint('filters', {layer:state.layer, page:state.filterPage, input_channel:$('filter-input').value, palette:$('palette').value}));
      if (generation !== state.viewGeneration || currentId !== state.data?.id) return;
      const pointwise = data.shape[1] === 1 && data.shape[2] === 1;
      $('mixing-view').hidden = !pointwise; $('filter-grid').hidden = pointwise;
      if (pointwise) {
        $('mixing-image').src = data.matrix;
        $('mixing-caption').textContent = `${data.matrix_shape[0]} output × ${data.matrix_shape[1]} input channels · weights ${number(data.matrix_range[0])} to ${number(data.matrix_range[1])}`;
      } else renderChannels(data, 'filter-grid', 'filter');
      $('filter-count').textContent = pointwise ? 'Complete channel-mixing matrix' : `${data.total} output filters · weights ${number(data.range[0])} to ${number(data.range[1])}`;
      $('filter-page-label').textContent = pointwise ? 'All' : `${data.page + 1} / ${data.pages}`;
      $('prev-filter-page').disabled = pointwise || data.page === 0; $('next-filter-page').disabled = pointwise || data.page + 1 === data.pages;
      busy(false, `${state.layer} · ${pointwise ? 'pointwise channel mixing' : 'spatial kernel slices'} · raw, signed weights`);
    }
  } catch (exception) {
    if (generation === state.viewGeneration) { error(exception.message); busy(false, 'Could not load this view. Check the session and retry.'); }
  }
}
function inspect(channel, kind) {
  $('dialog-title').textContent = `${kind === 'activation' ? 'Activation' : 'Filter'} / C${String(channel.index).padStart(3,'0')}`;
  $('dialog-image').src = channel.image;
  $('dialog-stats').innerHTML = [['Minimum',channel.min],['Maximum',channel.max],['Mean',channel.mean],['Std. deviation',channel.std],['Mean |value|',channel.energy],['Near-zero fraction',channel.zero_fraction]].map(([label,value]) => `<div><span>${label}</span><strong>${number(value)}</strong></div>`).join('');
  $('channel-download').dataset.name = `${kind}-${channel.index}.png`;
  $('channel-dialog').showModal();
}
function classLabel(index) { return `${index} · ${state.classes[index] || state.data?.predictions.find(item => item.index === index)?.label || ''}`; }
async function explain() {
  if (!state.data) return;
  const match = $('class-target').value.match(/^(\d{1,3})(?:\s|$)/);
  if (!match || Number(match[1]) > 999) { error('Choose a target from the ImageNet class suggestions (index 0–999).'); return; }
  const currentId = state.data.id;
  const generation = state.generation;
  $('cam-button').disabled = true;
  clearError(); busy(true, 'Computing the class gradient and spatial attribution…');
  try {
    const data = await api(endpoint('cam', {target:Number(match[1]), alpha:$('cam-alpha').value}), {method:'POST'});
    if (generation !== state.generation || currentId !== state.data?.id) return;
    $('cam-results').hidden = false;
    $('cam-input').src = state.data.crop; $('cam-overlay').src = data.overlay; $('cam-heatmap').src = data.heatmap;
    $('cam-caption').textContent = `${data.label} · ${data.trained ? 'Grad-CAM' : 'untrained demonstration'}`;
    $('cam-note').textContent = data.nonzero ? 'Warm regions have larger positive, gradient-weighted contributions for this class.' : 'No positive attribution was found for this class. The overlay is unchanged; this is a valid result.';
    $('cam-download').hidden = false;
    busy(false, `Explanation for class ${data.class_index}: ${data.label}`);
  } catch (exception) { if (generation === state.generation) { error(exception.message); busy(false); } }
  finally { if (generation === state.generation && state.data) $('cam-button').disabled = false; }
}
function downloadUrl(url, name) {
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = name;
  document.body.appendChild(anchor); anchor.click(); anchor.remove();
}
async function exportAnalysis() {
  if (!state.data) return;
  const currentId = state.data.id;
  $('export-button').disabled = true; clearError();
  try {
    const response = await fetch(endpoint('export', {layer:state.layer}));
    if (!response.ok) { const data = await response.json(); throw new Error(data.detail || 'Export failed.'); }
    const url = URL.createObjectURL(await response.blob());
    downloadUrl(url, 'mobilenet-analysis.zip'); setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (exception) { error(exception.message); }
  finally { if (currentId === state.data?.id) $('export-button').disabled = false; }
}
function chooseFile(file) {
  if (!file) return;
  if (file.size === 0 || file.size > 10 * 1024 * 1024) { error('Choose a non-empty image smaller than 10 MiB.'); return; }
  state.file = file; $$('.sample').forEach(button => {button.classList.remove('active'); button.setAttribute('aria-pressed','false');}); analyze();
}
function updateSidebarAccess() { $('sidebar').inert = window.matchMedia('(max-width:700px)').matches && !$('sidebar').classList.contains('open'); }
function closeMenu() { $('sidebar').classList.remove('open'); $('scrim').hidden = true; $('menu-button').setAttribute('aria-expanded','false'); updateSidebarAccess(); }
$('menu-button').addEventListener('click', () => { const open = !$('sidebar').classList.contains('open'); $('sidebar').classList.toggle('open',open); $('scrim').hidden = !open; $('menu-button').setAttribute('aria-expanded',String(open)); updateSidebarAccess(); });
$('scrim').addEventListener('click',closeMenu);
$('image-upload').addEventListener('change', event => chooseFile(event.target.files[0]));
$('dropzone').addEventListener('keydown', event => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); $('image-upload').click();}});
['dragenter','dragover'].forEach(type => $('dropzone').addEventListener(type,event => {event.preventDefault();$('dropzone').classList.add('dragover');}));
['dragleave','drop'].forEach(type => $('dropzone').addEventListener(type,event => {event.preventDefault();$('dropzone').classList.remove('dragover');}));
$('dropzone').addEventListener('drop', event => chooseFile(event.dataTransfer.files[0]));
$$('.sample').forEach(button => button.addEventListener('click', () => {state.sample = button.dataset.sample; state.file = null; $('image-upload').value = ''; $$('.sample').forEach(b => {b.classList.toggle('active',b === button); b.setAttribute('aria-pressed',String(b === button));}); analyze();}));
$('weights').addEventListener('change',analyze);
$('offline-button').addEventListener('click',() => {$('weights').value = 'untrained';analyze();});
$('layer-select').addEventListener('change',event => selectLayer(event.target.value));
['prev-layer','next-layer'].forEach((id,i) => $(id).addEventListener('click',() => {if (!state.data) return; const layers = state.data.model.layers; const index = layers.findIndex(l => l.name === state.layer) + (i ? 1 : -1); if (layers[index]) selectLayer(layers[index].name);}));
$('stage-track').addEventListener('click',event => {const button = event.target.closest('[data-stage]'); if (!button || !state.data) return; const layer = state.data.model.layers.find(l => l.stage === Number(button.dataset.stage)); if (state.tab === 'architecture') selectTab('activations'); selectLayer(layer.name);});
$('architecture-body').addEventListener('click',event => {const button = event.target.closest('[data-layer]');if (!button) return;selectTab('activations');selectLayer(button.dataset.layer);});
$$('[role=tab]').forEach(button => button.addEventListener('click',() => selectTab(button.dataset.tab)));
document.querySelector('.tablist').addEventListener('keydown',event => {
  const tabs = $$('[role=tab]'); const index = tabs.indexOf(document.activeElement);
  if (index < 0 || !['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
  event.preventDefault();
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  tabs[next].focus(); selectTab(tabs[next].dataset.tab);
});
['sort','normalization','palette'].forEach(id => $(id).addEventListener('change',() => {state.page = 0;renderView();}));
$('filter-input').addEventListener('change',() => {state.filterPage = 0;renderView();});
['prev-page','next-page'].forEach((id,i) => $(id).addEventListener('click',() => {state.page += i ? 1 : -1;renderView();}));
['prev-filter-page','next-filter-page'].forEach((id,i) => $(id).addEventListener('click',() => {state.filterPage += i ? 1 : -1;renderView();}));
['activation-grid','filter-grid'].forEach(id => $(id).addEventListener('click',event => {const card = event.target.closest('[data-channel]');if (!card || !state.data) return; const channel = (card.dataset.kind === 'activation' ? state.channelImages : state.filterImages)[card.dataset.channel]; if (channel) inspect(channel,card.dataset.kind);}));
$('close-dialog').addEventListener('click',() => $('channel-dialog').close());
$('channel-dialog').addEventListener('click',event => {if (event.target === $('channel-dialog')) {const r = $('channel-dialog').getBoundingClientRect();if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) $('channel-dialog').close();}});
$('channel-download').addEventListener('click',() => downloadUrl($('dialog-image').src,$('channel-download').dataset.name));
$('enlarge-input').addEventListener('click',() => {if (!state.data) return; $('dialog-title').textContent = 'Model input / 224 × 224'; $('dialog-image').src = state.data.crop; $('dialog-stats').replaceChildren(); $('channel-download').dataset.name = 'model-input.png';$('channel-dialog').showModal();});
$('cam-button').addEventListener('click',explain);
$('cam-download').addEventListener('click',() => downloadUrl($('cam-overlay').src,'gradcam-overlay.png'));
$('export-button').addEventListener('click',exportAnalysis);
function resetVisuals() {
  ['metric-params','metric-macs','metric-layers','metric-time'].forEach(id => $(id).textContent = '—');
  $('input-preview').hidden = true; $('input-image').removeAttribute('src');
  $('predictions').innerHTML = '<p class="muted small">Analyze an image to reveal its class scores.</p>';
  $('activation-grid').innerHTML = '<div class="empty-state"><span>⌘</span><h3>A new perspective awaits.</h3><p>Select an image or diagnostic to begin exploring.</p></div>';
  ['filter-grid','stage-track','architecture-body','layer-tags','dialog-stats'].forEach(id => $(id).replaceChildren());
  ['activation-shape','channel-count','filter-count','cam-note'].forEach(id => $(id).textContent = '');
  ['page-label','filter-page-label'].forEach(id => $(id).textContent = '—');
  ['mixing-view','cam-results','cam-download'].forEach(id => $(id).hidden = true);
  ['cam-input','cam-overlay','cam-heatmap','mixing-image','dialog-image'].forEach(id => $(id).removeAttribute('src'));
  $('channel-dialog').close();
  state.channelImages = {}; state.filterImages = {};
}
$('clear-session').addEventListener('click',async () => {
  const id = state.data?.id;
  ++state.generation; ++state.viewGeneration;
  state.data = null; state.file = null;
  $('image-upload').value = ''; resetVisuals();
  enableAnalysisControls(false); clearError(); $('demo-warning').hidden = true;
  $('prediction-status').textContent = 'Session cleared';
  busy(false,'Image and session cleared. Choose an input to start a new analysis.');
  if (id) {try {await api(`/api/analysis/${id}`,{method:'DELETE'});} catch (_) {error('The browser session was cleared, but the server could not be reached. Its temporary session will expire automatically.');}}
});
window.addEventListener('resize', updateSidebarAccess);
document.addEventListener('keydown', event => {if (event.key === 'Escape' && $('sidebar').classList.contains('open')) {closeMenu();$('menu-button').focus();}});
updateSidebarAccess();
async function init() {
  try {
    const [health,classes] = await Promise.all([api('/api/health'),api('/api/classes')]);
    state.classes = classes;
    if ([...$('weights').options].some(option => option.value === health.default_weights)) $('weights').value = health.default_weights;
    $('class-options').replaceChildren(...classes.map((label,index) => {const option = document.createElement('option');option.value = `${index} · ${label}`;return option;}));
    await analyze();
  } catch (exception) {error(`Cannot connect to the local Python server. ${exception.message}`);busy(false,'Start the application with: python interface.py');}
}
init();
