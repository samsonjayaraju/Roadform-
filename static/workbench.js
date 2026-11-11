const byId = (id) => document.getElementById(id);
const canvas = byId('canvas');
const video = byId('video');
const slider = byId('confidence');
let file = null;
let sourceUrl = '';
let resultUrl = '';
let sourceImage = null;
let boxes = [];
let clipStats = null;
let isClip = false;
let showOverlay = true;
let ready = false;
let working = false;

function message(text = '') {
  byId('notice').textContent = text;
  byId('notice').hidden = !text;
}

function setWorking(text = '') {
  working = Boolean(text);
  byId('busy').hidden = !working;
  byId('busy').textContent = text;
  byId('uploadBtn').disabled = working;
  byId('reset').disabled = working;
  byId('runClip').disabled = working;
  slider.disabled = working;
}

function visibleBoxes() {
  return boxes.filter(box => box[4] >= Number(slider.value) / 100);
}

function paint(target, overlay) {
  if (!sourceImage) return;
  const width = sourceImage.naturalWidth;
  const height = sourceImage.naturalHeight;
  if (target.width !== width || target.height !== height) {
    target.width = width;
    target.height = height;
  }
  const ctx = target.getContext('2d');
  ctx.drawImage(sourceImage, 0, 0);
  if (!overlay) return;
  const fontSize = Math.max(14, width / 58, height / 32);
  const padding = Math.ceil(fontSize / 5);
  ctx.lineWidth = Math.max(2, fontSize / 8);
  ctx.font = 'bold ' + fontSize + 'px system-ui, sans-serif';
  const hits = visibleBoxes();
  for (const [left, top, right, bottom] of hits) {
    ctx.fillStyle = 'rgba(230,31,50,.14)';
    ctx.fillRect(left, top, right - left, bottom - top);
    ctx.strokeStyle = '#e61f32';
    ctx.strokeRect(left, top, right - left, bottom - top);
  }
  const occupied = [];
  for (const [left, top, , bottom, score] of hits) {
    const label = Math.round(score * 100) + '%';
    const w = ctx.measureText(label).width + 2 * padding;
    const h = fontSize + 2 * padding;
    const x = Math.max(0, Math.min(left, width - w));
    const candidates = [top - h, bottom, top].map(y => Math.max(0, Math.min(y, height - h)));
    const y = candidates.find(y => occupied.every(rect =>
      x + w <= rect.left || x >= rect.right || y + h <= rect.top || y >= rect.bottom
    )) ?? candidates[2];
    occupied.push({left:x, top:y, right:x + w, bottom:y + h});
    ctx.fillStyle = '#9c1125';
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = '#fff';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x + padding, y + h / 2);
  }
}

function note(text) {
  const p = document.createElement('p');
  p.className = 'note';
  p.textContent = text;
  byId('detectionList').append(p);
}

function render() {
  const list = byId('detectionList');
  list.replaceChildren();
  byId('countLabel').textContent = isClip ? 'Frame detections' : 'Potholes';
  byId('overlay').disabled = working || !ready;
  byId('original').disabled = working || !ready;
  byId('overlay').setAttribute('aria-pressed', String(showOverlay));
  byId('original').setAttribute('aria-pressed', String(!showOverlay));
  byId('save').disabled = !ready || working;
  byId('save').lastChild.textContent = isClip ? ' Export annotated clip' : ' Export annotated image';
  byId('overlayInfo').textContent = showOverlay && ready ? 'Detection overlay' : 'Original media';
  byId('runClip').hidden = !isClip || ready;
  byId('count').textContent = '—';
  byId('avg').textContent = '—';
  if (isClip) {
    const url = ready && showOverlay ? resultUrl : sourceUrl;
    if (url && video.getAttribute('src') !== url) video.src = url;
    if (clipStats) {
      byId('count').textContent = clipStats.count;
      byId('avg').textContent = clipStats.count ? Math.round(100 * clipStats.mean) + '%' : '—';
      note(clipStats.frames + ' frames inspected. A pothole can appear in multiple frames. The exported clip is silent.');
    } else if (!working) {
      note('Analyze this clip at the selected confidence threshold.');
    }
    return;
  }
  paint(canvas, showOverlay && ready);
  if (!ready) return;
  const hits = visibleBoxes();
  byId('count').textContent = hits.length;
  byId('avg').textContent = hits.length ? Math.round(hits.reduce((sum, b) => sum + b[4], 0) * 100 / hits.length) + '%' : '—';
  if (!hits.length) note('No potholes meet this confidence threshold.');
  hits.forEach((box, index) => {
    const row = document.createElement('div');
    row.className = 'detection';
    const tag = document.createElement('code');
    tag.textContent = String(index + 1).padStart(2, '0');
    const name = document.createElement('span');
    name.textContent = 'Pothole';
    const bar = document.createElement('span');
    bar.className = 'bar';
    const fill = document.createElement('i');
    const score = Math.round(box[4] * 100) + '%';
    fill.style.width = score;
    bar.append(fill);
    const value = document.createElement('small');
    value.textContent = score;
    row.append(tag, name, bar, value);
    list.append(row);
  });
}

async function checked(response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || 'Inspection could not finish. Try the file again.');
  }
  return response;
}

function payload() {
  const data = new FormData();
  data.append('media', file);
  data.append('confidence', Number(slider.value) / 100);
  return data;
}

async function openMedia(selected) {
  if (working) return;
  message();
  if (selected && (!/\.(jpe?g|png|webp|mp4|mov)$/i.test(selected.name) || selected.size >= 100 * 1024 * 1024)) {
    message('Choose a JPG, PNG, WebP, MP4, or MOV file smaller than 100 MB.');
    return;
  }
  video.pause();
  video.removeAttribute('src');
  video.load();
  if (file) URL.revokeObjectURL(sourceUrl);
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  file = selected;
  sourceUrl = file ? URL.createObjectURL(file) : '/static/sample-road.jpg';
  isClip = Boolean(file && /\.(mp4|mov)$/i.test(file.name));
  byId('stage').classList.remove('portrait');
  ready = false;
  showOverlay = !isClip;
  sourceImage = null;
  resultUrl = '';
  boxes = [];
  clipStats = null;
  canvas.width = 1;
  canvas.hidden = isClip;
  video.hidden = !isClip;
  byId('sampleTag').hidden = Boolean(file);
  byId('credit').hidden = Boolean(file);
  byId('reset').hidden = !file;
  byId('filename').textContent = file ? file.name : 'asphalt-sample.jpg';
  byId('dimensions').textContent = isClip ? 'Road video' : 'Road image';
  if (isClip) { render(); return; }
  setWorking('Inspecting image…');
  render();
  try {
    const img = new Image();
    img.src = sourceUrl;
    const response = file ? fetch('/api/inspect', {method:'POST', body:payload()}) : fetch('/api/sample');
    const [result] = await Promise.all([response.then(checked).then(r => r.json()), img.decode()]);
    sourceImage = img;
    byId('stage').classList.toggle('portrait', img.naturalHeight > img.naturalWidth);
    boxes = result.boxes;
    byId('dimensions').textContent = result.width + ' × ' + result.height + ' px';
    ready = true;
  } catch (error) {
    message(error.name === 'EncodingError' ? 'This image could not be opened. Try another road image.' : error.message);
  } finally {
    setWorking();
    render();
  }
}

async function analyzeClip() {
  setWorking('Inspecting video frames…');
  message();
  render();
  try {
    const response = await checked(await fetch('/api/inspect', {method:'POST', body:payload()}));
    const blob = await response.blob();
    resultUrl = URL.createObjectURL(blob);
    clipStats = {count:Number(response.headers.get('X-Detections')), mean:Number(response.headers.get('X-Mean-Confidence')), frames:Number(response.headers.get('X-Frames'))};
    byId('dimensions').textContent = response.headers.get('X-Width') + ' × ' + response.headers.get('X-Height') + ' px';
    ready = true;
    showOverlay = true;
  } catch (error) {
    message(error.message);
  } finally {
    setWorking();
    render();
  }
}

video.addEventListener('loadedmetadata', () => {
  if (isClip) byId('stage').classList.toggle('portrait', video.videoHeight > video.videoWidth);
});
byId('uploadBtn').onclick = () => byId('upload').click();
byId('upload').onchange = event => {
  const selected = event.target.files[0];
  event.target.value = '';
  if (selected) openMedia(selected);
};
byId('reset').onclick = () => openMedia(null);
byId('runClip').onclick = analyzeClip;
byId('overlay').onclick = () => { showOverlay = true; render(); };
byId('original').onclick = () => { showOverlay = false; render(); };
slider.oninput = () => {
  byId('confValue').textContent = slider.value + '%';
  slider.style.setProperty('--fill', ((Number(slider.value) - 10) / 90 * 100) + '%');
  if (isClip && ready) {
    ready = false;
    showOverlay = false;
    clipStats = null;
    URL.revokeObjectURL(resultUrl);
    resultUrl = '';
  }
  render();
};
byId('full').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await byId('stage').requestFullscreen();
  } catch { message('Full-screen view is unavailable in this browser.'); }
};
byId('save').onclick = async () => {
  const link = document.createElement('a');
  const name = (file?.name || 'asphalt-sample.jpg').replace(/\.[^.]+$/, '');
  link.download = name + '-inspected.' + (isClip ? 'mp4' : 'png');
  if (isClip) link.href = resultUrl;
  else {
    const output = document.createElement('canvas');
    paint(output, true);
    const blob = await new Promise(resolve => output.toBlob(resolve, 'image/png'));
    if (!blob) { message('This image could not be exported.'); return; }
    link.href = URL.createObjectURL(blob);
    setTimeout(() => URL.revokeObjectURL(link.href), 30000);
  }
  link.click();
};
openMedia(null);
